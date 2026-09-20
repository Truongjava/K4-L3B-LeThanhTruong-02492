#!/usr/bin/env python3
"""Benchmark harness for Lab 07 (K4-L3B) — Shopee return/refund policies.

Four jobs, per the lab doc: read every corpus file and split its front matter
from its body, chunk the body, load the chunks into an EmbeddingStore, then run
the team's five benchmark queries and print the top-3 so they can be checked
against the gold answers.

Scoring follows docs/SCORING.md at two levels, because checking the doc_id alone
inflates the result: a strategy can fill all three top-3 slots from the right
file while none of those chunks contains the answer. A query scores 2 only when
the chunk that *carries the evidence* ranks first, 1 when it ranks 2-3, else 0.

Everything below the CHUNKER line is deliberately identical for every team
member. Each person swaps in their own strategy there and nothing else, so the
comparison measures the strategy rather than the harness.

Run:
    python bench.py                      # all five queries
    python bench.py --query 3            # just one
    python bench.py --provider lexical   # offline control, no model needed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.chunking import HeadingChunker, RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

CORPUS_DIR = Path("data/shopee-doi-tra-hoan-tien")
OUTPUT_FILE = Path("ket_qua_benchmark.txt")
CACHE_FILE = Path(".cache/embeddings.json")

# ============================================================================
# MỖI THÀNH VIÊN ĐI ĐÚNG DÒNG DƯỚI ĐÂY sang chiến lược của mình.
# Ví dụ:  CHUNKER = FixedSizeChunker(chunk_size=500, overlap=50)
#         CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
#         CHUNKER = HeadingChunker(max_section_size=500)
# Đừng sửa gì khác — cả nhóm phải chạy trên cùng một harness.
# ============================================================================
CHUNKER = HeadingChunker(max_section_size=800)

# Five team queries. `marker` is a distinctive string that must appear in the
# retrieved context for the answer to be derivable at all. It is checked
# together with `doc`, so a chunk from the right file that does not actually
# answer the question does not earn the point.
QUERIES = [
    {
        "id": 1,
        "kind": "số liệu",
        "question": "Đối với đơn hàng giao thực phẩm tươi sống và đông lạnh, người mua có bao nhiêu thời gian để gửi yêu cầu Trả hàng/Hoàn tiền?",
        "gold": "24 giờ kể từ lúc đơn hàng được cập nhật trạng thái 'Giao hàng thành công' (trừ lý do Chưa nhận được hàng).",
        "marker": "24 giờ",
        "doc": "quy-dinh-chung-tra-hang-hoan-tien",
        "filter": None,
    },
    {
        "id": 2,
        "kind": "điều kiện + mốc thời gian",
        "question": "Tính năng Trả hàng COM (đổi ý) áp dụng cho những nhóm người mua nào và từ thời điểm nào?",
        "gold": "Từ ngày 24/11/2025, áp dụng cho người mua hạng Kim Cương, Vàng và người dùng đăng ký thành công gói Shopee VIP.",
        "marker": "24/11/2025",
        "doc": "quy-dinh-chung-tra-hang-hoan-tien",
        "filter": None,
    },
    {
        "id": 3,
        "kind": "điều kiện — CẦN metadata_filter",
        "question": "Chi phí vận chuyển chiều hoàn trả sản phẩm do bên nào chịu khi đơn hàng được Shopee chấp thuận trả hàng/hoàn tiền?",
        "gold": "Người Bán chịu chi phí vận chuyển chiều hoàn trả đối với đơn được Shopee chấp thuận mà KHÔNG do lỗi của Người Mua hoặc đơn vị vận chuyển, và đối với đơn giao không thành công.",
        "marker": "không do lỗi của Người Mua hoặc đơn vị vận chuyển",
        "doc": "chinh-sach-tra-hang-hoan-tien-nguoi-ban",
        "filter": {"audience": "seller"},
    },
    {
        "id": 4,
        "kind": "quy trình + số liệu",
        "question": "Mã giảm giá được hoàn lại trong vòng bao lâu sau khi yêu cầu Trả hàng/Hoàn tiền được chấp nhận hoàn tiền?",
        "gold": "Trong vòng 48 giờ (không kể Thứ Bảy, Chủ Nhật và ngày lễ), tính từ khi yêu cầu được chấp nhận hoàn tiền.",
        "marker": "48 giờ",
        "doc": "quy-dinh-chung-tra-hang-hoan-tien",
        "filter": None,
    },
    {
        "id": 5,
        "kind": "liệt kê",
        "question": "Khi đóng gói hàng hoàn trả, người mua cần chuẩn bị những vật liệu gì và phải lưu ý điều gì?",
        "gold": "Chuẩn bị vật liệu đóng gói (hộp carton/bao bì, băng dính, vật liệu chèn) và phiếu gửi hàng; phải quay video quá trình đóng gói; dùng thêm hộp vận chuyển bên ngoài và không dán/viết lên hộp của nhà sản xuất.",
        "marker": "hộp vận chuyển bên ngoài",
        "doc": "cach-dong-goi-don-hoan-tra",
        "filter": None,
    },
]


def normalize_text(text: str) -> str:
    """NFC + lowercase + collapse whitespace, so matching survives NFD input.

    Vietnamese arrives in two Unicode forms: composed ("nay" with one accented
    code point) and decomposed (base letter plus a combining mark). A marker
    typed normally will not match a decomposed corpus, and the failure looks
    like a retrieval miss when it is really an encoding mismatch.
    """
    text = unicodedata.normalize("NFC", text).lower()
    return " ".join(text.split())


class LexicalHashEmbedder:
    """Offline lexical baseline: word and word-bigram features, hashed.

    This is the control condition. It needs no model, no network and no API key,
    so it answers "how much does a real embedder actually buy us?" -- with the
    mock embedder you only learn that hashing is bad, not how far the real
    embedder sits above plain lexical overlap.
    """

    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self._backend_name = f"lexical-hash-{dim}"

    def __call__(self, text: str) -> list[float]:
        words = re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vector = [0.0] * self.dim
        for token, count in Counter(features).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class CachedEmbedder:
    """Memoise embeddings across runs.

    Re-embedding an unchanged corpus is pure waste: costly against a paid API,
    slow against a local model. The cache key includes the backend name, so
    switching providers can never serve stale vectors from another model.
    """

    def __init__(self, embedder, path: Path, enabled: bool = True) -> None:
        self._embedder = embedder
        self._path = path
        self._enabled = enabled
        self._backend_name = getattr(embedder, "_backend_name", type(embedder).__name__)
        self._cache: dict[str, list[float]] = {}
        if enabled and path.exists():
            try:
                self._cache = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def __call__(self, text: str) -> list[float]:
        if not self._enabled:
            return self._embedder(text)
        key = hashlib.sha256(f"{self._backend_name}\0{text}".encode("utf-8")).hexdigest()
        if key not in self._cache:
            self._cache[key] = self._embedder(text)
        return self._cache[key]

    def save(self) -> None:
        if not self._enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._cache), encoding="utf-8")


def read_document(path: Path) -> Document:
    """Split one corpus file into front-matter metadata and body content.

    DATA_COLLECTION.md: the front matter becomes `metadata`, only what sits
    below it becomes `content`. Leaving the YAML block in the body would mean
    chunking and embedding the metadata too.
    """
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                match = re.match(r"^(\w+):\s*(.*)$", line)
                if match:
                    metadata[match.group(1)] = match.group(2).strip().strip('"')
            body = text[end + len("\n---\n") :]
    return Document(id=path.stem, content=body, metadata=metadata)


def build_store(embedder) -> tuple[EmbeddingStore, int, int]:
    """Chunk every corpus file and load the chunks into a fresh store."""
    store = EmbeddingStore(collection_name="bench", embedding_fn=embedder)
    files = chunks = 0
    for path in sorted(CORPUS_DIR.glob("*.md")):
        document = read_document(path)
        # Chunking happens here, outside the store: one chunk becomes one
        # Document. Document.id identifies the chunk, while metadata["doc_id"]
        # keeps pointing at the source file so delete_document() and the
        # per-file reporting below still work.
        for index, chunk in enumerate(CHUNKER.chunk(document.content)):
            store.add_documents(
                [
                    Document(
                        id=f"{path.stem}#{index}",
                        content=chunk,
                        metadata={**document.metadata, "doc_id": path.stem},
                    )
                ]
            )
            chunks += 1
        files += 1
    return store, files, chunks


def run_query(store: EmbeddingStore, query: dict, use_filter: bool) -> list[dict]:
    metadata_filter = query["filter"] if use_filter else None
    return store.search_with_filter(query["question"], top_k=3, metadata_filter=metadata_filter)


def carries_evidence(result: dict, query: dict) -> bool:
    """Right file AND the answer itself -- the doc_id alone is not enough."""
    if result["metadata"].get("doc_id") != query["doc"]:
        return False
    return normalize_text(query["marker"]) in normalize_text(result["content"])


def evaluate(results: list[dict], query: dict) -> dict:
    """Score one query at two levels, per docs/SCORING.md."""
    rank = next(
        (index for index, result in enumerate(results, 1) if carries_evidence(result, query)),
        None,
    )
    return {
        "doc_in": any(r["metadata"].get("doc_id") == query["doc"] for r in results),
        "marker_in": any(
            normalize_text(query["marker"]) in normalize_text(r["content"]) for r in results
        ),
        "evidence_rank": rank,
        "score": 2 if rank == 1 else 1 if rank in (2, 3) else 0,
    }


def format_results(results: list[dict], query: dict) -> list[str]:
    lines = []
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        hit = "✓" if carries_evidence(result, query) else " "
        preview = " ".join(result["content"].split())[:88]
        lines.append(
            f"  {rank}. score={result['score']:+.3f}  doc_id={str(metadata.get('doc_id', '?')):42} "
            f"audience={str(metadata.get('audience', '?')):7} evidence=[{hit}]"
        )
        lines.append(f"       {preview}...")
    return lines


def make_embedder(provider: str) -> CachedEmbedder:
    """Pick the backend, falling back to the mock on any failure.

    Cache only the backends that are expensive to recompute; the mock and the
    lexical baseline are effectively free.
    """
    cacheable = provider in {"local", "openai", "gemini"}
    try:
        if provider == "local":
            backend = LocalEmbedder(os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        elif provider == "openai":
            backend = OpenAIEmbedder(os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        elif provider == "gemini":
            backend = GeminiEmbedder(os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        elif provider == "lexical":
            backend = LexicalHashEmbedder()
        else:
            backend = _mock_embed
    except Exception as error:  # missing key, missing package, no network
        print(f"[warn] embedder '{provider}' không dùng được ({error}); quay về mock")
        backend, cacheable = _mock_embed, False
    return CachedEmbedder(backend, CACHE_FILE, enabled=cacheable)


def main() -> int:
    # Load .env before building the parser: the --provider default reads
    # EMBEDDING_PROVIDER, so loading it later would silently ignore .env.
    load_dotenv(override=False)

    parser = argparse.ArgumentParser(description="Run the team benchmark.")
    parser.add_argument("--query", type=int, help="Run only this query id")
    parser.add_argument(
        "--provider",
        default=os.getenv(EMBEDDING_PROVIDER_ENV, "local"),
        choices=("local", "openai", "gemini", "lexical", "mock"),
        help="Embedding backend (default: EMBEDDING_PROVIDER from .env, else local)",
    )
    args = parser.parse_args()

    embedder = make_embedder(args.provider)
    store, files, chunks = build_store(embedder)

    output: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        output.append(line)

    def dump(results: list[dict], query: dict) -> None:
        for line in format_results(results, query):
            emit(line)

    emit(f"=== BENCH — chiến lược: {type(CHUNKER).__name__} ===")
    emit(f"Corpus   : {CORPUS_DIR}")
    emit(f"Files    : {files}  →  chunks: {chunks}")
    emit(f"Embedder : {getattr(embedder, '_backend_name', type(embedder).__name__)}")
    if args.provider == "mock":
        emit("           ⚠ MockEmbedder băm MD5 nên không mã hoá ngữ nghĩa — số liệu")
        emit("             dưới đây là nhiễu. Dùng --provider lexical hoặc local để so sánh.")
    emit()

    judged: list[dict] = []

    for query in QUERIES:
        if args.query and query["id"] != args.query:
            continue
        results = run_query(store, query, use_filter=True)
        verdict = evaluate(results, query)
        judged.append((query, verdict))

        emit(f"--- Q{query['id']} · {query['kind']} ---")
        emit(f"Q      : {query['question']}")
        emit(f"Filter : {query['filter'] or '(không lọc)'}")
        emit(f"Gold   : {query['gold']}")
        emit(f"Marker : {query['marker']!r}")
        dump(results, query)
        emit(
            f"  → gold doc: {'CÓ' if verdict['doc_in'] else 'KHÔNG'}"
            f" · marker: {'CÓ' if verdict['marker_in'] else 'KHÔNG'}"
            f" · hạng chunk chứa đáp án: {verdict['evidence_rank'] or '—'}"
            f" · điểm: {verdict['score']}/2"
        )
        emit()

    # The A/B the lab doc requires for the filter query: same question with and
    # without metadata_filter. Identical output means the question is not yet
    # exercising the filter and needs rewording or a re-cut corpus.
    for query in QUERIES:
        if not query["filter"] or (args.query and query["id"] != args.query):
            continue
        without = run_query(store, query, use_filter=False)
        with_filter = run_query(store, query, use_filter=True)
        emit(f"=== A/B trên Q{query['id']} — metadata_filter có tác dụng không? ===")
        emit("  KHÔNG lọc:")
        dump(without, query)
        emit("  CÓ lọc:")
        dump(with_filter, query)
        same = [r["content"] for r in without] == [r["content"] for r in with_filter]
        emit(f"  → hai lượt giống hệt nhau: {'CÓ — câu hỏi CHƯA cần filter!' if same else 'KHÔNG — filter có tác dụng'}")
        emit()

    if judged:
        count = len(judged)
        hit1 = sum(1 for _, v in judged if v["evidence_rank"] == 1)
        hit3 = sum(1 for _, v in judged if v["evidence_rank"] is not None)
        mrr = sum(1 / v["evidence_rank"] for _, v in judged if v["evidence_rank"]) / count
        total = sum(v["score"] for _, v in judged)

        emit("=== TỔNG HỢP (theo docs/SCORING.md) ===")
        for query, verdict in judged:
            rank = verdict["evidence_rank"]
            emit(f"  Q{query['id']}  evidence_rank={rank if rank else '—':<2}"
                 f"  điểm={verdict['score']}/2   {query['kind']}")
        emit()
        emit(f"  Hit@1 = {hit1}/{count} ({hit1 / count:.0%})"
             f" · Hit@3 = {hit3}/{count} ({hit3 / count:.0%})"
             f" · MRR = {mrr:.3f}")
        emit(f"  lab_score = {total}/{2 * count}")

    embedder.save()

    OUTPUT_FILE.write_text("\n".join(output) + "\n", encoding="utf-8", newline="\n")
    print(f"\nĐã ghi {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())