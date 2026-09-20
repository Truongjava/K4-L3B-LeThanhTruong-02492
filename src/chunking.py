from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Split *after* the terminator, not on it: a plain re.split(r"[.!?]\s+")
        # consumes the punctuation and leaves every chunk a sentence stub.
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip())]
        sentences = [sentence for sentence in sentences if sentence]

        chunks: list[str] = []
        for start in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[start : start + self.max_sentences_per_chunk]
            joined = " ".join(group).strip()
            if joined:
                chunks.append(joined)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        pieces = [piece for piece in self._split(text, list(self.separators)) if piece.strip()]
        return self._consolidate(pieces)

    def _consolidate(self, chunks: list[str]) -> list[str]:
        """Absorb stub chunks into a neighbour when the two fit together.

        The ". " separator collides with legal clause numbering: Vietnamese
        regulations label items "a.", "b.", "2.1." and so on, so splitting on
        ". " strands the label while the clause body (too long to fit) recurses
        away. The leftover "b. " is 3 characters, matches nothing at retrieval
        time, and still consumes a top-k slot. Fold such stubs into whichever
        neighbour can hold them.
        """
        minimum = max(1, self.chunk_size // 4)
        merged: list[str] = []
        for chunk in chunks:
            fits = merged and len(merged[-1]) + len(chunk) <= self.chunk_size
            # Absorb a trailing stub into the chunk before it, or let a leading
            # stub take in the chunk after it.
            if fits and (len(chunk) < minimum or len(merged[-1]) < minimum):
                merged[-1] += chunk
            else:
                merged.append(chunk)
        return merged

    def _hard_cut(self, text: str) -> list[str]:
        """Last resort when no separator can help: slice by chunk_size."""
        return [
            text[start : start + self.chunk_size]
            for start in range(0, len(text), self.chunk_size)
        ]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case 1: the piece already fits.
        if len(current_text) <= self.chunk_size:
            return [current_text]
        # Base case 2: out of separators, so give up on boundaries and cut hard.
        if not remaining_separators:
            return self._hard_cut(current_text)

        separator, rest = remaining_separators[0], remaining_separators[1:]
        # Base case 3: no usable separator here -- either the empty-string
        # terminator, or one this piece simply does not contain.
        if not separator or separator not in current_text:
            return self._split(current_text, rest)

        pieces = current_text.split(separator)
        # Re-attach the separator to the piece it followed, so sentence and
        # paragraph boundaries survive into the chunk text.
        pieces = [piece + separator for piece in pieces[:-1]] + [pieces[-1]]

        # Two directions matter. Pieces too long recurse downward; pieces that
        # fit get merged upward until they nearly fill chunk_size. Skipping the
        # merge turns a many-short-lines file into hundreds of stub chunks.
        merged: list[str] = []
        buffer = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.extend(self._split(piece, rest))
            elif len(buffer) + len(piece) <= self.chunk_size:
                buffer += piece
            else:
                merged.append(buffer)
                buffer = piece
        if buffer:
            merged.append(buffer)
        return merged


class HeadingChunker:
    """Split a document along its own section headings.

    A regulation is already divided into sections by whoever drafted it, and
    each section is a complete semantic unit -- a far better boundary than a
    character count, which happily cuts a clause in half.

    A section longer than max_section_size falls back to RecursiveChunker. The
    heading is then prepended to *every* fragment, not just the first: without
    that, the second fragment onward loses all trace of which clause it belongs
    to and reads like an orphaned paragraph to the retriever.
    """

    # A heading is either a Markdown ATX heading or a numbered clause label
    # ("5.", "7.1."). Single-letter labels ("a.", "b.") are deliberately not
    # headings -- they are list items, and splitting on them shreds the corpus.
    HEADING = re.compile(r"^(?:#{1,6} \S|\d+(?:\.\d+)*\.\s+\S)")

    # A numbered line counts as a heading only while it stays short. Past this
    # it is a clause that merely opens with a number -- dieu-khoan-dich-vu-shopee
    # holds one of 1843 characters. Treating those as headings is catastrophic,
    # not merely untidy: the line becomes a repeated prefix on every fragment.
    MAX_HEADING_LENGTH = 200

    def __init__(self, max_section_size: int = 800) -> None:
        self.max_section_size = max_section_size

    def _is_heading(self, line: str) -> bool:
        if line.startswith("#"):
            return bool(re.match(r"^#{1,6} \S", line))
        if len(line) > self.MAX_HEADING_LENGTH:
            return False
        return bool(self.HEADING.match(line))

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        chunks: list[str] = []
        # A parent heading is often followed straight by its first subsection
        # ("1. DOI TUONG..." then "1.1. Doi Tuong Ap Dung"), leaving the parent
        # with an empty body. Emitting it alone yields a chunk of pure title
        # that matches queries but answers nothing, so carry it forward and let
        # it head the next section instead.
        pending = ""

        for heading, body in self._sections(text):
            title = f"{pending}\n\n{heading}".strip() if pending else heading
            if not body.strip():
                if title:
                    pending = title
                continue

            pending = ""
            content = f"{title}\n\n{body}".strip() if title else body.strip()
            if len(content) <= self.max_section_size:
                chunks.append(content)
            else:
                chunks.extend(self._split_long(title, body))

        if pending:
            chunks.append(pending)
        return chunks

    def _sections(self, text: str) -> list[tuple[str, str]]:
        """Pair each heading with the body that follows it.

        Any text before the first heading becomes a section with an empty
        heading, so nothing ahead of the first clause is dropped.
        """
        sections: list[tuple[str, list[str]]] = []
        heading: str = ""
        body: list[str] = []

        for line in text.splitlines():
            if self._is_heading(line):
                if heading or any(part.strip() for part in body):
                    sections.append((heading, body))
                heading, body = line, []
            else:
                body.append(line)
        if heading or any(part.strip() for part in body):
            sections.append((heading, body))

        return [(head, "\n".join(lines).strip()) for head, lines in sections]

    def _split_long(self, heading: str, body: str) -> list[str]:
        """Cut an oversized section, re-attaching the heading to every piece."""
        prefix = f"{heading}\n\n" if heading else ""
        # A heading long enough to starve the body is really a clause: fold it
        # into the body instead of repeating it. Without this the inner
        # chunker's budget collapses toward 1 and it slices character by
        # character, leaving every fragment over the limit anyway.
        if len(prefix) > self.max_section_size // 2:
            prefix, body = "", f"{prefix}{body}"

        # Budget the prefix out of the limit, otherwise every fragment ends up
        # longer than max_section_size by exactly the heading's length.
        budget = max(1, self.max_section_size - len(prefix))
        return [prefix + piece for piece in RecursiveChunker(chunk_size=budget).chunk(body)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(sum(value * value for value in vec_a))
    magnitude_b = math.sqrt(sum(value * value for value in vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        comparison: dict = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            comparison[name] = {
                "count": count,
                # Guard the empty-text case: a zero count would divide by zero.
                "avg_length": sum(len(chunk) for chunk in chunks) / count if count else 0.0,
                "chunks": chunks,
            }
        return comparison
