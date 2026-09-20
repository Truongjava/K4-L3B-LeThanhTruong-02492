#!/usr/bin/env python3
"""Clean crawled Shopee help-center pages into a normalized corpus.

`scripts/fetch_public_pages.py` saves raw page text. Shopee's help center pages
carry three fixed chrome fragments, are emitted as *decomposed* Vietnamese (NFD)
with non-breaking spaces, and on Windows round-trip to CRLF line endings. Left
alone, those defects silently break string matching against the corpus:

  * NFD means a gold-answer marker typed normally ("nay" with a composed accent)
    does not match the corpus bytes (base letter + U+0300 combining mark) -- the
    CP6 content-level check would report "not retrieved" for text that is plainly
    there, and the wrong chunk would look like the culprit.
  * NBSP means "7 ngay" does not match "7\\xa0ngay", so a deadline that is right
    there reads as absent.

So we strip the chrome, normalize to NFC with NBSP folded to a plain space, and
lift the stated effective date into `document_version` so each file carries real
provenance instead of `not-stated` wherever the source actually states one.

Normalization runs *before* chrome stripping, so the fixed fragments compare
correctly even when the source writes them with non-breaking spaces.

Idempotent: running it twice is a no-op.

Usage:
    python scripts/clean_shopee_pages.py data/shopee-doi-tra-hoan-tien
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

NBSP = " "

# Three fixed chrome fragments injected by help.shopee.vn on every article.
GREETING = "Xin chào, Shopee có thể giúp gì cho bạn?"
FEEDBACK_LEAD = "Bạn có hài lòng với bài viết này?"
TITLE_SUFFIX = "| Shopee Trung tâm trợ giúp"

# Effective-date patterns, tried in order. Every pattern exposes the date it
# wants via named groups d/m/y so the caller does not care how many dates the
# surrounding sentence mentions.
#
# The first pattern deliberately requires the *paired* form
# ("dang tai vao ngay X, co hieu luc ke tu ngay Y") rather than matching a bare
# "co hieu luc ke tu ngay Y". Some pages use that bare phrase for an individual
# provision rather than the document: quy-trinh-giai-quyet-tranh-chap.md says the
# procedure is published 15/3/2024 while "quy dinh ve Nguoi tieu dung de bi ton
# thuong" takes effect 01/07/2024. Matching the bare phrase there yields a
# plausible-but-wrong version -- worse than not-stated, because it looks right.
DATE_PATTERNS = [
    # "dang tai vao ngay 04/3/2026, co hieu luc ke tu ngay 11/3/2026"
    re.compile(
        r"đăng tải vào ngày\s*\d{1,2}/\d{1,2}/\d{4},\s*"
        r"có hiệu lực kể từ ngày\s*(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})"
    ),
    # "CHINH SACH VAN CHUYEN SHOPEE (phien ban ngay 11/12/2025)"
    re.compile(r"\(phiên bản ngày\s*(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})\)"),
    # "Ban Cap Nhat ngay 03/01/2025."
    re.compile(r"Bản Cập Nhật ngày\s*(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})"),
    # "Quy Trinh nay duoc cap nhat va dang tai vao ngay 15/3/2024."
    re.compile(r"cập nhật và đăng tải vào ngày\s*(?P<d>\d{1,2})/(?P<m>\d{1,2})/(?P<y>\d{4})"),
]


def split_front_matter(text: str) -> tuple[str, str]:
    """Return (front_matter_including_delimiters, body)."""
    if not text.startswith("---\n"):
        return "", text
    end = text.find("\n---\n", 4)
    if end == -1:
        return "", text
    return text[: end + len("\n---\n")], text[end + len("\n---\n") :]


def normalize(text: str) -> str:
    """NFC-normalize Vietnamese, fold NBSP to space, tidy whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(NBSP, " ")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def strip_chrome(body: str) -> tuple[str, int]:
    """Drop the three fixed help-center chrome fragments. Returns (body, removed).

    Expects `body` already normalized, so the fragments compare exactly.
    """
    kept: list[str] = []
    removed = 0
    for line in body.split("\n"):
        bare = line.strip()
        if bare == GREETING or bare.endswith(TITLE_SUFFIX):
            removed += 1
            continue
        if bare == FEEDBACK_LEAD:
            # The feedback widget is the last thing on the page; drop it along
            # with the two option labels that follow.
            removed += 3
            break
        kept.append(line)
    return "\n".join(kept), removed


def find_effective_date(body: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = pattern.search(body)
        if match:
            found = match.groupdict()
            return f"{found['y']}-{int(found['m']):02d}-{int(found['d']):02d}"
    return None


def existing_version(front_matter: str) -> str:
    """Read back the document_version already recorded, if it is a real one."""
    match = re.search(r"^document_version:\s*(.+)$", front_matter, re.M)
    if not match:
        return ""
    value = match.group(1).strip().strip('"')
    return "" if value == "not-stated" else value


def set_document_version(front_matter: str, version: str) -> str:
    return re.sub(
        r"^document_version:.*$",
        f'document_version: "{version}"',
        front_matter,
        count=1,
        flags=re.M,
    )


def unquote_front_matter(front_matter: str) -> str:
    """Drop the quotes the crawler always adds, wherever a bare value is safe.

    DATA_COLLECTION.md documents the canonical front matter unquoted, and the CP2
    checklist parses it with a plain `^(\\w+):\\s*(.+)$` regex that does not strip
    quotes. A fully-quoted block therefore reads as "THIEU METADATA" and the
    csv/front-matter cross-check reports LECH -- a false alarm on correct data.

    Values keep their quotes when removing them would change meaning: anything
    holding an escape, an inner quote, a colon-space or comment marker, or
    starting with a YAML indicator character. That last case is load-bearing --
    the return-process pages are titled "[Tra hang/Hoan tien] ...", and a plain
    scalar may not begin with "[" or YAML reads it as a flow sequence.
    """
    yaml_indicator_start = set("-?:,[]{}#&*!|>'\"%@`")
    rebuilt: list[str] = []
    for line in front_matter.split("\n"):
        match = re.match(r'^(\w+): "(.*)"$', line)
        if match:
            key, value = match.groups()
            unsafe = (
                not value
                or value[0] in yaml_indicator_start
                or any(token in value for token in (": ", " #", '"', "\\"))
            )
            if not unsafe:
                rebuilt.append(f"{key}: {value}")
                continue
        rebuilt.append(line)
    return "\n".join(rebuilt)


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    # lineterminator="\n": csv.writer defaults to CRLF, which would leave the
    # manifest on Windows line endings while every .md beside it is LF.
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean crawled Shopee pages.")
    parser.add_argument("corpus_dir", type=Path, help="Directory of crawled .md files")
    args = parser.parse_args()

    if not args.corpus_dir.is_dir():
        print(f"Not a directory: {args.corpus_dir}", file=sys.stderr)
        return 2

    versions: dict[str, str] = {}
    print(f"{'file':44} {'chrome':>6} {'NFD':>5} {'nbsp':>5} {'CRLF':>5}  version")

    for path in sorted(args.corpus_dir.glob("*.md")):
        original = path.read_text(encoding="utf-8")
        nfd_count = sum(1 for ch in original if unicodedata.combining(ch))
        nbsp_count = original.count(NBSP)
        crlf = "\r\n" in original

        front_matter, body = split_front_matter(original)
        body, removed = strip_chrome(normalize(body))
        # A page states its version once, in the section carrying the publication
        # date. After split_policy_by_audience.py moves sections around, the
        # seller half may hold no date at all -- so an already-recorded version
        # wins over a missing one, rather than being downgraded to not-stated.
        version = find_effective_date(body) or existing_version(front_matter) or "not-stated"

        header = unquote_front_matter(
            normalize(set_document_version(normalize(front_matter), version))
        )
        path.write_text(f"{header}\n\n{body}\n", encoding="utf-8", newline="\n")
        versions[path.stem] = version

        print(
            f"{path.name:44} {removed:>6} {nfd_count:>5} {nbsp_count:>5} "
            f"{str(crlf):>5}  {version}"
        )

    manifest_path = args.corpus_dir / "sources.csv"
    rows = read_manifest(manifest_path)
    if rows:
        fields = list(rows[0].keys())
        for row in rows:
            if row.get("doc_id") in versions:
                row["document_version"] = versions[row["doc_id"]]
            # The crawler writes str(Path), so on Windows this is a backslash
            # path. DATA_COLLECTION.md documents forward slashes, and a
            # backslash path does not resolve on macOS/Linux -- teammates
            # running the same corpus elsewhere would see a broken manifest.
            if row.get("file_path"):
                row["file_path"] = row["file_path"].replace("\\", "/")
        write_manifest(manifest_path, rows, fields)
        print(f"\nUpdated {manifest_path} ({len(rows)} rows)")

    stated = sum(1 for value in versions.values() if value != "not-stated")
    print(f"Effective date found in {stated}/{len(versions)} files; rest are not-stated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())