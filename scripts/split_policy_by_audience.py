#!/usr/bin/env python3
"""Split a mixed-audience policy into one file per audience.

`docs/DATA_COLLECTION.md` section 4: when one page covers buyer and seller on the
same topic, split it into separate files -- one audience each -- so that
`search_with_filter()` has something real to filter on. Without the split, a
question that does not name the asker returns both sides and the agent answers
for the wrong party, which is precisely the A/B the benchmark is meant to show.

`chinh-sach-tra-hang-hoan-tien.md` (article 77251) is that page: its section 1.1
says the policy binds "Nguoi Mua, Nguoi Ban, cac don vi cung cap dich vu van
chuyen". Sections 5 and 7 carve out seller rights and seller return-shipping
cost obligations; the rest is buyer-facing.

The split is a *partition*, not a copy. Seller sections move out of the buyer
file, so a buyer-filtered search cannot surface a seller answer and vice versa --
copying them into both would leave the filter with nothing to do. Section numbers
keep the source's own numbering so the document's internal cross-references stay
valid; all of those point at sections 3, 4 and 9, which stay in the buyer file.

Idempotent: skips if the seller file already exists. There is deliberately no
--overwrite: the split is destructive to the source file, so re-running it on an
already-split buyer file would find sections 5 and 7 gone and drop them for good.
To redo a bad split, re-crawl article 77251 first, then run this again.

Usage:
    python scripts/split_policy_by_audience.py data/shopee-doi-tra-hoan-tien
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

SOURCE_DOC_ID = "chinh-sach-tra-hang-hoan-tien"
SPLIT_DOC_ID = "chinh-sach-tra-hang-hoan-tien-nguoi-ban"

# Sections that belong to the seller side of article 77251.
SELLER_SECTIONS = {5, 7}

SPLIT_TITLE = "Chính sách Trả hàng và Hoàn tiền — Quyền và trách nhiệm của Người bán"
SPLIT_CATEGORY = "seller-return-obligations"

# A top-level heading: "7. TRACH NHIEM ...". Subsection headings ("7.1. ...") do
# not match, because a digit rather than a space follows the first period.
SECTION_HEADING = re.compile(r"^(\d{1,2})\. [A-ZĐ]", re.M)


def split_front_matter(text: str) -> tuple[str, str]:
    """Return (front_matter_with_delimiters, body)."""
    end = text.find("\n---\n", 4)
    if not text.startswith("---\n") or end == -1:
        return "", text
    return text[: end + len("\n---\n")], text[end + len("\n---\n") :]


def front_matter_dict(front_matter: str) -> dict[str, str]:
    return dict(re.findall(r"^(\w+):\s*(.+)$", front_matter, re.M))


def render_front_matter(fields: dict[str, str]) -> str:
    lines = ["---"]
    for key, value in fields.items():
        if value.startswith(("[", "-", "?", ":", ",", "#", "&", "*", "!", "|", ">", "'", "%", "@", "`")):
            lines.append(f'{key}: "{value}"')
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def section_spans(body: str) -> list[tuple[int, int, int]]:
    """Return [(section_number, start_offset, end_offset)] for top-level sections."""
    starts = [(int(m.group(1)), m.start()) for m in SECTION_HEADING.finditer(body)]
    return [
        (number, start, starts[i + 1][1] if i + 1 < len(starts) else len(body))
        for i, (number, start) in enumerate(starts)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a policy by audience.")
    parser.add_argument("corpus_dir", type=Path, help="Directory holding the .md corpus")
    args = parser.parse_args()

    source_path = args.corpus_dir / f"{SOURCE_DOC_ID}.md"
    split_path = args.corpus_dir / f"{SPLIT_DOC_ID}.md"
    if not source_path.is_file():
        print(f"Not found: {source_path}", file=sys.stderr)
        return 2
    if split_path.exists():
        print(f"Already split: {split_path.name} exists. To redo, re-crawl "
              f"{SOURCE_DOC_ID} first (this split is destructive to the source).")
        return 0

    front_matter, body = split_front_matter(source_path.read_text(encoding="utf-8"))

    spans = section_spans(body)
    if [number for number, _, _ in spans] != list(range(1, len(spans) + 1)):
        found = [number for number, _, _ in spans]
        print(f"Refusing to split: expected sections 1..N, found {found}", file=sys.stderr)
        return 2

    preamble = body[: spans[0][1]] if spans else body
    pieces = [(number in SELLER_SECTIONS, body[start:end]) for number, start, end in spans]

    # The split must be a clean partition: preamble plus every section, in the
    # source's order, has to reconstruct the body byte for byte. If it does not,
    # the section regex has drifted and we would silently drop or duplicate text.
    if (preamble + "".join(block for _, block in pieces)).strip() != body.strip():
        print("Refusing to split: content would not round-trip", file=sys.stderr)
        return 2

    buyer_body = preamble + "".join(block for is_seller, block in pieces if not is_seller)
    seller_body = preamble.rstrip("\n") + "\n\n" + "".join(
        "\n" + block for is_seller, block in pieces if is_seller
    )

    seller_fields = front_matter_dict(front_matter)
    seller_fields["doc_id"] = SPLIT_DOC_ID
    seller_fields["title"] = SPLIT_TITLE
    seller_fields["audience"] = "seller"
    seller_fields["category"] = SPLIT_CATEGORY

    split_path.write_text(
        render_front_matter(seller_fields) + "\n" + seller_body.strip("\n") + "\n",
        encoding="utf-8",
        newline="\n",
    )
    source_path.write_text(
        render_front_matter(front_matter_dict(front_matter)) + "\n" + buyer_body.strip("\n") + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest_path = args.corpus_dir / "sources.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    fields = list(rows[0].keys())
    template = next(row for row in rows if row["doc_id"] == SOURCE_DOC_ID)
    rows = [row for row in rows if row["doc_id"] != SPLIT_DOC_ID]
    rows.append(
        {
            **template,
            "doc_id": SPLIT_DOC_ID,
            "file_path": f"{args.corpus_dir.as_posix()}/{SPLIT_DOC_ID}.md",
            "title": SPLIT_TITLE,
        }
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda row: row["doc_id"]):
            writer.writerow({field: row.get(field, "") for field in fields})

    print(f"Seller sections moved : {sorted(SELLER_SECTIONS)}")
    print(f"Buyer file            : {source_path.name} ({len(buyer_body)} kt)")
    print(f"Seller file (new)     : {split_path.name} ({len(seller_body)} kt)")
    print(f"Manifest rows         : {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())