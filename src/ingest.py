from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning


# ============================================================
# Configuration
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = PROCESSED_DIR / "documents.jsonl"


# ============================================================
# Warnings
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=XMLParsedAsHTMLWarning,
)


# ============================================================
# Filename pattern
# ============================================================

FILENAME_PATTERN = re.compile(
    r"(?P<filing_date>\d{4}-\d{2}-\d{2})_"
    r"(?P<form>10-k|10-q)_"
    r"(?P<document>.+)\.htm$",
    re.IGNORECASE,
)


# ============================================================
# Metadata extraction
# ============================================================

def extract_metadata(filepath: Path) -> dict:
    """
    Extract metadata from the SEC filing path and filename.

    Example:

        data/raw/apple/
        2024-11-01_10-K_aapl-20240928.htm

    Produces:

        {
            "company": "apple",
            "filing_date": "2024-11-01",
            "form": "10-K",
            "document": "aapl-20240928",
            "source_file": "..."
        }
    """

    company = filepath.parent.name

    match = FILENAME_PATTERN.match(filepath.name)

    if not match:
        raise ValueError(
            f"Could not parse filename: {filepath.name}"
        )

    return {
        "company": company,
        "filing_date": match.group("filing_date"),
        "form": match.group("form").upper(),
        "document": match.group("document"),
        "source_file": str(filepath),
    }


# ============================================================
# HTML → Text
# ============================================================

def html_to_text(filepath: Path) -> str:
    """
    Convert an SEC filing HTML document into clean text.

    Removes:
    - XBRL metadata
    - scripts
    - styles
    - SVGs
    - other obvious HTML noise

    Returns:
        Clean human-readable text.
    """

    html = filepath.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    # --------------------------------------------------------
    # Remove XBRL / Inline XBRL metadata
    # --------------------------------------------------------

    for tag in soup.find_all(
        [
            "ix:header",
            "ix:hidden",
            "ix:resources",
        ]
    ):
        tag.decompose()

    # Remove XBRL namespace elements
    for tag in soup.find_all():

        name = tag.name.lower()

        if (
            name.startswith("ix:")
            or name.startswith("xbrli:")
            or name.startswith("xbrldi:")
        ):
            tag.decompose()

    # --------------------------------------------------------
    # Remove standard HTML noise
    # --------------------------------------------------------

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
        ]
    ):
        element.decompose()

    # --------------------------------------------------------
    # Extract visible text
    # --------------------------------------------------------

    text = soup.get_text(
        separator="\n",
        strip=True,
    )

    # --------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------

    lines = []

    for line in text.splitlines():

        line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# Process one document
# ============================================================

def process_document(filepath: Path) -> dict | None:
    """
    Process one SEC filing.

    Returns a dictionary containing:
    - metadata
    - cleaned text
    - character count
    """

    try:

        metadata = extract_metadata(filepath)

        text = html_to_text(filepath)

        if not text.strip():

            print(
                f"[WARNING] Empty document: {filepath}"
            )

            return None

        document = {
            **metadata,
            "text": text,
            "character_count": len(text),
        }

        return document

    except Exception as error:

        print(
            f"[ERROR] {filepath}: {error}"
        )

        return None


# ============================================================
# Find SEC filings
# ============================================================

def find_filings() -> list[Path]:
    """
    Find all .htm SEC filings recursively.
    """

    return sorted(
        RAW_DIR.rglob("*.htm")
    )


# ============================================================
# Ingestion
# ============================================================

def ingest(limit: int | None = None) -> None:
    """
    Process SEC filings and write them to JSONL.

    One line in documents.jsonl = one SEC filing.
    """

    # Create processed directory
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Find filings
    files = find_filings()

    print("=" * 60)
    print("SEC RAG INGESTION")
    print("=" * 60)

    print()
    print(f"Raw directory: {RAW_DIR}")
    print(f"Found filings: {len(files)}")

    # --------------------------------------------------------
    # Apply limit
    # --------------------------------------------------------

    if limit is not None:

        files = files[:limit]

        print(
            f"Processing first {limit} filing(s)"
        )

    print()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    processed = 0
    failed = 0
    total_characters = 0

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as output:

        for index, filepath in enumerate(
            files,
            start=1,
        ):

            print(
                f"[{index}/{len(files)}] "
                f"{filepath}"
            )

            document = process_document(
                filepath
            )

            # Failed document
            if document is None:

                failed += 1

                continue

            # ------------------------------------------------
            # Write ONE JSON object per line
            # ------------------------------------------------

            output.write(
                json.dumps(
                    document,
                    ensure_ascii=False,
                )
                + "\n"
            )

            processed += 1

            total_characters += document[
                "character_count"
            ]

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print()
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(
        f"Characters: {total_characters:,}"
    )
    print()
    print(f"Output: {OUTPUT_FILE}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into JSONL."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N filings.",
    )

    args = parser.parse_args()

    ingest(
        limit=args.limit
    )
