from __future__ import annotations
import os

import argparse
import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import anthropic
import ftfy
from charset_normalizer import from_path

from src import config
from src.utils.text_cleaning import normalize_whitespace, unified_clean
from src.utils.logging_config import logger
from src.utils.io import safe_write_jsonl


# ============================================================
# Configuration
# ============================================================

RAW_DIR = config.RAW_DIR
PROCESSED_DIR = config.PROCESSED_DIR
OUTPUT_FILE = config.DOCUMENTS_JSONL


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

        data/raw_backup/apple/
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


def table_to_markdown(table_tag) -> str:
    """
    Convert a BeautifulSoup <table> tag into a GitHub Flavored Markdown table
    using a virtual grid to handle colspan and rowspan.
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    # 1. Determine Grid Dimensions
    max_cols = 0
    for row in rows:
        cols = 0
        for cell in row.find_all(["td", "th"]):
            cols += int(cell.get("colspan", 1))
        max_cols = max(max_cols, cols)

    # 2. Initialize Matrix
    num_rows = len(rows)
    # We use a slightly larger matrix to handle rowspans that might extend beyond the last <tr>
    matrix = [[None] * max_cols for _ in range(num_rows * 2)]

    # 3. Populate Matrix
    r = 0
    for row in rows:
        c = 0
        for cell in row.find_all(["td", "th"]):
            # Find next available slot
            while c < max_cols and matrix[r][c] is not None:
                c += 1

            if c >= max_cols:
                break

            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            val = cell.get_text(strip=True).replace("\n", " ").replace("|", "\\|")

            for i in range(r, r + rs):
                if i < len(matrix):
                    for j in range(c, c + cs):
                        if j < max_cols:
                            matrix[i][j] = val
            c += cs
        r += 1

    # Trim matrix to actual used rows
    actual_rows = []
    for row in matrix:
        if any(cell is not None for cell in row):
            actual_rows.append(row)
        else:
            break

    if not actual_rows:
        return ""

    # 4. Render to Markdown
    markdown_rows = []
    for row in actual_rows:
        cell_texts = [cell if cell is not None else "" for cell in row]
        markdown_rows.append(f"| {' | '.join(cell_texts)} |")

    # Create separator based on max_cols
    separator = f"| {' | '.join(['---'] * max_cols)} |"

    if len(markdown_rows) > 1:
        markdown_rows.insert(1, separator)
    else:
        markdown_rows.append(separator)

    return "\n" + "\n".join(markdown_rows) + "\n"


def summarize_table(md_table: str) -> str:
    """
    Uses an LLM to generate a concise summary of a markdown table.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Summary unavailable (no API key)."

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "You are a financial analyst. Below is a table extracted from an SEC filing in Markdown format. "
        "Please provide a concise (1-3 sentence) summary of what this table represents, "
        "the key metrics it tracks, and the time period it covers.\n\n"
        f"Table:\n{md_table}\n\n"
        "Summary:"
    )

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        for block in message.content:
            if block.type == "text":
                return block.text.strip()
        raise ValueError("Anthropic returned no text block for the table summary.")
    except Exception as e:
        return f"Error generating summary: {e}"

def html_to_text(filepath: Path) -> str:
    """
    Convert an SEC filing HTML document into clean text.
    """
    try:
        result = from_path(filepath)
        html = str(result.best())
        html = ftfy.fix_text(html)
    except Exception as e:
        logger.warning(f"Encoding error in {filepath}: {e}")
        html = filepath.read_text(encoding="utf-8", errors="ignore")

    soup = BeautifulSoup(html, "lxml")

    # Remove XBRL / Inline XBRL metadata
    for tag in soup.find_all(["ix:header", "ix:hidden", "ix:resources"]):
        tag.decompose()

    for tag in soup.find_all():
        name = tag.name.lower()
        if name.startswith("ix:") or name.startswith("xbrli:") or name.startswith("xbrldi:"):
            tag.decompose()

    # Remove standard HTML noise
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    # Convert Tables to Markdown with Summaries
    for table in soup.find_all("table"):
        markdown_table = table_to_markdown(table)
        summary = ""
        if 500 < len(markdown_table) < 15000:
            summary = summarize_table(markdown_table)
            summary = f"\nTable Summary: {summary}\n"
        table.replace_with(f"\n{summary}{markdown_table}\n")

    text = soup.get_text(separator="\n", strip=True)
    return unified_clean(text)


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

        # --------------------------------------------------------
        # Section Splitting (Item Detection)
        # --------------------------------------------------------
        # SEC filings use 'Item X' as primary section markers.
        # We split the text into sections to maintain hierarchy.
        item_pattern = re.compile(r"(Item\s+\d+[A-Z]?\.?\s+[^ \n\r]+)", re.IGNORECASE)
        sections = []

        # Find all matches for "Item X ..."
        matches = list(item_pattern.finditer(text))

        if matches:
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i+1].start() if i + 1 < len(matches) else len(text)

                section_name = matches[i].group(1).strip()
                section_text = text[start:end].strip()

                sections.append({
                    "section_name": section_name,
                    "text": section_text
                })
        else:
            # Fallback if no items found
            sections.append({
                "section_name": "General",
                "text": text
            })

        document = {
            **metadata,
            "sections": sections,
            "text": text, # Keep original for backward compatibility
            "character_count": len(text),
        }

        return document

    except Exception as error:
        logger.error(f"Error processing {filepath}: {error}")
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
