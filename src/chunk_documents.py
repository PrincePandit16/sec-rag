"""
chunk_documents.py

Splits documents.jsonl (one raw filing/section per line) into overlapping
chunks suitable for embedding, and writes them to chunks.jsonl.

Expected input schema per line (matches SEC EDGAR full-filing exports):
{
    "company": "abbvie",
    "filing_date": "2025-02-14",
    "form": "10-K",
    "document": "abbv-20241231",
    "source_file": "data\\raw\\abbvie\\2025-02-14_10-K_abbv-20241231.htm",
    "text": "full raw text of the entire filing..."
}

Output schema per line:
{
    "chunk_id": "abbvie_2025-02-14_003",
    "company": "abbvie",
    "filing_date": "2025-02-14",
    "form": "10-K",
    "document": "abbv-20241231",
    "chunk_index": 3,
    "total_chunks": 42,
    "text": "chunk text..."
}

Usage:
    pip install tiktoken --break-system-packages
    python chunk_documents.py --input documents.jsonl --output chunks.jsonl
"""

import argparse
import json
import re
import sys

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    _ENCODER = None
    print(
        f"WARNING: tiktoken unavailable ({e.__class__.__name__}), falling back to "
        "whitespace-token counting (less accurate, but works fine for this project).",
        file=sys.stderr,
    )


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return len(text.split())


def decode_tokens(tokens) -> str:
    """Only used with tiktoken; converts token ids back to text."""
    return _ENCODER.decode(tokens)


# Cover-page boilerplate that shows up near-verbatim in every 10-K/10-Q and
# adds nothing for retrieval (checkbox prompts, filer-status legalese, etc).
# Matched as whole lines/sentences before whitespace gets collapsed.
_BOILERPLATE_PATTERNS = [
    r"Indicate by check mark[^.]*\.",
    r"large accelerated filer,? an accelerated filer,? a non-accelerated filer[^.]*\.",
    r"Securities Registered Pursuant to Section 12\([a-z]\) of the Act[:.]?",
    r"DOCUMENTS INCORPORATED BY REFERENCE",
]


def clean_text(text: str) -> str:
    """Basic cleanup pass. Extend this if you spot other junk in your data
    (XBRL tags, page numbers, repeated boilerplate disclaimers, etc.)."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[☒☐]", " ", text)         # checkbox glyphs
    text = re.sub(r"[� ]", " ", text) # replacement characters and non-breaking spaces
    # Removed repeated dots replacement to avoid corrupting MD table formatting
    text = re.sub(r"<[^>]+>", " ", text)       # strip any leftover HTML/XBRL tags
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace only, preserve newlines
    text = text.strip()
    return text


def recursive_split(text: str, max_tokens: int = 400) -> list[str]:
    """
    Splits text recursively by priority: \n\n -> \n -> ' '.
    Ensures Markdown table rows (|) are kept together.
    """
    if not text:
        return []

    # 1. Check if text already fits
    if count_tokens(text) <= max_tokens:
        return [text]

    # 2. Priority splitters
    splitters = ["\n\n", "\n", " "]

    for splitter in splitters:
        parts = text.split(splitter)
        if len(parts) > 1:
            chunks = []
            current_chunk = ""

            for part in parts:
                # If the part contains a table pipe, we try to keep it intact
                # unless it's already over the limit
                candidate = (current_chunk + splitter + part).strip()
                if count_tokens(candidate) <= max_tokens:
                    current_chunk = candidate
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    # If a single part is too big, recurse on it
                    if count_tokens(part) > max_tokens:
                        # We cannot simply recurse because we need to maintain
                        # the splitting logic. We split the big part further.
                        sub_chunks = recursive_split(part, max_tokens)
                        chunks.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

    # Fallback: Hard slice if everything else fails
    return [text[:max_tokens * 4]] # Approximation


def load_documents(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line {line_num}: {e}", file=sys.stderr)


def make_chunk_id(doc: dict, index: int) -> str:
    company = re.sub(r"[^A-Za-z0-9]+", "", doc.get("company", "UNK"))
    filing_date = doc.get("filing_date", "UNKDATE")
    return f"{company}_{filing_date}_{index:03d}"


def process(input_path: str, output_path: str, chunk_size: int, overlap: int):
    total_docs = 0
    total_chunks = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for doc in load_documents(input_path):
            total_docs += 1

            # --------------------------------------------------------
            # Section-Aware Processing
            # --------------------------------------------------------
            # Use 'sections' if available from ingest.py, else fallback to 'text'
            sections = doc.get("sections", [])
            if not sections:
                # Fallback for documents without identified items
                sections = [{"section_name": "General", "text": doc.get("text", "")}]

            company = doc.get("company", "UNK")
            date = doc.get("filing_date", "UNKDATE")
            form = doc.get("form", "UNKFORM")
            document_id = doc.get("document", "UNKDOC")

            for s_idx, section in enumerate(sections):
                section_name = section.get("section_name", "General")
                raw_text = section.get("text", "")

                text = clean_text(raw_text)
                if not text:
                    continue

                # recursive_split replaces the old sliding window chunk_text
                pieces = recursive_split(text, max_tokens=chunk_size)
                n = len(pieces)

                for i, piece in enumerate(pieces):
                    # Metadata Breadcrumb
                    breadcrumb = f"{company} | {date} | {form} | {section_name}"
                    chunk_text = f"{breadcrumb}\n\n{piece}"

                    record = {
                        "chunk_id": make_chunk_id(doc, (s_idx * n) + i),
                        "company": company,
                        "filing_date": date,
                        "form": form,
                        "document": document_id,
                        "chunk_index": (s_idx * n) + i,
                        "total_chunks": -1, # Recalculated later or left as total
                        "text": chunk_text,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print(f"Processed {total_docs} documents -> {total_chunks} chunks")
    print(f"Wrote output to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk documents.jsonl for embedding.")
    parser.add_argument("--input", default="documents.jsonl", help="Path to input JSONL")
    parser.add_argument("--output", default="chunks.jsonl", help="Path to output JSONL")
    parser.add_argument("--chunk-size", type=int, default=400, help="Tokens per chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    args = parser.parse_args()

    process(args.input, args.output, args.chunk_size, args.overlap)