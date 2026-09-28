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
    text = re.sub(r"\.{2,}", " ", text)       # repeated dots in tables
    text = re.sub(r"<[^>]+>", " ", text)       # strip any leftover HTML/XBRL tags
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace only, preserve newlines
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    """Splits text into overlapping chunks by token count.
    Falls back to word-based splitting if tiktoken isn't available."""
    if not text:
        return []

    if _ENCODER is not None:
        tokens = _ENCODER.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(decode_tokens(chunk_tokens))
            if end == len(tokens):
                break
            start += chunk_size - overlap
        return chunks
    else:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += chunk_size - overlap
        return chunks


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

            raw_text = doc.get("text", "")
            text = clean_text(raw_text)
            if not text:
                continue

            pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            n = len(pieces)

            for i, piece in enumerate(pieces):
                record = {
                    "chunk_id": make_chunk_id(doc, i),
                    "company": doc.get("company"),
                    "filing_date": doc.get("filing_date"),
                    "form": doc.get("form"),
                    "document": doc.get("document"),
                    "chunk_index": i,
                    "total_chunks": n,
                    "text": piece,
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