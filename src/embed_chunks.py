from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = Path("data/processed")

INPUT_FILE = PROCESSED_DIR / "chunks.jsonl"
OUTPUT_FILE = PROCESSED_DIR / "embeddings.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_BATCH_SIZE = 32

DEFAULT_DEVICE = "auto"


# ============================================================
# Device detection
# ============================================================

def detect_device() -> str:
    """
    Automatically select the best available device.

    CUDA -> NVIDIA GPU
    MPS  -> Apple Silicon GPU
    CPU  -> fallback
    """

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

    except Exception:
        pass

    return "cpu"


# ============================================================
# Load chunks
# ============================================================

def load_chunks(
    input_path: Path,
):
    """
    Read chunks.jsonl one record at a time.
    """

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                yield json.loads(line)

            except json.JSONDecodeError as error:

                print(
                    f"[WARNING] "
                    f"Skipping malformed line "
                    f"{line_number}: {error}",
                    file=sys.stderr,
                )


# ============================================================
# Find already embedded chunks
# ============================================================

def load_completed_chunk_ids(
    output_path: Path,
) -> set[str]:

    completed = set()

    if not output_path.exists():

        return completed

    print(
        "Checking existing embeddings..."
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(line)

                chunk_id = record.get(
                    "chunk_id"
                )

                if chunk_id:
                    completed.add(chunk_id)

            except json.JSONDecodeError:

                continue

    print(
        f"Already embedded: "
        f"{len(completed):,}"
    )

    return completed


# ============================================================
# Write embeddings
# ============================================================

def write_embeddings(
    output_file,
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        record = {
            "chunk_id": chunk.get(
                "chunk_id"
            ),

            "company": chunk.get(
                "company"
            ),

            "filing_date": chunk.get(
                "filing_date"
            ),

            "form": chunk.get(
                "form"
            ),

            "document": chunk.get(
                "document"
            ),

            "chunk_index": chunk.get(
                "chunk_index"
            ),

            "total_chunks": chunk.get(
                "total_chunks"
            ),

            "text": chunk.get(
                "text"
            ),

            "embedding": embedding.tolist(),
        }

        output_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    # Make sure the batch is written
    # to disk immediately.

    output_file.flush()


# ============================================================
# Main processing
# ============================================================

def process(
    input_path: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
    device: str,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if device == "auto":

        device = detect_device()

    print()
    print("=" * 60)
    print("LOCAL EMBEDDING PIPELINE")
    print("=" * 60)

    print()
    print(
        f"Model: {model_name}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Input: {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading embedding model..."
    )

    start_time = time.time()

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    model_load_time = (
        time.time() - start_time
    )

    print(
        f"Model loaded in "
        f"{model_load_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Embedding dimension
    # --------------------------------------------------------

    embedding_dimension = (
        model.get_sentence_embedding_dimension()
    )

    print(
        f"Embedding dimension: "
        f"{embedding_dimension}"
    )

    # --------------------------------------------------------
    # Find completed chunks
    # --------------------------------------------------------

    completed_ids = (
        load_completed_chunk_ids(
            output_path
        )
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_input = 0
    skipped = 0
    embedded = 0
    batch_number = 0

    batch: list[dict] = []

    start_time = time.time()

    # --------------------------------------------------------
    # IMPORTANT:
    # Append mode allows resume.
    # --------------------------------------------------------

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as output_file:

        for chunk in load_chunks(
            input_path
        ):

            total_input += 1

            chunk_id = chunk.get(
                "chunk_id"
            )

            # ------------------------------------------------
            # Skip existing embeddings
            # ------------------------------------------------

            if chunk_id in completed_ids:

                skipped += 1
                continue

            # ------------------------------------------------
            # Make sure text exists
            # ------------------------------------------------

            text = chunk.get(
                "text",
                "",
            )

            if not text.strip():

                print(
                    f"[WARNING] "
                    f"Empty chunk: {chunk_id}"
                )

                continue

            batch.append(chunk)

            # ------------------------------------------------
            # Process batch
            # ------------------------------------------------

            if len(batch) >= batch_size:

                batch_number += 1

                process_batch(
                    model=model,
                    batch=batch,
                    output_file=output_file,
                    completed_ids=completed_ids,
                    batch_number=batch_number,
                    embedded_before=embedded,
                    start_time=start_time,
                )

                embedded += len(batch)

                batch = []

        # ----------------------------------------------------
        # Process final batch
        # ----------------------------------------------------

        if batch:

            batch_number += 1

            process_batch(
                model=model,
                batch=batch,
                output_file=output_file,
                completed_ids=completed_ids,
                batch_number=batch_number,
                embedded_before=embedded,
                start_time=start_time,
            )

            embedded += len(batch)

    # ========================================================
    # Final statistics
    # ========================================================

    elapsed = (
        time.time() - start_time
    )

    total_embeddings = (
        len(completed_ids)
    )

    print()
    print("=" * 60)
    print("EMBEDDING COMPLETE")
    print("=" * 60)

    print()

    print(
        f"Input chunks: "
        f"{total_input:,}"
    )

    print(
        f"Skipped existing: "
        f"{skipped:,}"
    )

    print(
        f"New embeddings: "
        f"{embedded:,}"
    )

    print(
        f"Total embeddings: "
        f"{total_embeddings:,}"
    )

    print(
        f"Embedding dimension: "
        f"{embedding_dimension}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    if elapsed > 0:

        print(
            f"Speed: "
            f"{embedded / elapsed:.2f} chunks/sec"
        )

    print()

    print(
        f"Output: {output_path}"
    )

    print()


# ============================================================
# Process one batch
# ============================================================

def process_batch(
    model: SentenceTransformer,
    batch: list[dict],
    output_file,
    completed_ids: set[str],
    batch_number: int,
    embedded_before: int,
    start_time: float,
) -> None:

    texts = [
        chunk["text"]
        for chunk in batch
    ]

    print(
        f"Batch {batch_number} | "
        f"{len(batch)} chunks | "
        f"Total embedded: "
        f"{embedded_before:,}"
    )

    # --------------------------------------------------------
    # Generate embeddings
    #
    # normalize_embeddings=True means vectors are normalized
    # and cosine similarity can be calculated efficiently.
    # --------------------------------------------------------

    embeddings = model.encode(
        texts,
        batch_size=len(batch),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    # --------------------------------------------------------
    # Write immediately
    # --------------------------------------------------------

    write_embeddings(
        output_file,
        batch,
        embeddings,
    )

    # --------------------------------------------------------
    # Mark as completed
    # --------------------------------------------------------

    for chunk in batch:

        completed_ids.add(
            chunk["chunk_id"]
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    elapsed = (
        time.time() - start_time
    )

    total_done = (
        embedded_before
        + len(batch)
    )

    speed = (
        total_done / elapsed
        if elapsed > 0
        else 0
    )

    print(
        f"  Done | "
        f"{total_done:,} chunks | "
        f"{speed:.2f} chunks/sec"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate local embeddings "
            "for SEC document chunks."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help="Input chunks JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output embeddings JSONL file.",
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="SentenceTransformer model.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        help="Embedding device.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not args.input.exists():

        parser.error(
            f"Input file does not exist: "
            f"{args.input}"
        )

    if args.batch_size <= 0:

        parser.error(
            "batch-size must be greater than 0."
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    process(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
