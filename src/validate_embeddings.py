from __future__ import annotations

import json
import math
from pathlib import Path


EMBEDDINGS_FILE = Path("data/processed/embeddings.jsonl")
EXPECTED_DIMENSION = 384


def validate_embeddings() -> None:
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: File not found: {EMBEDDINGS_FILE}")
        return

    total = 0
    missing_embedding = 0
    wrong_dimension = 0
    invalid_values = 0
    duplicate_ids = 0
    missing_text = 0
    missing_metadata = 0

    seen_ids = set()

    print("=" * 65)
    print("EMBEDDING VALIDATION")
    print("=" * 65)
    print(f"File: {EMBEDDINGS_FILE}")
    print()

    with EMBEDDINGS_FILE.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON at line {line_number}: {e}")
                continue

            # ------------------------------------------------
            # Check chunk_id
            # ------------------------------------------------

            chunk_id = record.get("chunk_id")

            if not chunk_id:
                print(f"[WARNING] Missing chunk_id at line {line_number}")
            elif chunk_id in seen_ids:
                duplicate_ids += 1
                print(f"[WARNING] Duplicate chunk_id: {chunk_id}")
            else:
                seen_ids.add(chunk_id)

            # ------------------------------------------------
            # Check text
            # ------------------------------------------------

            if not record.get("text"):
                missing_text += 1

            # ------------------------------------------------
            # Check metadata
            # ------------------------------------------------

            required_metadata = [
                "company",
                "filing_date",
                "form",
                "document",
                "chunk_index",
                "total_chunks",
            ]

            missing = [
                field
                for field in required_metadata
                if field not in record
            ]

            if missing:
                missing_metadata += 1

            # ------------------------------------------------
            # Check embedding
            # ------------------------------------------------

            embedding = record.get("embedding")

            if embedding is None:
                missing_embedding += 1
                continue

            if not isinstance(embedding, list):
                print(
                    f"[ERROR] Embedding is not a list "
                    f"at line {line_number}"
                )
                wrong_dimension += 1
                continue

            # ------------------------------------------------
            # Check dimension
            # ------------------------------------------------

            if len(embedding) != EXPECTED_DIMENSION:
                wrong_dimension += 1

                if wrong_dimension <= 10:
                    print(
                        f"[WARNING] Wrong dimension at line "
                        f"{line_number}: {len(embedding)}"
                    )

            # ------------------------------------------------
            # Check NaN / Infinity / invalid values
            # ------------------------------------------------

            for value in embedding:

                if not isinstance(value, (int, float)):
                    invalid_values += 1
                    break

                if not math.isfinite(value):
                    invalid_values += 1
                    break

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)

    print(f"Total records:             {total:,}")
    print(f"Unique chunk IDs:          {len(seen_ids):,}")
    print(f"Missing embeddings:        {missing_embedding:,}")
    print(f"Wrong dimensions:          {wrong_dimension:,}")
    print(f"Invalid embedding values:  {invalid_values:,}")
    print(f"Duplicate chunk IDs:       {duplicate_ids:,}")
    print(f"Missing text:              {missing_text:,}")
    print(f"Missing metadata:          {missing_metadata:,}")

    print()

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    if (
        total == 45_089
        and
        missing_embedding == 0
        and
        wrong_dimension == 0
        and
        invalid_values == 0
        and
        duplicate_ids == 0
        and
        missing_text == 0
        and
        missing_metadata == 0
    ):
        print("STATUS: PASS")
        print()
        print("All embeddings are valid.")
        print("Ready for vector database ingestion.")

    else:
        print("STATUS: CHECK REQUIRED")
        print()
        print("Some records need investigation.")


if __name__ == "__main__":
    validate_embeddings()
