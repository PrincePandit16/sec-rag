from __future__ import annotations

import json
from pathlib import Path

import chromadb


EMBEDDINGS_FILE = Path("data/processed/embeddings.jsonl")
DB_DIR = Path("data/vector_db")

COLLECTION_NAME = "sec_filings"


def build_vector_database() -> None:

    print("=" * 60)
    print("BUILDING SEC VECTOR DATABASE")
    print("=" * 60)

    print(f"Input:      {EMBEDDINGS_FILE}")
    print(f"Database:   {DB_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print()

    DB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Persistent local Chroma database
    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "SEC 10-K and 10-Q filing chunks",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_dimension": 384,
        },
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    total = 0

    print("Loading embeddings...")

    with EMBEDDINGS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            chunk_id = record["chunk_id"]

            embedding = record["embedding"]

            text = record["text"]

            metadata = {
                "company": record["company"],
                "filing_date": record["filing_date"],
                "form": record["form"],
                "document": record["document"],
                "chunk_index": record["chunk_index"],
                "total_chunks": record["total_chunks"],
            }

            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append(metadata)

            total += 1

    print(f"Loaded: {total:,} embeddings")
    print()

    # --------------------------------------------------------
    # Insert into Chroma
    # --------------------------------------------------------

    batch_size = 1_000

    print("Writing to Chroma...")

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

        print(
            f"Inserted {end:,}/{total:,}"
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    count = collection.count()

    print()
    print("=" * 60)
    print("VECTOR DATABASE COMPLETE")
    print("=" * 60)

    print(f"Expected records: {total:,}")
    print(f"Database records: {count:,}")
    print(f"Location:         {DB_DIR}")

    if count == total:
        print()
        print("STATUS: PASS")
        print("Vector database is ready.")
    else:
        print()
        print("STATUS: ERROR")
        print("Record count does not match.")


if __name__ == "__main__":
    build_vector_database()
