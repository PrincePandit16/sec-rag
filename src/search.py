from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# CONFIG
# ============================================================
CHROMA_DIR = Path("data/vector_db")
COLLECTION_NAME = "sec_filings"



EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 100


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    print(f"Embedding device: {model.device}")

    return model


# ============================================================
# LOAD DATABASE
# ============================================================

def load_collection():

    print("Loading Chroma database...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Documents: {collection.count():,}"
    )

    return collection


# ============================================================
# QUERY PARSING
# ============================================================

def detect_company(query: str) -> str | None:

    companies = {
        "apple": "apple",
        "microsoft": "microsoft",
        "amazon": "amazon",
        "google": "alphabet",
        "alphabet": "alphabet",
        "meta": "meta",
        "facebook": "meta",
        "nvidia": "nvidia",
        "tesla": "tesla",
        "jpmorgan": "jpmorgan",
        "jp morgan": "jpmorgan",
        "johnson & johnson": "johnson-johnson",
        "j&j": "johnson-johnson",
    }

    query_lower = query.lower()

    for name, company in companies.items():

        if name in query_lower:
            return company

    return None


def detect_year(query: str) -> int | None:

    match = re.search(
        r"\b(20\d{2})\b",
        query,
    )

    if match:
        year = int(match.group(1))
        current_year = 2026
        if year > current_year:
            print(f"\nWarning: Detected year {year} is in the future. Searching without year filter.")
            return None
        return year

    return None


def detect_form(query: str) -> str | None:

    query_lower = query.lower()

    if "10-k" in query_lower:
        return "10-K"

    if "10-k" in query_lower:
        return "10-K"

    if "10-q" in query_lower:
        return "10-Q"

    return None


# ============================================================
# QUERY TYPE
# ============================================================

def extract_keywords(query: str) -> list[str]:

    query_lower = query.lower()

    keyword_groups = {

        "revenue": [
            "revenue",
            "net sales",
            "total net sales",
            "sales",
        ],

        "profit": [
            "profit",
            "net income",
            "earnings",
        ],

        "margin": [
            "gross margin",
            "operating margin",
            "margin",
        ],

        "expenses": [
            "expense",
            "expenses",
            "operating expenses",
            "r&d",
            "research and development",
        ],

        "assets": [
            "assets",
            "total assets",
        ],

        "debt": [
            "debt",
            "borrowings",
            "liabilities",
        ],

        "cash": [
            "cash",
            "cash equivalents",
            "cash flow",
        ],
    }

    keywords = []

    for group_keywords in keyword_groups.values():

        for keyword in group_keywords:

            if keyword in query_lower:

                keywords.append(keyword)

    return list(dict.fromkeys(keywords))


# ============================================================
# METADATA FILTER
# ============================================================

def build_where_filter(
    company: str | None,
    year: int | None,
    form: str | None,
):

    conditions = []

    if company:

        conditions.append(
            {
                "company": company
            }
        )

    if form:

        conditions.append(
            {
                "form": form
            }
        )

    # IMPORTANT:
    #
    # We DON'T directly filter filing_date by year
    # because filing date and fiscal year are not always
    # the same thing.
    #
    # Example:
    #
    # Apple's FY2024 10-K
    # filing date = 2024-11-01
    #
    # Therefore year filtering is handled separately.

    if len(conditions) == 0:

        return None

    if len(conditions) == 1:

        return conditions[0]

    return {
        "$and": conditions
    }


# ============================================================
# SEARCH
# ============================================================

def search(
    collection,
    model,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    company: str | None = None,
    year: int | None = None,
    form: str | None = None,
):

    # --------------------------------------------------------
    # Detect filters
    # --------------------------------------------------------

    detected_company = detect_company(query)

    detected_year = detect_year(query)

    detected_form = detect_form(query)

    if company is None:
        company = detected_company

    if year is None:
        year = detected_year

    if form is None:
        form = detected_form

    keywords = extract_keywords(query)

    print()
    print("Search parameters")
    print("-" * 60)

    print(
        f"Company: {company or 'ANY'}"
    )

    print(
        f"Year:    {year or 'ANY'}"
    )

    print(
        f"Form:    {form or 'ANY'}"
    )

    print(
        f"Keywords: {', '.join(keywords) if keywords else 'NONE'}"
    )

    # --------------------------------------------------------
    # Embed query
    # --------------------------------------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    # --------------------------------------------------------
    # Metadata filtering
    # --------------------------------------------------------

    where = build_where_filter(
        company=company,
        year=year,
        form=form,
    )

    # --------------------------------------------------------
    # Vector search
    # --------------------------------------------------------

    search_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": candidate_k,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where is not None:

        search_kwargs["where"] = where

    results = collection.query(
        **search_kwargs
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        score = 0.0

        # --------------------------------------------
        # Keyword boost
        # --------------------------------------------

        document_lower = document.lower()

        matched_keywords = 0

        for keyword in keywords:

            if keyword.lower() in document_lower:

                matched_keywords += 1

        # Each matched keyword gives a proportional boost.
        #
        # Lower distance = better.
        #
        # We convert that into a relevance score.

        semantic_score = 1 - distance

        total_boost = 0.0

        if keywords:
            # Normalize keyword boost by the number of total query keywords
            # to prevent long queries from artificially inflating scores.
            keyword_boost = (
                (matched_keywords / len(keywords)) * 0.2
            )
            total_boost += keyword_boost

        # --------------------------------------------
        # Strong Match Boost (Golden Pattern)
        # --------------------------------------------
        # High precision boost for the actual answer pattern:
        # e.g., "total net sales" or "total revenue" followed by a currency value.
        if re.search(r"(total net sales|total revenue|net sales)\s*[:\$-]?\s*[\d,.]+", document_lower):
            total_boost += 1.5

        # --------------------------------------------
        # Table Header Boost
        # --------------------------------------------
        # Lower weight for generic headers to prevent noise from outranking answers.
        generic_headers = [
            "consolidated statements of operations",
            "consolidated balance sheets",
            "consolidated statements of shareholders' equity",
        ]
        for header in generic_headers:
            if header in document_lower:
                total_boost += 0.1
                break

        # --------------------------------------------
        # Financial Value Boost
        # --------------------------------------------
        # Give an extra boost to chunks that look like financial tables
        # (contain '$' and numbers)
        if "$" in document and any(char.isdigit() for char in document):
            total_boost += 0.05

        # --------------------------------------------
        # Contextual Penalty (De-prioritize Expenses for Revenue queries)
        # --------------------------------------------
        if keywords:
            is_revenue_query = any(kw in ["revenue", "net sales", "total net sales", "sales"] for kw in keywords)
            if is_revenue_query:
                expense_keywords = ["operating expenses", "provision for income taxes", "research and development", "selling, general and administrative"]
                if any(exp_kw in document_lower for exp_kw in expense_keywords):
                    total_boost -= 0.5

        if year is not None:

            filing_date = str(
                metadata.get(
                    "filing_date",
                    ""
                )
            )

            if filing_date.startswith(
                str(year)
            ):

                total_boost += 0.05

        score = (
            semantic_score + total_boost
        )

        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "score": score,
                "matched_keywords": matched_keywords,
            }
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return candidates[:top_k]


def synthesize_answer(
    query: str,
    results: list[dict],
) -> str | None:
    """Uses an LLM to synthesize the search results into a natural language answer."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Construct context from retrieved chunks
    context_parts = []
    for i, res in enumerate(results, start=1):
        context_parts.append(f"[{i}] {res['document']}")

    context_text = "\n\n".join(context_parts)

    prompt = (
        "You are a professional financial analyst. Answer the user's query based "
        "ONLY on the provided SEC filing excerpts. If the answer is not contained "
        "in the excerpts, clearly state that you do not have enough information. "
        "Be precise, include numeric values exactly as they appear, and cite the "
        "context using [number].\n\n"
        f"Context:\n{context_text}\n\n"
        f"Query: {query}\n\n"
        "Answer:"
    )

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return None


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    query: str,
    results: list[dict],
    synthesized_answer: str | None = None,
):

    print()
    print("=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    if synthesized_answer:
        print("\nAI SUMMARY:")
        print("-" * 20)
        print(synthesized_answer)
        print("-" * 20)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        print("\n(Synthesis failed or returned no answer)")

    print()
    print(
        f"Query: {query}"
    )

    for index, result in enumerate(
        results,
        start=1,
    ):

        metadata = result["metadata"]

        print()
        print("-" * 80)

        print(
            f"Result #{index}"
        )

        print(
            f"Chunk ID:     "
            f"{metadata.get('chunk_id', 'N/A')}"
        )

        print(
            f"Company:      "
            f"{metadata.get('company', 'N/A')}"
        )

        print(
            f"Filing date:  "
            f"{metadata.get('filing_date', 'N/A')}"
        )

        print(
            f"Form:         "
            f"{metadata.get('form', 'N/A')}"
        )

        print(
            f"Document:     "
            f"{metadata.get('document', 'N/A')}"
        )

        print(
            f"Chunk index:  "
            f"{metadata.get('chunk_index', 'N/A')}"
        )

        print(
            f"Distance:     "
            f"{result['distance']:.6f}"
        )

        print(
            f"Score:        "
            f"{result['score']:.6f}"
        )

        if result["matched_keywords"]:

            print(
                f"Keyword matches: "
                f"{result['matched_keywords']}"
            )

        print()

        print("Text:")

        print(
            result["document"]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Search SEC filing embeddings."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to display",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of vector candidates before reranking",
    )

    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Filter by company",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter by year",
    )

    parser.add_argument(
        "--form",
        type=str,
        default=None,
        choices=[
            "10-K",
            "10-Q",
        ],
        help="Filter by SEC form",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SEC RAG SEARCH")
    print("=" * 80)

    model = load_model()

    collection = load_collection()

    results = search(
        collection=collection,
        model=model,
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        company=args.company,
        year=args.year,
        form=args.form,
    )

    # Synthesis step
    synthesized_answer = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        synthesized_answer = synthesize_answer(
            query=args.query,
            results=results,
        )
    elif os.environ.get("ANTHROPIC_API_KEY") is None:
        # Optional: notify user that synthesis is disabled
        pass

    print_results(
        query=args.query,
        results=results,
        synthesized_answer=synthesized_answer,
    )


if __name__ == "__main__":
    main()
