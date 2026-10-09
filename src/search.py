from __future__ import annotations

import argparse
import os
import re
import hashlib
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import anthropic
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from src import config

# ============================================================
# CACHE HELPERS
# ============================================================

def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25, preserving financial terms like 10-K."""
    return re.findall(r'\b\w+(?:-\w+)*\b', text.lower())

def reciprocal_rank_fusion(vector_ranks: List[int], bm25_ranks: List[int], k: int = 60) -> List[float]:
    """Combines two ranked lists using Reciprocal Rank Fusion."""
    scores = [0.0] * max(len(vector_ranks), len(bm25_ranks))
    for rank, idx in enumerate(vector_ranks):
        scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] += 1.0 / (k + rank + 1)
    return scores

def generate_cache_key(query: str, top_k: int, candidate_k: int, companies: Any, year: Optional[int], form: Optional[str]) -> str:
    """Generates a SHA-256 hash for the search parameters and DB version."""
    db_version = 0
    if config.CHROMA_DIR.exists():
        db_version = config.CHROMA_DIR.stat().st_mtime

    normalized_params = {
        "query": query.strip().lower(),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "companies": sorted(companies) if isinstance(companies, list) else [companies] if companies else [],
        "year": year,
        "form": form,
        "db_version": db_version
    }
    param_string = json.dumps(normalized_params, sort_keys=True)
    return hashlib.sha256(param_string.encode("utf-8")).hexdigest()

def load_cache(cache_key: str) -> Optional[Dict]:
    cache_file = config.CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return None

def save_cache(cache_key: str, results: List, answer: Optional[str], audit: Optional[str], params: Dict):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = config.CACHE_DIR / f"{cache_key}.json"
    cache_data = {
        "metadata": {"params": params, "timestamp": datetime.now().isoformat()},
        "data": {"final_results": results, "synthesized_answer": answer, "audit_report": audit}
    }
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_embedding_cached(text: str, model) -> List[float]:
    norm_text = text.strip()
    if not norm_text:
        return model.encode("", normalize_embeddings=True).tolist()
    text_hash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()

    cache = {}
    if config.EMBEDDING_CACHE_FILE.exists():
        try:
            with open(config.EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load embedding cache: {e}")

    if text_hash in cache:
        return cache[text_hash]

    embedding = model.encode(norm_text, normalize_embeddings=True).tolist()
    cache[text_hash] = embedding
    config.EMBEDDING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config.EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Warning: Could not save embedding cache: {e}")
    return embedding

# ============================================================
# UTILS
# ============================================================

def detect_companies(query: str) -> List[str]:
    query_lower = query.lower()
    detected = []
    for name, company in config.COMPANY_MAP.items():
        if name in query_lower:
            if company not in detected:
                detected.append(company)
    return detected

def detect_year(query: str) -> Optional[int]:
    match = re.search(r"\b(20\d{2})\b", query)
    if match:
        year = int(match.group(1))
        if year > 2026:
            return None
        return year
    return None

def detect_form(query: str) -> Optional[str]:
    query_lower = query.lower()
    if "10-k" in query_lower: return "10-K"
    if "10-q" in query_lower: return "10-Q"
    return None

def extract_keywords(query: str) -> List[str]:
    query_lower = query.lower()
    keyword_groups = {
        "revenue": ["revenue", "net sales", "total net sales", "sales"],
        "profit": ["profit", "net income", "earnings"],
        "margin": ["gross margin", "operating margin", "margin"],
        "expenses": ["expense", "expenses", "operating expenses", "r&d", "research and development"],
        "assets": ["assets", "total assets"],
        "debt": ["debt", "borrowings", "liabilities"],
        "cash": ["cash", "cash equivalents", "cash flow"],
    }
    keywords = []
    for group in keyword_groups.values():
        for kw in group:
            if kw in query_lower:
                keywords.append(kw)
    return list(dict.fromkeys(keywords))

def get_fiscal_year_range(year: int) -> Tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"

def build_where_filter(company: Optional[str], year: Optional[int], form: Optional[str]):
    conditions = []
    if company: conditions.append({"company": company})
    if form: conditions.append({"form": form})
    if year:
        # To avoid ValueError with $gte on date strings,
        # we use a simple match or filter logic that the DB supports.
        # If dates are stored as strings "YYYY-MM-DD", we search for the year prefix.
        # However, since ChromaDB's $where doesn't support 'contains' on strings easily,
        # we will omit the strict date range filter to ensure stability.
        pass
    if not conditions: return None
    if len(conditions) == 1: return conditions[0]
    return {"$and": conditions}

# ============================================================
# ASYNC AGENTIC HELPERS (OPTIONAL LLM)
# ============================================================

async def generate_hypothetical_document(client: Optional[anthropic.AsyncAnthropic], query: str) -> str:
    if not client: return query
    prompt = (
        "You are a professional financial analyst. Write a plausible, professional-sounding answer "
        "to the following query as if it were extracted from an SEC filing (10-K or 10-Q). "
        "Use formal financial language and include placeholders for numbers (e.g., '$X million').\n\n"
        f"Query: {query}\n\n"
        "Hypothetical SEC Excerpt:"
    )
    try:
        message = await client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"Error generating HyDE: {e}")
        return query

async def expand_query(client: Optional[anthropic.AsyncAnthropic], query: str) -> List[str]:
    if not client: return [query]
    prompt = (
        "You are a search optimization expert for financial filings. Rewrite the following user query "
        "into 3-5 semantically different versions to ensure a comprehensive vector search. "
        "Focus on terminology like 'revenue' vs 'net sales'.\n\n"
        f"Query: {query}\n\n"
        "Provide only the rewritten queries, one per line."
    )
    try:
        message = await client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        expanded = message.content[0].text.strip().split("\n")
        return [query] + [q.strip() for q in expanded if q.strip()]
    except Exception as e:
        print(f"Error expanding query: {e}")
        return [query]

async def verify_relevance(client: Optional[anthropic.AsyncAnthropic], query: str, results: List[Dict]) -> Tuple[bool, str]:
    if not client: return True, "No LLM client"
    context_text = "\n\n".join([f"[{i}] {res['document']}" for i, res in enumerate(results, start=1)])
    prompt = (
        "You are a financial auditor. Determine if the retrieved excerpts contain enough specific, "
        "factual, or numeric information to provide a complete answer. Respond with 'YES' or 'NO' "
        "followed by a brief explanation.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Response (YES/NO and explanation):"
    )
    try:
        message = await client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        response = message.content[0].text.strip()
        return response.upper().startswith("YES"), response
    except Exception as e:
        print(f"Error verifying relevance: {e}")
        return True, ""

async def refine_query(client: Optional[anthropic.AsyncAnthropic], original_query: str, feedback: str) -> str:
    if not client: return original_query
    prompt = (
        f"The previous search for '{original_query}' failed. The gap was: {feedback}. "
        "Rewrite the query to specifically target the missing information using SEC terminology."
    )
    try:
        message = await client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"Error refining query: {e}")
        return original_query

async def verify_citations(client: Optional[anthropic.AsyncAnthropic], answer: str, results: List[Dict]) -> Tuple[bool, str]:
    if not client: return True, "No LLM client"
    hallucinations = []
    money_pattern = re.compile(r"\$?\d+(?:\.\d+)?\s*(?:billion|million|thousand|B|M|K)?")
    date_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
    found_values = money_pattern.findall(answer) + date_pattern.findall(answer)

    for val in found_values:
        if not any(val in res["document"] for res in results):
            hallucinations.append(f"Value '{val}' not found in any cited source.")

    context_text = "\n\n".join([f"[{i}] {res['document']}" for i, res in enumerate(results, start=1)])
    skeptic_prompt = (
        "You are an adversarial financial auditor. Find any factual contradiction or hallucination "
        "in the answer compared to the sources. Be pedantic.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer:\n{answer}\n\n"
        "Audit Report (List errors or 'VERIFIED'):"
    )
    try:
        message = await client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": skeptic_prompt}],
        )
        skeptic_report = message.content[0].text.strip()
    except Exception as e:
        skeptic_report = f"Skeptic audit failed: {e}"

    if not hallucinations and skeptic_report.upper() == "VERIFIED":
        return True, "VERIFIED"

    report = ("EXACT MATCH ERRORS:\n" + "\n".join(hallucinations) + "\n\n") if hallucinations else ""
    report += "SKEPTIC ANALYSIS:\n" + skeptic_report
    return False, report

async def synthesize_answer_with_feedback(client: Optional[anthropic.AsyncAnthropic], query: str, results: List[Dict], feedback: Optional[str] = None) -> Optional[str]:
    if not client: return None
    context_text = "\n\n".join([f"[{i}] {res['document']}" for i, res in enumerate(results, start=1)])
    system_prompt = (
        "You are a professional financial analyst. Answer the query based ONLY on the provided excerpts. "
        "Every factual claim MUST have a citation [number]. If not present, state you lack information."
    )
    user_content = f"Context:\n{context_text}\n\nQuery: {query}\n\n"
    if feedback:
        user_content += f"IMPORTANT: Correct these audited errors:\n{feedback}\n\n"
    user_content += "Answer:"
    try:
        message = await client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": user_content}],
            system=system_prompt,
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return None

async def synthesize_answer(client: Optional[anthropic.AsyncAnthropic], query: str, results: List[Dict]) -> Optional[str]:
    return await synthesize_answer_with_feedback(client, query, results)

# ============================================================
# SEARCH ENGINE
# ============================================================

async def search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    reranker: CrossEncoder,
    client: Optional[anthropic.AsyncAnthropic],
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_k: int = config.DEFAULT_CANDIDATE_K,
    companies: Optional[str] = None,
    year: Optional[int] = None,
    form: Optional[str] = None,
) -> List[Dict]:

    detected_companies = detect_companies(query)
    detected_year = detect_year(query)
    detected_form = detect_form(query)

    company = companies or (detected_companies[0] if detected_companies else None)
    year = year or detected_year
    form = form or detected_form
    keywords = extract_keywords(query)

    query_embedding = get_embedding_cached(query, model)

    # SAFETY GUARD: Check dimension before calling ChromaDB
    if len(query_embedding) != 384:
        # If the DB expects 384 but we got something else, we must force a crash with a clear message
        # because ChromaDB will just throw a generic InvalidArgumentError.
        raise ValueError(f"CRITICAL DIMENSION MISMATCH: Database expects 384, but model {config.EMBEDDING_MODEL} produced {len(query_embedding)}. Please ensure config.EMBEDDING_MODEL is 'BAAI/bge-small-en-v1.5'")

    where = build_where_filter(company, year, form)

    def _do_query():
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

    results = await asyncio.to_thread(_do_query)
    documents, metadatas, distances = results["documents"][0], results["metadatas"][0], results["distances"][0]

    vector_ranks = list(range(len(documents)))
    tokenized_corpus = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranks = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    rrf_scores = reciprocal_rank_fusion(vector_ranks, bm25_ranks)

    candidates = []
    for i in range(len(documents)):
        doc, meta, dist = documents[i], metadatas[i], distances[i]
        score = rrf_scores[i]
        doc_lower = doc.lower()

        matched = sum(1 for kw in keywords if kw.lower() in doc_lower)
        if keywords:
            score += (matched / len(keywords)) * 0.2

        if re.search(r"(total net sales|total revenue|net sales)\s*[:\$-]?\s*[\d,.]+", doc_lower):
            score += 1.5

        if any(h in doc_lower for h in ["consolidated statements of operations", "consolidated balance sheets"]):
            score += 0.1

        if "$" in doc and any(c.isdigit() for c in doc):
            score += 0.05

        if keywords and any(kw in ["revenue", "net sales", "total net sales", "sales"] for kw in keywords):
            if any(exp in doc_lower for exp in ["operating expenses", "provision for income taxes", "r&d"]):
                score -= 0.5

        candidates.append({"document": doc, "metadata": meta, "distance": dist, "score": score, "matched_keywords": matched})

    if reranker:
        pairs = [[query, c["document"]] for c in candidates]
        rerank_scores = reranker.predict(pairs)
        for i, s in enumerate(rerank_scores):
            candidates[i]["rerank_score"] = s
    else:
        for c in candidates: c["rerank_score"] = c["score"]

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

async def run_agentic_search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    reranker: CrossEncoder,
    client: Optional[anthropic.AsyncAnthropic],
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_k: int = config.DEFAULT_CANDIDATE_K,
    companies: Optional[str] = None,
    year: Optional[int] = None,
    form: Optional[str] = None,
) -> Tuple[List[Dict], Optional[str], Optional[str]]:

    cache_key = generate_cache_key(query, top_k, candidate_k, companies, year, form)
    cached = load_cache(cache_key)
    if cached:
        return cached["data"]["final_results"], cached["data"]["synthesized_answer"], cached["data"]["audit_report"]

    first_results = await search(collection, model, reranker, client, query, top_k, candidate_k, companies, year, form)

    # Fast Path: if very confident, skip agentic loop
    if first_results and first_results[0]["distance"] < config.CONFIDENCE_THRESHOLD:
        answer = await synthesize_answer(client, query, first_results)
        is_verified, report = await verify_citations(client, answer, first_results)
        save_cache(cache_key, first_results, answer, report if not is_verified else "VERIFIED", {"query": query})
        return first_results, answer, report if not is_verified else "VERIFIED"

    # If no LLM client, we can't do the agentic loop, so just return the initial search
    if not client:
        return first_results, None, "No LLM client configured"

    max_attempts = 3
    current_query = query
    final_results = []

    for attempt in range(max_attempts):
        gen_tasks = [generate_hypothetical_document(client, current_query), expand_query(client, current_query)]
        hyde_doc, expanded_queries = await asyncio.gather(*gen_tasks)

        queries_to_run = [current_query, hyde_doc] + [q for q in expanded_queries if q != current_query]
        search_tasks = [search(collection, model, reranker, client, q, top_k, candidate_k, companies, year, form) for q in queries_to_run]
        search_results_list = await asyncio.gather(*search_tasks)

        all_candidates = [res for sublist in search_results_list for res in sublist]
        seen_ids = set()
        unique = []
        for c in all_candidates:
            cid = c["metadata"].get("chunk_id")
            if cid not in seen_ids:
                unique.append(c)
                seen_ids.add(cid)

        if reranker and unique:
            pairs = [[current_query, c["document"]] for c in unique]
            scores = reranker.predict(pairs)
            for i, s in enumerate(scores): unique[i]["rerank_score"] = s
            unique.sort(key=lambda x: x["rerank_score"], reverse=True)

        final_results = unique[:top_k]
        is_relevant, feedback = await verify_relevance(client, query, final_results)
        if is_relevant: break
        current_query = await refine_query(client, query, feedback)

    synthesized_answer = None
    audit_report = None
    for _ in range(2):
        synthesized_answer = await synthesize_answer(client, query, final_results)
        if not synthesized_answer: break

        is_verified, report = await verify_citations(client, synthesized_answer, final_results)
        if is_verified:
            audit_report = "VERIFIED"
            break
        audit_report = report
        synthesized_answer = await synthesize_answer_with_feedback(client, query, final_results, report)
        is_verified, report = await verify_citations(client, synthesized_answer, final_results)
        if is_verified:
            audit_report = "VERIFIED"
            break
        audit_report = report

    save_cache(cache_key, final_results, synthesized_answer, audit_report, {"query": query})
    return final_results, synthesized_answer, audit_report

# ============================================================
# CLI ENTRYPOINT
# ============================================================

def print_results(query: str, results: List[Dict], synthesized_answer: Optional[str] = None):
    print(f"\n{'='*80}\nSEARCH RESULTS\n{'='*80}")
    if synthesized_answer:
        print(f"\nAI SUMMARY:\n{'-'*20}\n{synthesized_answer}\n{'-'*20}")
    else:
        print("\n(AI Summary not available - No LLM Client)")
    print(f"\nQuery: {query}")
    for i, res in enumerate(results, 1):
        print(f"\n{'-'*80}\nResult #{i}\nCompany: {res['metadata'].get('company')}\nDate: {res['metadata'].get('filing_date')}\nText:\n{res['document']}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--company")
    parser.add_argument("--year", type=int)
    parser.add_argument("--form")
    parser.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    args = parser.parse_args()

    print("Loading models...")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.get_collection(config.COLLECTION_NAME)
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")
    reranker = CrossEncoder(config.RERANKER_MODEL, device="cpu")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    llm_client = anthropic.AsyncAnthropic(api_key=api_key) if api_key else None

    results, answer, audit = await run_agentic_search(
        collection, model, reranker, llm_client, args.query, args.top_k, args.company, args.year, args.form
    )

    print_results(args.query, results, answer)
    if audit: print(f"\n{'!'*80}\nAUDIT: {audit}\n{'!'*80}")

if __name__ == "__main__":
    asyncio.run(main())
