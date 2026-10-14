from __future__ import annotations

import argparse
import os
import re
import hashlib
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import chromadb
import ftfy
from chromadb.api.types import Where
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from groq import Groq

from src import config

SEARCH_CACHE_VERSION = 2
_LEXICAL_INDEX_CACHE: dict[
    tuple[str, int],
    tuple[list[str], list[str], list[Any], list[list[str]], BM25Okapi],
] = {}

# ============================================================
# LLM CLIENT WRAPPER
# ============================================================

class LLMClient:
    """Wrapper to handle LLM calls via Groq Cloud API."""
    def __init__(self, api_key: Optional[str] = None):
        self.client = Groq(api_key=api_key) if api_key else None

    async def complete(self, prompt: str, system_prompt: str = "", max_tokens: int = 500) -> Optional[str]:
        if not self.client:
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # Groq's client is synchronous, run in thread to prevent blocking async loop
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                messages=messages,
                model=config.LLM_MODEL,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception as e:
            print(f"Groq LLM Error: {e}")
            return None

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
        "db_version": db_version,
        "search_version": SEARCH_CACHE_VERSION,
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

    # Ensure results are JSON serializable (convert float32 to float)
    serializable_results = []
    for res in results:
        item = res.copy()
        if "distance" in item:
            item["distance"] = float(item["distance"])
        if "rerank_score" in item:
            item["rerank_score"] = float(item["rerank_score"])
        serializable_results.append(item)

    cache_data = {
        "metadata": {"params": params, "timestamp": datetime.now().isoformat()},
        "data": {"final_results": serializable_results, "synthesized_answer": answer, "audit_report": audit}
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
    if re.search(r"\b(?:fiscal year|annual report|annual filing|year ended)\b", query_lower):
        return "10-K"
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


def get_lexical_index(
    collection: chromadb.Collection,
) -> tuple[list[str], list[str], list[Any], list[list[str]], BM25Okapi]:
    collection_id = str(getattr(collection, "id", id(collection)))
    collection_count = collection.count()
    cache_key = (collection_id, collection_count)
    cached = _LEXICAL_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    indexed = collection.get(include=["documents", "metadatas"])
    ids = indexed["ids"]
    documents = indexed["documents"] or []
    metadatas = indexed["metadatas"] or [{} for _ in documents]
    if not documents:
        raise ValueError("The SEC filings collection contains no searchable documents.")

    tokenized_documents = [tokenize(document) for document in documents]
    index = (
        ids,
        documents,
        metadatas,
        tokenized_documents,
        BM25Okapi(tokenized_documents),
    )
    for old_key in [key for key in _LEXICAL_INDEX_CACHE if key[0] == collection_id]:
        del _LEXICAL_INDEX_CACHE[old_key]
    _LEXICAL_INDEX_CACHE[cache_key] = index
    return index


def get_period_year(metadata: dict) -> Optional[int]:
    document_id = str(metadata.get("document", ""))
    match = re.search(r"\b(20\d{2})\d{4}\b", document_id)
    return int(match.group(1)) if match else None


def expand_financial_query(query: str) -> str:
    query_lower = query.lower()
    expansions = {
        "revenue": ("revenue", "net sales", "total net sales", "total revenue", "sales"),
        "profit": ("profit", "net income", "earnings"),
        "cash": (
            "cash flow",
            "cash flows",
            "cash from operations",
            "net cash from operations",
            "operating activities",
        ),
        "risk": ("risk factors", "Item 1A", "risks and uncertainties"),
    }
    terms = [query]
    for trigger, synonyms in expansions.items():
        if trigger in query_lower or (
            trigger == "revenue" and "sales" in query_lower
        ):
            terms.extend(synonyms)
    return " ".join(dict.fromkeys(terms))


def financial_evidence_bonus(
    query: str,
    document: str,
    metadata: dict,
    year: Optional[int],
) -> float:
    text = document.lower()
    query_lower = query.lower()
    bonus = 0.0
    has_numbers = re.search(r"\d", text) is not None

    if has_numbers and any(term in query_lower for term in ("revenue", "sales")):
        table_revenue_row = re.search(
            r"\b(?:total net sales|total revenue|net sales)\b"
            r"(?:\s*\|\s*(?:total net sales|total revenue|net sales)){0,2}"
            r"\s*\|\s*\$\s*\|\s*[\d,]+",
            text,
        )
        revenue_with_amount = re.search(
            r"\b(?:revenue|net sales)\b.{0,100}?\$\s*[\d,.]+",
            text,
            flags=re.DOTALL,
        )
        if table_revenue_row:
            bonus += 0.30
        elif revenue_with_amount:
            bonus += 0.12

    if (
        has_numbers
        and re.search(r"\b(?:cash from operations|net cash from operations|cash flows?)\b", text)
        and "cash" in query_lower
    ):
        bonus += 0.10
        if re.search(
            r"\b(?:cash from operations|net cash from operations)\b"
            r".{0,100}?\$\s*[\d,.]+",
            text,
            flags=re.DOTALL,
        ):
            bonus += 0.20

    if (
        re.search(r"\b(?:risk factors|item 1a)\b", text)
        and any(term in query_lower for term in ("risk", "risks"))
    ):
        bonus += 0.08

    if year is not None and get_period_year(metadata) == year:
        bonus += 0.12

    return bonus

def build_where_filter(
    company: Optional[str],
    year: Optional[int],
    form: Optional[str],
) -> Where | None:
    conditions = []
    if company:
        conditions.append({"company": company})
    if form:
        conditions.append({"form": form})
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

async def generate_hypothetical_document(client: Optional[LLMClient], query: str) -> str:
    if not client: return query
    prompt = (
        "You are a professional financial analyst. Write a plausible, professional-sounding answer "
        "to the following query as if it were extracted from an SEC filing (10-K or 10-Q). "
        "Use formal financial language and include placeholders for numbers (e.g., '$X million').\n\n"
        f"Query: {query}\n\n"
        "Hypothetical SEC Excerpt:"
    )
    return await client.complete(prompt) or query

async def expand_query(client: Optional[LLMClient], query: str) -> List[str]:
    if not client: return [query]
    prompt = (
        "You are a search optimization expert for financial filings. Rewrite the following user query "
        "into 3-5 semantically different versions to ensure a comprehensive vector search. "
        "Focus on terminology like 'revenue' vs 'net sales'.\n\n"
        f"Query: {query}\n\n"
        "Provide only the rewritten queries, one per line."
    )
    res = await client.complete(prompt, max_tokens=200)
    if not res: return [query]
    expanded = res.strip().split("\n")
    return [query] + [q.strip() for q in expanded if q.strip()]

async def verify_relevance(client: Optional[LLMClient], query: str, results: List[Dict]) -> Tuple[bool, str]:
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
    res = await client.complete(prompt, max_tokens=150)
    if not res: return True, ""
    response = res.strip()
    return response.upper().startswith("YES"), response

async def refine_query(client: Optional[LLMClient], original_query: str, feedback: str) -> str:
    if not client: return original_query
    prompt = (
        f"The previous search for '{original_query}' failed. The gap was: {feedback}. "
        "Rewrite the query to specifically target the missing information using SEC terminology."
    )
    return await client.complete(prompt, max_tokens=100) or original_query

async def verify_citations(client: Optional[LLMClient], answer: str, results: List[Dict]) -> Tuple[bool, str]:
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
    skeptic_report = await client.complete(skeptic_prompt, max_tokens=400) or "Skeptic audit failed"

    if not hallucinations and skeptic_report.upper() == "VERIFIED":
        return True, "VERIFIED"

    report = ("EXACT MATCH ERRORS:\n" + "\n".join(hallucinations) + "\n\n") if hallucinations else ""
    report += "SKEPTIC ANALYSIS:\n" + skeptic_report
    return False, report

async def synthesize_answer_with_feedback(client: Optional[LLMClient], query: str, results: List[Dict], feedback: Optional[str] = None) -> Optional[str]:
    if not client: return None
    context_text = "\n\n".join(
        f"[{i}] {res['metadata'].get('company')} "
        f"{res['metadata'].get('form')} filed {res['metadata'].get('filing_date')}\n"
        f"{res['document']}"
        for i, res in enumerate(results, start=1)
    )
    system_prompt = (
        "You are a careful financial analyst. Answer only from the supplied SEC filing excerpts. "
        "Answer the question directly and cite every factual claim with its source number. "
        "The only valid citation format is the bracketed source number shown before an excerpt, "
        "for example [1] or [2]. Never invent page, line, or range citations such as L1-L4. "
        "For financial figures, preserve the exact number, units, fiscal period, and label shown in the source; "
        "do not infer a number from a heading or combine values from different periods. "
        "Use plain ASCII punctuation and spaces. "
        "If the excerpts do not show the requested fact, say that the available excerpts are insufficient."
    )
    user_content = f"Context:\n{context_text}\n\nQuery: {query}\n\n"
    if feedback:
        user_content += f"IMPORTANT: Correct these audited errors:\n{feedback}\n\n"
    user_content += "Answer:"
    answer = await client.complete(user_content, system_prompt=system_prompt, max_tokens=1000)
    if answer is None:
        return None

    answer = ftfy.fix_text(answer)
    answer = answer.replace("âs", "'s").replace("â¯", " ")
    answer = answer.replace("\ufffd", "")
    answer = re.sub(r"ã(?=\s|[.,;:!?]|$)", "", answer)
    answer = re.sub(r"\bL\d+(?:\s*-\s*L\d+)?\b", "", answer)
    answer = re.sub(r"[ \t]+", " ", answer)
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    return answer.strip()

async def synthesize_answer(client: Optional[LLMClient], query: str, results: List[Dict]) -> Optional[str]:
    return await synthesize_answer_with_feedback(client, query, results)

# ============================================================
# SEARCH ENGINE
# ============================================================

async def search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    reranker: CrossEncoder,
    client: Optional[LLMClient],
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_k: int = config.DEFAULT_CANDIDATE_HITS,
    companies: Optional[str] = None,
    year: Optional[int] = None,
    form: Optional[str] = None,
) -> List[Dict]:

    detected_companies = detect_companies(query)
    company = companies or (detected_companies[0] if detected_companies else None)
    year = year or detect_year(query)
    form = form or detect_form(query)
    keywords = extract_keywords(query)
    candidate_limit = max(1, candidate_k)

    query_embedding = get_embedding_cached(query, model)
    if len(query_embedding) != 384:
        raise ValueError(
            f"CRITICAL DIMENSION MISMATCH: Database expects 384, but model "
            f"{config.EMBEDDING_MODEL} produced {len(query_embedding)}. "
            "Please ensure config.EMBEDDING_MODEL is 'BAAI/bge-small-en-v1.5'."
        )

    ids, corpus_documents, corpus_metadatas, tokenized_documents, bm25 = get_lexical_index(collection)
    metadata_by_id = dict(zip(ids, corpus_metadatas))
    document_by_id = dict(zip(ids, corpus_documents))

    eligible_ids = [
        chunk_id
        for chunk_id in ids
        if (company is None or metadata_by_id[chunk_id].get("company") == company)
        and (form is None or metadata_by_id[chunk_id].get("form") == form)
    ]
    if year is not None and form == "10-K":
        matching_period_ids = [
            chunk_id
            for chunk_id in eligible_ids
            if get_period_year(metadata_by_id[chunk_id]) == year
        ]
        if matching_period_ids:
            eligible_ids = matching_period_ids

    if not eligible_ids:
        return []

    where = build_where_filter(company, None, form)

    def _do_query():
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=min(candidate_limit, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    vector_results = await asyncio.to_thread(_do_query)
    vector_ids_result = vector_results.get("ids")
    vector_ids = vector_ids_result[0] if vector_ids_result and vector_ids_result[0] else []
    vector_distances_result = vector_results.get("distances")
    vector_distances = (
        vector_distances_result[0]
        if vector_distances_result and vector_distances_result[0]
        else []
    )
    eligible_id_set = set(eligible_ids)

    semantic_ranked_ids = [
        chunk_id for chunk_id in vector_ids if chunk_id in eligible_id_set
    ]
    semantic_distances = {
        chunk_id: float(distance)
        for chunk_id, distance in zip(vector_ids, vector_distances)
        if chunk_id in eligible_id_set
    }

    expanded_query = expand_financial_query(query)
    query_tokens = set(tokenize(expanded_query))
    query_tokens.difference_update(
        {"what", "was", "the", "did", "for", "in", "of", "how", "does", "report", "reported", "were", "is", "a", "an", "to", "from", "and", "fiscal", "year"}
    )
    if year is not None:
        query_tokens.discard(str(year))
    if company:
        query_tokens.difference_update(tokenize(company))
    bm25_scores = bm25.get_scores(tokenize(expanded_query))
    lexical_ranked_ids = [
        ids[index]
        for index in sorted(
            (index for index in range(len(ids)) if ids[index] in eligible_id_set),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )
        if bm25_scores[index] > 0
        or query_tokens.intersection(tokenized_documents[index])
    ][:candidate_limit]

    candidate_ids = list(dict.fromkeys(semantic_ranked_ids + lexical_ranked_ids))
    if not candidate_ids:
        return []

    fused_scores: dict[str, float] = {}
    for ranked_ids in (semantic_ranked_ids, lexical_ranked_ids):
        for rank, chunk_id in enumerate(ranked_ids):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank + 1)

    rerank_scores: dict[str, float] = {}
    if reranker:
        pairs = [[query, document_by_id[chunk_id]] for chunk_id in candidate_ids]
        predicted_scores = reranker.predict(pairs)
        rerank_scores = {
            chunk_id: float(score)
            for chunk_id, score in zip(candidate_ids, predicted_scores)
        }
        reranker_ranked_ids = sorted(
            rerank_scores,
            key=rerank_scores.__getitem__,
            reverse=True,
        )
        for rank, chunk_id in enumerate(reranker_ranked_ids):
            fused_scores[chunk_id] += 1.0 / (60 + rank + 1)

    candidates = []
    for chunk_id in candidate_ids:
        document = document_by_id[chunk_id]
        metadata = dict(metadata_by_id[chunk_id])
        metadata.setdefault("chunk_id", chunk_id)
        matched = sum(1 for keyword in keywords if keyword.lower() in document.lower())
        score = fused_scores[chunk_id] + financial_evidence_bonus(
            query,
            document,
            metadata,
            year,
        )
        candidates.append(
            {
                "document": document,
                "metadata": metadata,
                "distance": semantic_distances.get(chunk_id),
                "score": float(score),
                "rerank_score": rerank_scores.get(chunk_id, float(score)),
                "matched_keywords": matched,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]

async def run_agentic_search(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    reranker: CrossEncoder,
    client: Optional[LLMClient],
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_k: int = config.DEFAULT_CANDIDATE_HITS,
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
    first_distance = first_results[0].get("distance") if first_results else None
    if first_distance is not None and first_distance < config.CONFIDENCE_THRESHOLD:
        answer = await synthesize_answer(client, query, first_results)
        if answer is None:
            report = "Answer synthesis unavailable"
            save_cache(cache_key, first_results, answer, report, {"query": query})
            return first_results, answer, report

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
            cid = c["metadata"].get("chunk_id") or c["document"]
            if cid not in seen_ids:
                unique.append(c)
                seen_ids.add(cid)

        if reranker and unique:
            pairs = [[current_query, c["document"]] for c in unique]
            scores = reranker.predict(pairs)
            reranked = sorted(
                range(len(unique)),
                key=lambda i: float(scores[i]),
                reverse=True,
            )
            for rank, i in enumerate(reranked):
                unique[i]["rerank_score"] = float(scores[i])
                unique[i]["score"] += 1.0 / (60 + rank + 1)
            unique.sort(key=lambda item: item["score"], reverse=True)

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
        if synthesized_answer is None:
            audit_report = "Answer synthesis unavailable after citation feedback"
            break

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

    # Deduplicate sources for display
    unique_sources = {}
    for i, res in enumerate(results, 1):
        meta = res['metadata']
        source_key = (meta.get('company'), meta.get('form'), meta.get('filing_date'))
        if source_key not in unique_sources:
            unique_sources[source_key] = []
        unique_sources[source_key].append(i)

    for idx, (source_meta, chunk_indices) in enumerate(unique_sources.items(), 1):
        company, form, date = source_meta
        indices_str = ", ".join(map(str, chunk_indices))
        print(f"\nSource #{idx} [{indices_str}]: {company} {form} filed {date}")
        # Optionally show the first chunk's text as a snippet
        first_res = results[chunk_indices[0]-1]
        print(f"Text snippet: {first_res['document'][:200]}...")

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

    api_key = os.environ.get("GROQ_API_KEY")
    llm_client = LLMClient(api_key=api_key) if api_key else None

    results, answer, audit = await run_agentic_search(
        collection, model, reranker, llm_client, args.query,
        top_k=args.top_k,
        companies=args.company,
        year=args.year,
        form=args.form
    )

    print_results(args.query, results, answer)
    if audit: print(f"\n{'!'*80}\nAUDIT: {audit}\n{'!'*80}")

if __name__ == "__main__":
    asyncio.run(main())
