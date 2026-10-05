# SEC-RAG System Improvement Roadmap

This document outlines the technical improvements required to evolve the SEC-RAG system from a basic prototype to a production-grade financial intelligence tool.

## 1. Retrieval Precision & Accuracy

### 1.1 Strict Temporal Filtering
**Problem:** Current implementation allows "temporal leakage" where data from adjacent years (e.g., 2023) is returned for a specific year query (e.g., 2024).
**Proposed Solution:**
- Replace the `startsWith(year)` logic with a strict date range filter.
- Implement a mapping of Fiscal Years to actual filing date ranges.
- Use ChromaDB's `$gte` and `$lte` operators for the `filing_date` metadata field.

### 1.2 Multi-Entity Retrieval (Cross-Company Analysis)
**Problem:** The system currently isolates searches to a single company, making it impossible to perform comparative analysis (e.g., "Apple vs Microsoft").
**Proposed Solution:**
- **Query Decomposition:** Update `src/search.py` to detect multiple company names in a single query.
- **Parallel Retrieval:** execute independent search pipelines for each detected entity.
- **Merged Synthesis:** Combine results from all entities before passing them to the LLM for comparative synthesis.

### 1.3 Hybrid Search (BM25 + Vector)
**Problem:** Pure vector search can miss specific financial terms or alphanumeric IDs that aren't well-represented in the embedding space.
**Proposed Solution:**
- Integrate **BM25 (Best Matching 25)** for keyword-based retrieval.
- Implement **Reciprocal Rank Fusion (RRF)** to combine the rankings from the semantic vector search and the keyword search.

---

## 2. Data Quality & Ingestion

### 2.1 Structure-Aware Table Parsing
**Problem:** Tables are currently ingested as flattened text, losing the relational structure between headers and values (e.g., R&D expenses across multiple years).
**Proposed Solution:**
- Replace raw HTML text extraction with a **Markdown Table Parser**.
- Use libraries like `beautifulsoup4` to identify `<table>` tags and convert them to clean Markdown.
- **Table Summarization:** Use an LLM during ingestion to generate a natural language summary of each table to improve searchability.

### 2.2 Text Cleaning & Encoding
**Problem:** Artifacts like `Company�s` indicate encoding issues during the ingestion of `.htm` files.
**Proposed Solution:**
- Enforce `UTF-8` encoding explicitly across the pipeline.
- Implement a cleaning pass to remove non-printable characters and normalize SEC-specific symbols.

---

## 3. Agentic Workflow Enhancements

### 3.1 Dynamic Query Expansion
**Problem:** Users may use terms that don't match the formal language of SEC filings (e.g., "money made" vs "net sales").
**Proposed Solution:**
- Expand the `expand_query` function to use a financial-specific taxonomy.
- Implement "Hypothetical Document Embeddings" (HyDE) to generate a fake answer first and search based on that answer's embedding.

### 3.2 Citation Auditing & Hallucination Guardrails
**Problem:** Without a strict audit, the LLM may synthesize plausible-sounding but incorrect financial figures.
**Proposed Solution:**
- Strengthen the `verify_citations` logic to require an exact string match for numbers.
- Implement a "N-way" verification where multiple LLM calls verify the fact before it is presented to the user.

---

## 4. Performance & Infrastructure

### 4.1 Embedding Model Upgrade
**Problem:** `bge-small-en-v1.5` is efficient but may lack the nuance required for complex financial terminology.
**Proposed Solution:**
- Evaluate and migrate to `bge-large-en-v1.5` or a domain-specific model like `FinBERT`.

### 4.2 Caching Layer
**Problem:** Repeated queries for the same fiscal year/company are computationally expensive (loading models and querying the DB).
**Proposed Solution:**
- Implement a Redis or file-based cache for common query-filter combinations.
