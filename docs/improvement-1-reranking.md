# Improvement 1: Two-Stage Retrieval (Cross-Encoder Reranking)

## Overview
Basic vector search (Bi-Encoders) is fast but can lose precision. To reach production grade, we will implement a "Retrieve and Re-rank" architecture.

## Implementation Strategy
1. **Stage 1 (Retrieval)**: Use the current ChromaDB hybrid search to retrieve a larger set of candidates (e.g., `top_k=50` instead of `10`).
2. **Stage 2 (Reranking)**: Pass these 50 candidates through a **Cross-Encoder model** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
3. **Final Selection**: Sort the results based on the Cross-Encoder score and return the top 5 to the LLM.

## Why This Standouts
Cross-Encoders analyze the query and the document simultaneously, capturing deep semantic interactions that simple cosine similarity misses. This drastically reduces "false positive" results.

## Action Items
- [ ] Integrate `sentence-transformers` CrossEncoder class in `src/search.py`.
- [ ] Update `search()` function to implement the two-stage flow.
- [ ] Benchmark accuracy vs. basic vector search.
