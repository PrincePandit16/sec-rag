# Improvement 3: Agentic Workflow & LLM Orchestration

## Overview
Move from a linear "Retrieve $\rightarrow$ Answer" flow to an intelligent agent that can self-correct, expand queries, and verify its own answers.

## Implementation Strategy
1. **Query Expansion (Multi-Query)**:
   - Use the LLM to rewrite a single user query into 3-5 different versions to cover all semantic angles.
   - Retrieve results for all versions and deduplicate them.
2. **Self-Correction Loop**:
   - After retrieval, the LLM performs a "Relevance Check."
   - If the LLM determines the retrieved chunks don't contain the answer, it automatically triggers a new search with a refined query.
3. **Verification & Citations**:
   - Implement a "Citation Auditor" step where the LLM must map every claim in the final answer back to a specific line/chunk in the source text.

## Why This Standouts
This transforms the system from a simple search engine into a "Financial Analyst Agent" that knows when it doesn't have enough information and knows how to go find it.

## Action Items
- [ ] Implement a query expansion module in `src/search.py`.
- [ ] Create a `verify_answer()` loop using a framework like LangGraph or a custom python loop.
- [ ] Enhance `synthesize_answer()` to enforce strict citation rules.
