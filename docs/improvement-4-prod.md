# Improvement 4: Production Infrastructure & Evaluation

## Overview
Move the project from a set of scripts to a professional software package with a measurable accuracy score.

## Implementation Strategy
1. **API Layer (FastAPI)**:
   - Wrap the search and synthesis logic into a FastAPI backend.
   - Create endpoints: `/query` (search), `/health` (system status), and `/upload` (trigger ingestion).
2. **User Interface (Streamlit/Next.js)**:
   - Build a dashboard where users can select a company, ask questions, and see the "Source Chunks" alongside the "AI Answer."
3. **Evaluation Framework (RAGAS)**:
   - Use the **RAGAS** (RAG Assessment) framework to generate a "Golden Dataset" of Question-Answer pairs.
   - Measure and report:
     - **Faithfulness**: Does the answer only use the provided context?
     - **Answer Relevance**: Does the answer actually address the user's query?
     - **Context Precision**: Are the retrieved chunks actually relevant?

## Why This Standouts
Providing a "Faithfulness Score" of 90% is a powerful engineering claim. It shows you aren't just "guessing" that the system works, but you have mathematically proven it.

## Action Items
- [ ] Create `src/api.py` using FastAPI.
- [ ] Build a basic Streamlit UI in `ui/app.py`.
- [ ] Integrate RAGAS to create an evaluation report in `docs/eval_report.md`.
