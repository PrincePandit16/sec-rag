from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.search import (
    load_model,
    load_reranker,
    load_collection,
    search,
    synthesize_answer,
    run_agentic_search,
    SentenceTransformer,
    CrossEncoder,
)
import chromadb
from src.ingest import ingest
from src.chunk_documents import process as chunk_docs
from src.embed_chunks import process as embed_chunks

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sec-rag-api")

# ============================================================
# MODELS & STATE
# ============================================================

class SystemState:
    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self.reranker: Optional[CrossEncoder] = None
        self.collection: Optional[chromadb.collection.Collection] = None

state = SystemState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing system state...")
    state.embedding_model = load_model()
    state.reranker = load_reranker()
    state.collection = load_collection()
    logger.info("System ready.")
    yield
    # Shutdown
    logger.info("Shutting down system...")

app = FastAPI(lifespan=lifespan)

# ============================================================
# SCHEMAS
# ============================================================

class QueryRequest(BaseModel):
    query: str
    company: Optional[str] = None
    year: Optional[int] = None
    form: Optional[str] = None
    agentic: bool = False
    top_k: int = 10

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[dict]
    audit_report: Optional[str] = None
    attempts: int = 1

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health():
    if state.collection is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return {
        "status": "healthy",
        "documents_count": state.collection.count(),
        "models_loaded": {
            "embedding": state.embedding_model is not None,
            "reranker": state.reranker is not None,
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if state.collection is None or state.embedding_model is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    if req.agentic:
        # Run agentic loop
        results, answer, audit = run_agentic_search(
            collection=state.collection,
            model=state.embedding_model,
            reranker=state.reranker,
            query=req.query,
            top_k=req.top_k,
            company=req.company,
            year=req.year,
            form=req.form,
        )
        return QueryResponse(
            answer=answer,
            sources=results,
            audit_report=audit,
            attempts=3 # Agentic loop max attempts
        )
    else:
        # Simple search
        results = search(
            collection=state.collection,
            model=state.embedding_model,
            reranker=state.reranker,
            query=req.query,
            top_k=req.top_k,
            company=req.company,
            year=req.year,
            form=req.form,
        )
        answer = synthesize_answer(req.query, results)
        return QueryResponse(
            answer=answer,
            sources=results,
            attempts=1
        )

async def run_full_pipeline():
    """Orchestrates the full ingestion pipeline."""
    try:
        logger.info("Starting full ingestion pipeline...")

        # 1. Raw HTML -> JSONL
        ingest()

        # 2. JSONL -> Chunks
        chunk_docs(
            input_path="data/processed/documents.jsonl",
            output_path="data/processed/chunks.jsonl",
            chunk_size=400,
            overlap=50
        )

        # 3. Chunks -> Embeddings
        embed_chunks(
            input_path=Path("data/processed/chunks.jsonl"),
            output_path=Path("data/processed/embeddings.jsonl"),
            model_name="BAAI/bge-small-en-v1.5",
            batch_size=32,
            device="auto"
        )

        # Note: In a real production system, we would now trigger
        # a ChromaDB collection update from the embeddings.jsonl.

        logger.info("Ingestion pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_full_pipeline)
    return {"message": "Ingestion pipeline triggered in background."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
