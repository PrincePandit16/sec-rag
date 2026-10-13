from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
import chromadb
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

from src import config
from src.search import (
    search,
    synthesize_answer,
    run_agentic_search,
    LLMClient,
)
from src.ingest import ingest
from src.chunk_documents import process as chunk_docs
from src.embed_chunks import process as embed_chunks
from src.build_vector_db import build_vector_database
from src.validate_embeddings import validate_embeddings

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
        self.collection: Optional[chromadb.Collection] = None
        self.llm_client: Optional[LLMClient] = None

state = SystemState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing system state...")

    # Load Models
    logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
    state.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    # Log the dimension to be 100% sure
    test_emb = state.embedding_model.encode("test")
    logger.info(f"Embedding model loaded. Dimension: {len(test_emb)}")

    state.reranker = CrossEncoder(config.RERANKER_MODEL, device="cpu")

    # Load DB
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    state.collection = client.get_collection(config.COLLECTION_NAME)

    # Async LLM Client
    api_key = os.environ.get("GROQ_API_KEY")
    state.llm_client = LLMClient(api_key=api_key)

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
    companies: Optional[str] = None
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
            "llm_client": state.llm_client is not None,
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if (
        state.collection is None
        or state.embedding_model is None
        or state.reranker is None
    ):
        raise HTTPException(status_code=503, detail="System not initialized")

    if req.agentic:
        # Run async agentic loop
        try:
            results, answer, audit = await run_agentic_search(
                collection=state.collection,
                model=state.embedding_model,
                reranker=state.reranker,
                client=state.llm_client,
                query=req.query,
                top_k=req.top_k,
                companies=req.companies,
                year=req.year,
                form=req.form,
            )
        except Exception as e:
            logger.error(f"Agentic search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search engine error: {str(e)}")

        return QueryResponse(
            answer=answer,
            sources=results,
            audit_report=audit,
            attempts=3
        )
    else:
        # Simple search (converted to async in search.py)
        try:
            results = await search(
                collection=state.collection,
                model=state.embedding_model,
                reranker=state.reranker,
                client=state.llm_client,
                query=req.query,
                top_k=req.top_k,
                companies=req.companies,
                year=req.year,
                form=req.form,
            )
            answer = await synthesize_answer(state.llm_client, req.query, results)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search engine error: {str(e)}")

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
            model_name=config.EMBEDDING_MODEL,
            batch_size=32,
            device="auto"
        )

        if not validate_embeddings():
            raise RuntimeError("Embedding validation failed; database was not rebuilt.")

        # 4. Embeddings -> ChromaDB
        build_vector_database()

        logger.info("Ingestion pipeline completed successfully.")
    except Exception:
        logger.exception("Pipeline failed")
        raise

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_full_pipeline)
    return {"message": "Ingestion pipeline triggered in background."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
