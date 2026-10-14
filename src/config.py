import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_backup"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
DOCUMENTS_JSONL = PROCESSED_DIR / "documents.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"

CHROMA_DIR = PROJECT_ROOT / "data/vector_db"
CACHE_DIR = PROJECT_ROOT / "data/cache/search"
EMBEDDING_CACHE_FILE = PROJECT_ROOT / "data/cache/embedding_cache.json"
COLLECTION_NAME = "sec_filings"

# ============================================================
# MODELS
# ============================================================
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Groq-hosted model for synthesis and query refinement.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")

# ============================================================
# HYPERPARAMETERS
# ============================================================
DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_HITS = 100
CONFIDENCE_THRESHOLD = 0.1  # Distance below which we use the "Fast Path"

# ============================================================
# COMPANY MAPPING
# ============================================================
COMPANY_MAP = {
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
