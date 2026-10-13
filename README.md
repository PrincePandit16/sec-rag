# SEC Filings RAG System

A Retrieval-Augmented Generation (RAG) system designed to process unstructured SEC filings (10-K, 10-Q) and convert them into a searchable vector database for professional financial analysis.

## 🚀 Features

- **End-to-End Pipeline**: Automated workflow from raw SEC HTML downloads to vector ingestion.
- **Hybrid Search**: Combines semantic vector search (via `BAAI/bge-small-en-v1.5`) with specialized financial boosting.
- **Financial Intelligence**: 
  - **Keyword Boosting**: Prioritizes chunks containing financial terms like "net sales" or "net income".
  - **Golden Pattern Matching**: High-precision detection of key financial metrics (e.g., "Total net sales $X billion").
  - **Table Awareness**: Boosts results that look like financial tables.
  - **Contextual Penalty**: Intelligently de-prioritizes expense sections when the user is specifically querying for revenue.
- **AI Synthesis**: Integrates with Claude 3.5 Sonnet to synthesize retrieved chunks into natural language answers with citations.

## 🛠️ Architecture

The system follows a standard RAG pipeline:
1. **Download**: Fetches raw `.htm` filings from the SEC $\rightarrow$ `data/raw/`
2. **Chunking**: Splits documents into manageable segments for better retrieval $\rightarrow$ `data/processed/`
3. **Embedding**: Generates high-dimensional vectors using Sentence-Transformers.
4. **Ingestion**: Populates a persistent ChromaDB vector store $\rightarrow$ `data/vector_db/`
5. **Search**: Performs hybrid retrieval and reranking based on financial relevance.

## 📦 Installation

### Prerequisites
- Python 3.10+
- `uv` (recommended) or `pip`

### Setup
```bash
# Clone the repository
git clone https://github.com/PrincePandit16/sec-rag.git
cd sec-rag

# Install dependencies
uv sync
# For the optional RAG evaluation tools:
uv sync --extra evaluation
```

### Environment Variables
Set a Groq API key to enable AI synthesis and agentic search. An Anthropic key is
optional and is used only to summarize tables during filing ingestion.
```bash
export GROQ_API_KEY='your-groq-api-key-here'
export ANTHROPIC_API_KEY='your-anthropic-api-key-here'
# Optional; defaults to a currently supported Groq model.
export LLM_MODEL='openai/gpt-oss-120b'
```

## 💻 Usage

### Running the Pipeline
To build the database from scratch:
```bash
uv run python -m src.download_filings
uv run python -m src.ingest
uv run python -m src.chunk_documents
uv run python -m src.embed_chunks
uv run python -m src.validate_embeddings
uv run python -m src.build_vector_db
```

The pipeline order matters: `ingest` creates `documents.jsonl`, chunking reads that
file, and the vector database is built only after embedding validation succeeds.
Rebuilding replaces the `sec_filings` ChromaDB collection; other collections are
left untouched.

Start the API with:
```bash
uv run python -m src.api
```

### Searching the Database
You can query the system using the `search.py` script:

**Basic Search:**
```bash
uv run python -m src.search "What is the revenue of Apple?"
```

**Filtered Search:**
```bash
uv run python -m src.search "What are the risk factors?" --company apple --year 2024 --form 10-K
```

**Adjusting Results:**
```bash
uv run python -m src.search "Net income" --top-k 20
```

## 📂 Project Structure

```text
sec-rag/
├── data/
│   ├── raw/            # Original SEC .htm filings
│   ├── processed/      # Intermediate JSONL chunks & embeddings
│   └── vector_db/      # ChromaDB persistent storage
├── src/
│   ├── search.py       # Hybrid search & reranking logic
│   ├── ingest.py       # Database population
│   ├── embed_chunks.py  # Vector generation
│   └── chunk_documents.py # Text splitting logic
├── pyproject.toml      # Project dependencies
└── README.md           # Documentation
```
