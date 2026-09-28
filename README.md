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
# OR
pip install -r requirements.txt
```

### Environment Variables
To use the AI synthesis feature, add your Anthropic API key:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## 💻 Usage

### Running the Pipeline
To build the database from scratch:
```bash
python src/download_filings.py
python src/chunk_documents.py
python src/embed_chunks.py
python src/ingest.py
```

### Searching the Database
You can query the system using the `search.py` script:

**Basic Search:**
```bash
python src/search.py "What is the revenue of Apple?"
```

**Filtered Search:**
```bash
python src/search.py "What are the risk factors?" --company apple --year 2024 --form 10-K
```

**Adjusting Results:**
```bash
python src/search.py "Net income" --top-k 20
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
