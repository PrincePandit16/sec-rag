from typing import TypedDict, List, Optional, Dict, Any

class DocumentMetadata(TypedDict):
    company: str
    filing_date: str
    form: str
    document: str
    source_file: str
    chunk_id: Optional[str]

class SearchResult(TypedDict):
    document: str
    metadata: DocumentMetadata
    distance: Optional[float]
    score: float
    rerank_score: Optional[float]
    matched_keywords: int
