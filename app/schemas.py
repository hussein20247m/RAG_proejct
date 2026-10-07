
"""Pydantic models for API request/response schemas."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PDFUploadResponse(BaseModel):
    """Response model for PDF upload."""
    pdf_id: str
    name: str
    collection_name: str
    doc_count: int
    page_count: int
    upload_timestamp: datetime


class PDFListItem(BaseModel):
    """Model for PDF in list response."""
    pdf_id: str
    name: str
    collection_name: str
    upload_timestamp: datetime
    doc_count: int
    page_count: int
    is_sample: bool


class QueryRequest(BaseModel):
    """Request model for RAG query."""
    question: str
    model: str = "qwen2.5:0.5b"
    pdf_ids: Optional[List[str]] = None
    session_id: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=12)
    min_score: Optional[float] = None


class SourceInfo(BaseModel):
    """Source information for retrieved documents."""
    pdf_name: str
    pdf_id: str
    chunk_index: int
    # Additive optional fields (backward compatible; legacy names kept)
    page: Optional[int] = None
    score: Optional[float] = None
    source_path: Optional[str] = None
    file_type: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for RAG query."""
    answer: str
    sources: List[SourceInfo]
    metadata: Dict[str, Any]
    session_id: str
    message_id: int


class ModelInfo(BaseModel):
    """Ollama model information."""
    name: str
    size: int
    modified_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    chromadb_collections: int
    total_pdfs: int
    indexed_documents: int = 0  # additive optional field


class ScanReport(BaseModel):
    """Result of one incremental scan over the drop zone."""
    new: int = 0
    modified: int = 0
    touched: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped_unsupported: int = 0
    skipped_too_large: int = 0
    failed: List[Dict[str, str]] = Field(default_factory=list)
    duration_s: float = 0.0
    skipped: Optional[str] = None  # e.g. "ollama_unavailable"


class IndexStatusResponse(BaseModel):
    """Aggregated index state from the manifest."""
    total_files: int
    indexed: int
    errors: int
    total_chunks: int
    embedding_model: Optional[str] = None
    last_scan_at: Optional[datetime] = None
    ollama_reachable: bool = False


class DocumentListItem(BaseModel):
    """One manifest row (the drop-zone view)."""
    file_id: str
    rel_path: str
    file_type: str
    status: str
    size: int
    mtime: float
    chunk_count: int
    page_count: int
    error: Optional[str] = None
    indexed_at: datetime


class SessionInfo(BaseModel):
    """Chat session summary."""
    session_id: str
    created_at: datetime
    last_active: datetime


class MessageOut(BaseModel):
    """Persisted chat message including its stored source attribution."""
    message_id: int
    session_id: str
    role: str
    content: str
    sources: Optional[List[SourceInfo]] = None
    timestamp: datetime
