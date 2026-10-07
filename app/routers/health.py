"""Health check endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import ollama

from ..dependencies import get_db, get_vector_store
from ..schemas import HealthResponse
from ..database import PDFMetadata, IndexedFile
from ...core.embeddings import VectorStore

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health_check(
    db: Session = Depends(get_db),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """Check API health."""

    # Check Ollama connection
    ollama_connected = False
    try:
        ollama.list()
        ollama_connected = True
    except Exception:
        pass

    # Real collection count from the shared Chroma store (was: counting
    # subdirectories of data/vectors, which is always 0 - defect #25)
    collection_count = vector_store.collection_count()

    # Check total PDFs and manifest size
    total_pdfs = db.query(PDFMetadata).count()
    indexed_documents = db.query(IndexedFile).count()

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        ollama_connected=ollama_connected,
        chromadb_collections=collection_count,
        total_pdfs=total_pdfs,
        indexed_documents=indexed_documents,
    )
