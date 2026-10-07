
"""FastAPI dependencies for dependency injection.

Services are process-wide singletons (`@lru_cache`) so the Chroma client /
vector store is created once per process, not once per request.
"""
from functools import lru_cache

from sqlalchemy.orm import Session

from . import database


def get_db():
    """Database session dependency."""
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache(maxsize=1)
def get_vector_store():
    """Shared Chroma-backed vector store (one persistent client per process)."""
    from ..core.embeddings import VectorStore

    return VectorStore()


@lru_cache(maxsize=1)
def get_ingestion_service():
    """Shared incremental ingestion service."""
    from .services.ingestion_service import IngestionService

    return IngestionService(vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_rag_service():
    """Shared RAG service (retrieval + generation + session persistence)."""
    from ..core.llm import LLMManager
    from ..core.reg import RAGPipeline
    from .services.reg_services import RAGService

    pipeline = RAGPipeline(
        vector_db=get_vector_store(),
        llm_manager=LLMManager(),
    )
    return RAGService(pipeline)


@lru_cache(maxsize=1)
def get_pdf_service():
    """Shared PDF/upload service (delegates indexing to the ingestion service)."""
    from .services.pdf_services import PDFService

    return PDFService(ingestion=get_ingestion_service())
