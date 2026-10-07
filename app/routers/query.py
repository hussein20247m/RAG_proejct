"""Query endpoint: the missing answer path (QueryRequest -> QueryResponse)."""
import logging

import ollama
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..config import settings
from ..dependencies import get_db, get_rag_service
from ..schemas import QueryRequest, QueryResponse
from ..services.reg_services import RAGService

router = APIRouter(prefix="/api/v1", tags=["query"])
logger = logging.getLogger(__name__)


def _available_models() -> list:
    """Chat model names served by Ollama (raises when the server is down)."""
    listing = ollama.list()
    names = []
    for model in getattr(listing, "models", None) or []:
        name = getattr(model, "model", None)
        if not name and hasattr(model, "model_dump"):
            name = model.model_dump().get("model", "")
        if name:
            names.append(name)
    return names


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Answer a question from the indexed documents with source attribution."""
    try:
        available = _available_models()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Ollama is not reachable at {settings.OLLAMA_HOST}; "
                f"start it or fix OLLAMA_HOST ({exc})"
            ),
        )
    if request.model not in available:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{request.model}'. "
                f"Available: {', '.join(available) or 'none'}"
            ),
        )
    return rag_service.query(request, db)
