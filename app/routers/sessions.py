"""Chat session endpoints: list, read messages, delete."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..dependencies import get_db, get_rag_service
from ..schemas import MessageOut, SessionInfo
from ..services.reg_services import RAGService

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.get("", response_model=List[SessionInfo])
def list_sessions(
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """List chat sessions (most recently active first)."""
    return rag_service.list_sessions(db)


@router.get("/{session_id}/messages", response_model=List[MessageOut])
def get_messages(
    session_id: str,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Stored messages of a session, including their sources JSON."""
    messages = rag_service.get_messages(session_id, db)
    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return messages


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Delete a session and all of its messages."""
    if not rag_service.delete_session(session_id, db):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}
