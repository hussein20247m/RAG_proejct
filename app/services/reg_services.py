"""RAG query service: query orchestration + session/message persistence."""
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from .. import database
from ..database import ChatMessage, ChatSession
from ..schemas import (
    MessageOut,
    QueryRequest,
    QueryResponse,
    SessionInfo,
    SourceInfo,
)
from ...core.reg import RAGPipeline

logger = logging.getLogger(__name__)


class RAGService:
    """Runs RAG queries and persists chat sessions / messages.

    Attribution is computed from retrieval metadata (never parsed out of the
    answer text): ``sources[i]`` always corresponds to context block ``[i+1]``.
    """

    def __init__(self, rag_pipeline: RAGPipeline, db_factory=None):
        self.pipeline = rag_pipeline
        self._db_factory = db_factory

    @property
    def db_factory(self):
        """Resolvable at call time so tests can swap the session factory."""
        return (
            self._db_factory
            if self._db_factory is not None
            else database.SessionLocal
        )

    # ----------------------------------------------------------------- query

    def query(self, request: QueryRequest, db: Optional[Session] = None) -> QueryResponse:
        """Answer one question and persist the exchange."""
        owns_db = db is None
        if owns_db:
            db = self.db_factory()
        try:
            result = self.pipeline.answer(
                request.question,
                pdf_ids=request.pdf_ids,
                top_k=request.top_k,
                min_score=request.min_score,
            )
            sources = self._to_sources(result.documents, result.scores)
            session_row, session_new = self._resolve_session(request.session_id, db)

            message = ChatMessage(
                session_id=session_row.session_id,
                role="assistant",
                content=result.text,
                sources=[source.model_dump() for source in sources],
                timestamp=datetime.now(),
            )
            db.add(message)
            session_row.last_active = datetime.now()
            db.commit()
            db.refresh(message)

            metadata = dict(result.metadata)
            metadata["session_new"] = session_new
            return QueryResponse(
                answer=result.text,
                sources=sources,
                metadata=metadata,
                session_id=session_row.session_id,
                message_id=message.message_id,
            )
        finally:
            if owns_db:
                db.close()

    @staticmethod
    def _to_sources(documents, scores) -> List[SourceInfo]:
        """Map retrieved chunks to SourceInfo (deduped, score-sorted order kept)."""
        sources: List[SourceInfo] = []
        seen: set = set()
        for doc, score in zip(documents, scores):
            meta = doc.metadata
            key = (meta.get("file_id"), meta.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                SourceInfo(
                    pdf_name=meta.get("filename") or str(meta.get("source_path", "unknown")),
                    pdf_id=str(meta.get("file_id", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    page=meta.get("page"),
                    score=round(float(score), 4),
                    source_path=meta.get("source_path"),
                    file_type=meta.get("file_type"),
                )
            )
        return sources

    @staticmethod
    def _resolve_session(
        session_id: Optional[str], db: Session
    ) -> Tuple[ChatSession, bool]:
        """Create the session when absent, else refresh ``last_active``."""
        now = datetime.now()
        row = db.get(ChatSession, session_id) if session_id else None
        if row is not None:
            row.last_active = now
            return row, False
        session_id = session_id or uuid.uuid4().hex
        row = ChatSession(session_id=session_id, created_at=now, last_active=now)
        db.add(row)
        db.flush()
        return row, True

    # -------------------------------------------------------------- sessions

    def list_sessions(self, db: Optional[Session] = None) -> List[SessionInfo]:
        owns_db = db is None
        if owns_db:
            db = self.db_factory()
        try:
            rows = (
                db.query(ChatSession)
                .order_by(ChatSession.last_active.desc())
                .all()
            )
            return [
                SessionInfo(
                    session_id=row.session_id,
                    created_at=row.created_at,
                    last_active=row.last_active,
                )
                for row in rows
            ]
        finally:
            if owns_db:
                db.close()

    def get_messages(
        self, session_id: str, db: Optional[Session] = None
    ) -> Optional[List[MessageOut]]:
        """Stored messages (incl. sources JSON) or None when the session is unknown."""
        owns_db = db is None
        if owns_db:
            db = self.db_factory()
        try:
            if db.get(ChatSession, session_id) is None:
                return None
            rows = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.message_id)
                .all()
            )
            return [
                MessageOut(
                    message_id=row.message_id,
                    session_id=row.session_id,
                    role=row.role,
                    content=row.content,
                    sources=row.sources,
                    timestamp=row.timestamp,
                )
                for row in rows
            ]
        finally:
            if owns_db:
                db.close()

    def delete_session(self, session_id: str, db: Optional[Session] = None) -> bool:
        """Delete a session and its messages. False when unknown."""
        owns_db = db is None
        if owns_db:
            db = self.db_factory()
        try:
            row = db.get(ChatSession, session_id)
            if row is None:
                return False
            db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).delete()
            db.delete(row)
            db.commit()
            return True
        finally:
            if owns_db:
                db.close()
