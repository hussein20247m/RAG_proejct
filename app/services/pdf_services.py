
"""Document upload service.

Uploads are written into the drop zone (``DATA_DIR/uploads``) and then indexed
by the *same* incremental pipeline that handles dragged-in files — one
pipeline, no special case. The legacy ``pdfs`` table stays populated so the
frozen upload/list/delete API keeps working.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ..config import settings
from ..database import IndexedFile, PDFMetadata
from .ingestion_service import (
    FileTooLargeError,
    IngestionService,
    IngestionError,
    OllamaUnavailableError,
    ollama_status,
)
from ...core.embeddings import VectorStore

logger = logging.getLogger(__name__)


class PDFService:
    """Service for upload/list/delete operations on the legacy PDF API."""

    def __init__(self, ingestion: Optional[IngestionService] = None):
        self.ingestion = ingestion if ingestion is not None else IngestionService()
        self.storage_dir = Path(settings.UPLOAD_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def upload_and_process(self, file: UploadFile, db: Session) -> PDFMetadata:
        """Save an upload into the drop zone and index it immediately.

        Order matters (defect #21): validate size and Ollama reachability
        *before* anything touches disk, and remove the file again if indexing
        fails, so a failed upload never leaves an orphan behind.

        Raises:
            FileTooLargeError: over MAX_FILE_SIZE_MB (router maps to 400).
            OllamaUnavailableError: server down (router maps to 503).
            IngestionError: parse/embed failure (router maps to 422).
        """
        filename = Path(file.filename or "").name
        content = await file.read()
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(content) > max_bytes:
            raise FileTooLargeError(
                f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB limit"
            )

        reachable, reason = ollama_status()
        if not reachable:
            raise OllamaUnavailableError(reason)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        dest = self.storage_dir / filename
        async with aiofiles.open(dest, "wb") as handle:
            await handle.write(content)

        try:
            # Heavy parse+embed work runs in the threadpool (never the event loop)
            # and takes the ingestion single-flight lock.
            snapshot = await run_in_threadpool(self.ingestion.index_path, dest)
        except IngestionError:
            # Remove the orphan file; the manifest keeps an `error` row that the
            # next scan retries (or drops once the file is gone).
            try:
                dest.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - defensive
                logger.warning("Could not remove failed upload %s", dest)
            raise

        now = datetime.now()
        pdf_id = str(snapshot["file_id"])
        row = db.get(PDFMetadata, pdf_id)
        if row is None:
            row = PDFMetadata(pdf_id=pdf_id, is_sample=False)
        row.name = filename
        row.collection_name = VectorStore.COLLECTION_NAME  # constant: "documents"
        row.upload_timestamp = now
        row.doc_count = int(snapshot["chunk_count"])
        row.page_count = int(snapshot["page_count"])
        row.file_path = str(dest)
        row = db.merge(row)
        db.commit()
        db.refresh(row)
        logger.info(
            "uploaded %s -> file_id=%s (%d chunks)",
            filename,
            pdf_id,
            row.doc_count,
        )
        return row

    def list_pdfs(
        self, db: Session, limit: int = 100, offset: int = 0
    ) -> List[PDFMetadata]:
        """List uploaded PDFs with pagination (newest first)."""
        return (
            db.query(PDFMetadata)
            .order_by(PDFMetadata.upload_timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_pdf(self, pdf_id: str, db: Session) -> Optional[PDFMetadata]:
        """Get single PDF metadata by id."""
        return db.query(PDFMetadata).filter(PDFMetadata.pdf_id == pdf_id).first()

    def delete_pdf(self, pdf_id: str, db: Session) -> bool:
        """Delete an uploaded document via the unified delete.

        Removes vectors (by ``file_id``), the manifest row, the legacy ``pdfs``
        row and the file on disk (otherwise the next scan would re-add it).
        """
        row = db.get(IndexedFile, pdf_id)
        legacy = db.query(PDFMetadata).filter(PDFMetadata.pdf_id == pdf_id).first()
        if row is None and legacy is None:
            return False
        return self.ingestion.delete_entry(pdf_id, remove_disk=True, db=db)
