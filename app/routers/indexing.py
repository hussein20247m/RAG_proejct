"""Indexing endpoints: scan, status, drop-zone document listing/deletion."""
import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..dependencies import get_db, get_ingestion_service
from ..schemas import DocumentListItem, IndexStatusResponse, ScanReport
from ..database import IndexMeta, IndexedFile
from ..services.ingestion_service import ScanAlreadyRunning, ollama_status

router = APIRouter(prefix="/api/v1", tags=["indexing"])
logger = logging.getLogger(__name__)


@router.post("/index/scan", response_model=ScanReport)
def trigger_scan(
    ingestion=Depends(get_ingestion_service),
):
    """Run one incremental scan of the drop zone (409 if one is running).

    Returns 503 with the full report in the body when Ollama is unreachable
    and there was work to embed - a graceful skip, never a 500 traceback.
    """
    try:
        report = ingestion.scan()
    except ScanAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if report.skipped == "ollama_unavailable":
        return JSONResponse(
            status_code=503,
            content={
                **report.model_dump(mode="json"),
                "detail": (
                    f"Ollama is not reachable at {settings.OLLAMA_HOST}; "
                    "start it or fix OLLAMA_HOST"
                ),
            },
        )
    return report


@router.get("/index/status", response_model=IndexStatusResponse)
def index_status(db: Session = Depends(get_db)):
    """Aggregated index state from the manifest."""
    total = db.query(IndexedFile).count()
    indexed = db.query(IndexedFile).filter(IndexedFile.status == "indexed").count()
    errors = db.query(IndexedFile).filter(IndexedFile.status == "error").count()
    chunks = (
        db.query(func.coalesce(func.sum(IndexedFile.chunk_count), 0)).scalar() or 0
    )
    model_row = db.get(IndexMeta, "embedding_model")
    scan_row = db.get(IndexMeta, "last_scan_at")
    last_scan_at = None
    if scan_row is not None:
        try:
            last_scan_at = datetime.fromisoformat(scan_row.value)
        except ValueError:  # pragma: no cover - defensive
            last_scan_at = None
    reachable, _ = ollama_status()
    return IndexStatusResponse(
        total_files=total,
        indexed=indexed,
        errors=errors,
        total_chunks=int(chunks),
        embedding_model=model_row.value if model_row else None,
        last_scan_at=last_scan_at,
        ollama_reachable=reachable,
    )


@router.get("/documents", response_model=List[DocumentListItem])
def list_documents(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    ingestion=Depends(get_ingestion_service),
):
    """List manifest rows (the drop-zone view GET /pdfs cannot provide)."""
    rows = ingestion.list_manifest(limit=limit, offset=offset, db=db)
    return [DocumentListItem(**row) for row in rows]


@router.delete("/documents/{file_id}")
def delete_document(
    file_id: str,
    remove_file: bool = Query(True),
    db: Session = Depends(get_db),
    ingestion=Depends(get_ingestion_service),
):
    """Remove a document from the index (and disk, by default).

    ``remove_file=false`` only makes sense with a future tombstone feature:
    the next scan will re-index the file.
    """
    deleted = ingestion.delete_entry(file_id, remove_disk=remove_file, db=db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document removed from index"}
