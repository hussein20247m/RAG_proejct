
"""PDF management endpoints (uploads land in the drop zone: data/uploads)."""
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..dependencies import get_db, get_pdf_service
from ..schemas import PDFListItem, PDFUploadResponse
from ..services.pdf_services import PDFService
from ..services.ingestion_service import (
    FileTooLargeError,
    IngestionError,
    OllamaUnavailableError,
)

router = APIRouter(prefix="/api/v1/pdfs", tags=["pdfs"])


@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    pdf_service: PDFService = Depends(get_pdf_service),
):
    """Upload and index a document (extension allow-list from settings)."""
    suffix = Path(file.filename or "").suffix.lower()
    allowed = {ext.lower() for ext in settings.INGEST_EXTENSIONS}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix or '<none>'}'. "
                f"Allowed: {', '.join(sorted(allowed))}"
            ),
        )

    try:
        pdf_metadata = await pdf_service.upload_and_process(file, db)
    except FileTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PDFUploadResponse(
        pdf_id=pdf_metadata.pdf_id,
        name=pdf_metadata.name,
        collection_name=pdf_metadata.collection_name,
        doc_count=pdf_metadata.doc_count,
        page_count=pdf_metadata.page_count,
        upload_timestamp=pdf_metadata.upload_timestamp,
    )


@router.get("", response_model=List[PDFListItem])
def list_pdfs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    pdf_service: PDFService = Depends(get_pdf_service),
):
    """List uploaded documents (paginated, newest first)."""
    pdfs = pdf_service.list_pdfs(db, limit=limit, offset=offset)
    return [
        PDFListItem(
            pdf_id=pdf.pdf_id,
            name=pdf.name,
            collection_name=pdf.collection_name,
            upload_timestamp=pdf.upload_timestamp,
            doc_count=pdf.doc_count,
            page_count=pdf.page_count,
            is_sample=pdf.is_sample,
        )
        for pdf in pdfs
    ]


@router.delete("/{pdf_id}")
def delete_pdf(
    pdf_id: str,
    db: Session = Depends(get_db),
    pdf_service: PDFService = Depends(get_pdf_service),
):
    """Delete a PDF: vectors + manifest row + pdfs row + file on disk."""
    success = pdf_service.delete_pdf(pdf_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="PDF not found")
    return {"message": "PDF deleted successfully"}
