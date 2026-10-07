
"""FastAPI main application."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import get_ingestion_service
from .routers import health, indexing, models, pdfs, query, sessions

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging, ensure data dirs, schedule the startup scan.

    Startup only *schedules* the incremental scan (in the default executor):
    the server accepts requests immediately and indexing proceeds in the
    background. A warm start is a stat() comparison over the drop zone only.
    """
    # 1. Logging first, before anything else runs.
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # 2. Ensure the drop zone and internal storage exist.
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Fire-and-forget incremental startup scan (gated by INDEX_ON_STARTUP;
    #    the scan itself degrades gracefully when Ollama is unreachable).
    if settings.INDEX_ON_STARTUP:
        loop = asyncio.get_running_loop()

        def _startup_scan() -> None:
            try:
                report = get_ingestion_service().scan()
                if report.skipped:
                    logger.warning("Startup scan skipped: %s", report.skipped)
            except Exception:  # noqa: BLE001 - a scan must never kill startup
                logger.exception("Startup scan failed")

        loop.run_in_executor(None, _startup_scan)
    else:
        logger.info("Startup scan disabled (INDEX_ON_STARTUP=false)")

    yield


# Initialize FastAPI
app = FastAPI(
    title="Ollama PDF RAG API",
    description="REST API for PDF-based RAG with Ollama",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(models.router)
app.include_router(pdfs.router)
app.include_router(query.router)
app.include_router(indexing.router)
app.include_router(sessions.router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Ollama PDF RAG API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
