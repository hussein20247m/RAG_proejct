
"""Single ingestion pipeline: recursive discovery -> incremental manifest scan -> index.

Drop-zone files (top-level ``data/``, walked recursively) and API uploads both
flow through this service, so there is exactly one indexing path.

Incremental rules (Phase 6):
- unchanged files (same size + mtime, status ``indexed``) are never re-embedded;
- modified files replace their old chunks (deterministic ids, add-then-prune);
- deleted files have their vectors and manifest rows removed;
- a failed parse/embed marks the row ``error`` and leaves the old vectors alone;
  the next scan re-processes rows whose status is not ``indexed``.
"""
import hashlib
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ollama

from ..config import settings
from ..database import IndexedFile, IndexMeta, PDFMetadata
from ..schemas import ScanReport
from ...core.document import DocumentProcessor
from ...core.embeddings import VectorStore

logger = logging.getLogger(__name__)

_HASH_BUFFER = 1024 * 1024  # 1 MB streaming buffer for sha256


class IngestionError(Exception):
    """Base class for per-file ingestion failures."""


class OllamaUnavailableError(IngestionError):
    """The Ollama server cannot be reached (ingest endpoints return 503)."""


class FileTooLargeError(IngestionError):
    """Uploaded file exceeds MAX_FILE_SIZE_MB (returns 400)."""


class ScanAlreadyRunning(RuntimeError):
    """A scan is already holding the single-flight lock (returns 409)."""


def ollama_status() -> Tuple[bool, str]:
    """(reachable, error message) for the configured Ollama server."""
    try:
        ollama.list()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - any failure means "unreachable"
        return False, f"Ollama is not reachable at {settings.OLLAMA_HOST}: {exc}"


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (1 MB buffer, no whole-file reads)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BUFFER), b""):
            digest.update(block)
    return digest.hexdigest()


def file_id_for(rel_path: str) -> str:
    """Stable identity of a path (never Python's process-randomized hash())."""
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]


class IngestionService:
    """Walks the drop zone and indexes new/changed/removed files incrementally."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        doc_processor: Optional[DocumentProcessor] = None,
        session_factory=None,
    ):
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        self.doc_processor = doc_processor if doc_processor is not None else DocumentProcessor()
        self._session_factory = session_factory
        self._lock = threading.Lock()

    @property
    def session_factory(self):
        """Resolvable at call time so tests can swap the session factory."""
        from .. import database

        return self._session_factory if self._session_factory is not None else database.SessionLocal

    # ------------------------------------------------------------- discovery

    def discover(self) -> Tuple[Dict[str, Tuple[int, float]], int, int]:
        """Recursively walk DATA_DIR.

        Returns ``{rel_path: (size, mtime)}`` plus counts of unsupported and
        oversized files (both recorded, never errors).
        """
        root = settings.DATA_DIR
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        allowed = {ext.lower() for ext in settings.INGEST_EXTENSIONS}
        found: Dict[str, Tuple[int, float]] = {}
        unsupported = 0
        too_large = 0
        if not root.exists():
            return found, unsupported, too_large

        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") for part in relative.parts):
                continue  # hidden files/dirs (.git, .DS_Store, temp files)
            if path.suffix.lower() not in allowed:
                unsupported += 1
                continue
            try:
                stat = path.stat()
            except OSError:
                continue  # vanished mid-walk; the next scan sees it (or doesn't)
            if stat.st_size > max_bytes:
                too_large += 1
                continue
            found[relative.as_posix()] = (stat.st_size, stat.st_mtime)
        return found, unsupported, too_large

    # ----------------------------------------------------------------- scan

    def scan(self) -> ScanReport:
        """Run one incremental scan. Raises ScanAlreadyRunning if one is live."""
        if not self._lock.acquire(blocking=False):
            raise ScanAlreadyRunning("A scan is already running")
        try:
            return self._scan_locked()
        finally:
            self._lock.release()

    def _scan_locked(self) -> ScanReport:
        started = time.perf_counter()
        report = ScanReport()
        root = settings.DATA_DIR

        discovered, unsupported, too_large = self.discover()
        report.skipped_unsupported = unsupported
        report.skipped_too_large = too_large

        # ---- Phase A: classify against the manifest ------------------------
        new: List[str] = []
        pending: Dict[str, Optional[str]] = {}
        removed: List[str] = []
        touched = 0
        unchanged = 0
        model_changed = False

        with self.session_factory() as db:
            rows = {row.rel_path: row for row in db.query(IndexedFile).all()}
            stored_model = db.get(IndexMeta, "embedding_model")
            model_changed = (
                stored_model is not None
                and stored_model.value != settings.OLLAMA_EMBEDDING_MODEL
            )

            if model_changed:
                # Vectors are dimension-bound to the model: full rebuild.
                logger.warning(
                    "Embedding model changed (%s -> %s); performing a full rebuild",
                    stored_model.value,
                    settings.OLLAMA_EMBEDDING_MODEL,
                )
                removed = list(rows)
                new = list(discovered)
            else:
                for rel in sorted(set(discovered) - set(rows)):
                    new.append(rel)
                for rel in sorted(set(rows) - set(discovered)):
                    removed.append(rel)
                for rel in sorted(set(discovered) & set(rows)):
                    size, mtime = discovered[rel]
                    row = rows[rel]
                    if size == row.size and mtime == row.mtime and row.status == "indexed":
                        unchanged += 1  # fast path: stat() only, no hashing
                        continue
                    if size == row.size and mtime == row.mtime:
                        pending[rel] = None  # stats match but last attempt failed
                        continue
                    try:
                        content_hash = sha256_file(root / rel)
                    except OSError as exc:
                        report.failed.append(
                            {"path": rel, "error": f"unreadable: {str(exc)[:400]}"}
                        )
                        continue
                    if content_hash == row.content_hash and row.status == "indexed":
                        row.size = size
                        row.mtime = mtime
                        touched += 1  # touched: bytes identical, stats refreshed
                    else:
                        pending[rel] = content_hash
            db.commit()

        report.unchanged = unchanged
        report.touched = touched

        # ---- Phase B: Ollama reachability pre-check ------------------------
        work_rels: List[str] = list(new) + sorted(pending)
        if work_rels:
            reachable, reason = ollama_status()
            if not reachable:
                # Never crash a scan because the LLM server is down: report
                # every pending file as skipped/failed and keep the old index.
                report.skipped = "ollama_unavailable"
                for rel in work_rels:
                    report.failed.append(
                        {"path": rel, "error": f"ollama_unavailable: {reason}"}
                    )
                logger.warning("Scan skipped: %s", reason)
                new, pending, work_rels = [], [], []
                if model_changed:
                    # The rebuild was cancelled: keep the old index (and rows)
                    # fully intact until embedding is possible again.
                    removed = []

        # ---- Phase C: embedding-model rebuild (only when work is possible) --
        if model_changed and work_rels:
            self.vector_store.delete_collection()
            with self.session_factory() as db:
                db.query(IndexedFile).delete()
                db.commit()
            report.removed = len(removed)
            removed = []  # rows already cleared with the vectors

        # ---- Phase D: removals ---------------------------------------------
        for rel in removed:
            try:
                self.drop_file(rel)
                report.removed += 1
            except Exception as exc:  # noqa: BLE001 - one bad row must not abort
                logger.exception("Failed to drop %s: %s", rel, exc)
                report.failed.append({"path": rel, "error": str(exc)[:500]})

        # ---- Phase E: index new/changed files (per-file error isolation) ----
        total = len(work_rels)
        for position, rel in enumerate(new, start=1):
            ok, error = self._index_with_hash(root / rel, rel, mode="new")
            if ok:
                report.new += 1
                logger.info("indexed [%d/%d] %s (new)", position, total, rel)
            else:
                report.failed.append({"path": rel, "error": error})
        for position, rel in enumerate(sorted(pending), start=1):
            ok, error = self._index_with_hash(root / rel, rel, content_hash=pending[rel], mode="replace")
            if ok:
                report.modified += 1
                logger.info("indexed [%d/%d] %s (modified)", position, total, rel)
            else:
                report.failed.append({"path": rel, "error": error})

        # ---- Phase F: bookkeeping -------------------------------------------
        with self.session_factory() as db:
            if report.skipped is None:
                # Keep the stored model stale when embedding was skipped so the
                # rebuild is retried on the next reachable scan.
                self._set_meta(db, "embedding_model", settings.OLLAMA_EMBEDDING_MODEL)
            self._set_meta(db, "last_scan_at", datetime.now().isoformat())
            db.commit()

        report.duration_s = round(time.perf_counter() - started, 3)
        logger.info(
            "scan finished: new=%d modified=%d touched=%d unchanged=%d removed=%d "
            "failed=%d skipped=%s (%.2fs)",
            report.new,
            report.modified,
            report.touched,
            report.unchanged,
            report.removed,
            len(report.failed),
            report.skipped,
            report.duration_s,
        )
        return report

    @staticmethod
    def _set_meta(db, key: str, value: str) -> None:
        row = db.get(IndexMeta, key)
        if row is None:
            db.add(IndexMeta(key=key, value=value))
        else:
            row.value = value

    def _index_with_hash(
        self,
        path: Path,
        rel_path: str,
        content_hash: Optional[str] = None,
        mode: str = "new",
    ) -> Tuple[bool, str]:
        try:
            return self.index_file(path, rel_path, content_hash=content_hash, mode=mode)
        except Exception as exc:  # noqa: BLE001 - defensive; index_file shouldn't raise
            logger.exception("Unexpected indexing failure for %s", rel_path)
            return False, str(exc)[:500]

    # ------------------------------------------------------------ index_file

    def index_file(
        self,
        path: Path,
        rel_path: str,
        content_hash: Optional[str] = None,
        mode: str = "new",
    ) -> Tuple[bool, str]:
        """Parse -> clean/split -> embed -> replace vectors -> upsert manifest row.

        External calls (parse, embed) happen outside any DB transaction; the
        manifest row flips to ``indexed`` only after the vectors are in place.
        Returns ``(ok, error_message)``; failures never abort a scan.
        """
        file_id = file_id_for(rel_path)
        file_type = Path(rel_path).suffix.lower().lstrip(".")
        stat = None
        try:
            stat = path.stat()
            if content_hash is None:
                content_hash = sha256_file(path)

            documents = self.doc_processor.load(path)
            chunks = self.doc_processor.split_documents(documents)
            self.decorate_chunks(chunks, rel_path, file_id)
            page_count = self._page_count(documents)

            # Idempotent replace: deterministic ids upsert first, then any stale
            # chunk from a previous (possibly interrupted) run is pruned. A parse
            # or embed failure happens before this point, so a working index is
            # never destroyed by a failed replace.
            new_ids = self.vector_store.add_chunks(chunks, file_id) if chunks else []
            self.vector_store.prune_file(file_id, new_ids)

            with self.session_factory() as db:
                row = db.get(IndexedFile, file_id)
                if row is None:
                    row = IndexedFile(file_id=file_id, rel_path=rel_path)
                row.content_hash = content_hash
                row.size = stat.st_size
                row.mtime = stat.st_mtime
                row.file_type = file_type
                row.status = "indexed"
                row.error = None
                row.chunk_count = len(chunks)
                row.page_count = page_count
                row.indexed_at = datetime.now()
                db.merge(row)
                db.commit()
            logger.debug("indexed %s -> %d chunks (%s)", rel_path, len(chunks), mode)
            return True, ""
        except Exception as exc:  # noqa: BLE001 - per-file isolation by design
            error = str(exc)[:500]
            logger.warning("Failed to index %s: %s", rel_path, error)
            try:
                with self.session_factory() as db:
                    row = db.get(IndexedFile, file_id)
                    if row is None:
                        db.add(
                            IndexedFile(
                                file_id=file_id,
                                rel_path=rel_path,
                                content_hash=content_hash or "",
                                size=stat.st_size if stat else 0,
                                mtime=stat.st_mtime if stat else 0.0,
                                file_type=file_type,
                                status="error",
                                error=error,
                                chunk_count=0,
                                page_count=0,
                                indexed_at=datetime.now(),
                            )
                        )
                    else:
                        row.status = "error"
                        row.error = error
                    db.commit()
            except Exception:  # noqa: BLE001 - never mask the original failure
                logger.exception("Could not record failure for %s", rel_path)
            return False, error

    @staticmethod
    def decorate_chunks(chunks: Sequence, rel_path: str, file_id: str) -> None:
        """Write the Phase-3 metadata schema on every chunk (in place)."""
        filename = Path(rel_path).name
        file_type = Path(rel_path).suffix.lower().lstrip(".")
        for index, chunk in enumerate(chunks):
            meta = chunk.metadata
            meta["file_id"] = file_id
            meta["filename"] = filename
            meta["source_path"] = rel_path
            meta["file_type"] = file_type
            meta["chunk_index"] = index
            # Chroma metadata cannot hold None: keep page/row only when present.
            for key in [k for k, v in meta.items() if v is None]:
                del meta[key]

    @staticmethod
    def _page_count(documents: Sequence) -> int:
        """max(page)+1 for paged formats, 1 for documents without page info."""
        pages = [
            doc.metadata.get("page")
            for doc in documents
            if isinstance(doc.metadata.get("page"), int)
        ]
        if pages:
            return max(pages) + 1
        return 1 if documents else 0

    # -------------------------------------------------------------- targeting

    def index_path(self, path: Path) -> Dict[str, object]:
        """Index a single file (upload hook). Takes the single-flight lock."""
        path = Path(path)
        try:
            rel_path = path.relative_to(settings.DATA_DIR).as_posix()
        except ValueError as exc:
            raise IngestionError(f"{path} is outside the drop zone {settings.DATA_DIR}") from exc

        with self._lock:  # serialize with scans (blocking is correct here)
            ok, error = self.index_file(path, rel_path, mode="upload")
            if not ok:
                raise IngestionError(error or "indexing failed")
            with self.session_factory() as db:
                row = db.get(IndexedFile, file_id_for(rel_path))
                if row is None:  # pragma: no cover - defensive
                    raise IngestionError("indexed row missing after indexing")
                return {
                    "file_id": row.file_id,
                    "rel_path": row.rel_path,
                    "chunk_count": row.chunk_count,
                    "page_count": row.page_count,
                    "status": row.status,
                }

    # ------------------------------------------------------------- deletion

    def drop_file(self, rel_path: str) -> None:
        """Remove a file that vanished from disk: vectors + manifest + legacy row."""
        with self.session_factory() as db:
            row = db.query(IndexedFile).filter(IndexedFile.rel_path == rel_path).first()
            if row is None:
                return
            file_id = row.file_id
            abs_path = settings.DATA_DIR / rel_path
            self.vector_store.delete_file(file_id)
            legacy = (
                db.query(PDFMetadata)
                .filter(
                    (PDFMetadata.pdf_id == file_id)
                    | (PDFMetadata.file_path == str(abs_path))
                )
                .first()
            )
            if legacy is not None:
                db.delete(legacy)
            db.delete(row)
            db.commit()
        logger.info("dropped removed file %s", rel_path)

    def delete_entry(self, file_id: str, remove_disk: bool = True, db=None) -> bool:
        """Unified delete: vectors + manifest row + legacy pdfs row + file on disk.

        The disk removal is the default because a file left behind would be
        re-added by the next scan. Returns False when the id is unknown.
        """
        owns_db = db is None
        if owns_db:
            db = self.session_factory()
        try:
            row = db.get(IndexedFile, file_id)
            legacy = db.query(PDFMetadata).filter(PDFMetadata.pdf_id == file_id).first()
            if row is None and legacy is None:
                return False

            if row is not None:
                path: Optional[Path] = settings.DATA_DIR / row.rel_path
            elif legacy is not None and legacy.file_path:
                path = Path(legacy.file_path)
            else:
                path = None

            with self._lock:  # never race a running scan
                self.vector_store.delete_file(file_id)
                if row is not None:
                    db.delete(row)
                if legacy is not None:
                    db.delete(legacy)
                db.commit()

            if (
                remove_disk
                and path is not None
                and path.exists()
                and path.is_relative_to(settings.DATA_DIR)
            ):
                path.unlink()
                logger.info("removed %s from disk", path)
            return True
        finally:
            if owns_db:
                db.close()

    # ---------------------------------------------------------------- status

    def list_manifest(self, limit: int = 100, offset: int = 0, db=None) -> List[dict]:
        """Manifest rows as plain dicts (safe to read after the session closes)."""
        owns_db = db is None
        if owns_db:
            db = self.session_factory()
        try:
            rows = (
                db.query(IndexedFile)
                .order_by(IndexedFile.rel_path)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [
                {
                    "file_id": r.file_id,
                    "rel_path": r.rel_path,
                    "file_type": r.file_type,
                    "status": r.status,
                    "size": r.size,
                    "mtime": r.mtime,
                    "chunk_count": r.chunk_count,
                    "page_count": r.page_count,
                    "error": r.error,
                    "indexed_at": r.indexed_at,
                }
                for r in rows
            ]
        finally:
            if owns_db:
                db.close()
