
"""Vector store (Chroma) wrapper for the shared `documents` collection."""
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from ..app.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings and database operations.

    One persistent Chroma collection (`documents`) holds the whole corpus;
    chunks are addressed by deterministic ids `{file_id}:{chunk_index}` and
    filterable by the `file_id` metadata key.
    """

    COLLECTION_NAME = "documents"

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embeddings=None,
    ):
        self.embedding_model = embedding_model or settings.OLLAMA_EMBEDDING_MODEL
        self.persist_directory = Path(persist_directory or settings.VECTOR_DB_DIR)
        if embeddings is not None:
            # Injected (tests): fake embeddings instead of the live Ollama model.
            self.embeddings = embeddings
        else:
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                base_url=settings.OLLAMA_HOST,
            )
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._vector_db: Optional[Chroma] = None

    @property
    def vector_db(self) -> Chroma:
        """Lazily create/open the shared, persistent, cosine-space collection."""
        if self._vector_db is None:
            logger.info(
                "Opening Chroma collection %r at %s",
                self.COLLECTION_NAME,
                self.persist_directory,
            )
            self._vector_db = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory),
                collection_metadata={"hnsw:space": "cosine"},
            )
            self._warn_if_not_cosine()
        return self._vector_db

    def _warn_if_not_cosine(self) -> None:
        """Collection metadata is fixed at creation - warn on a non-cosine store."""
        try:
            meta = self._vector_db._collection.metadata or {}
            if meta and meta.get("hnsw:space") != "cosine":
                logger.warning(
                    "Existing collection %r in %s was created without cosine space "
                    "(metadata=%s); delete the directory and re-index once to get "
                    "interpretable similarity scores.",
                    self.COLLECTION_NAME,
                    self.persist_directory,
                    meta,
                )
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ writes

    def add_chunks(self, docs: Sequence[Document], file_id: str) -> List[str]:
        """Add one file's chunks with deterministic ids, in embed-batch-size groups.

        Re-running with the same file yields the same ids, so Chroma upserts
        instead of duplicating (idempotent re-index).
        """
        batch_size = max(1, settings.EMBED_BATCH_SIZE)
        all_ids: List[str] = []
        for start in range(0, len(docs), batch_size):
            batch = list(docs[start:start + batch_size])
            ids = [
                f"{file_id}:{doc.metadata.get('chunk_index', start + i)}"
                for i, doc in enumerate(batch)
            ]
            self._add_with_retry(batch, ids)
            all_ids.extend(ids)
        return all_ids

    def _add_with_retry(
        self,
        docs: Sequence[Document],
        ids: Sequence[str],
        attempts: int = 3,
    ) -> None:
        """3 attempts, exponential backoff 1s/2s/4s, transport errors only.

        Any other exception fails the *file* (the manifest marks it `error`)
        rather than the whole scan.
        """
        delay = 1.0
        for attempt in range(1, attempts + 1):
            try:
                self.vector_db.add_documents(documents=list(docs), ids=list(ids))
                return
            except (httpx.TransportError, ConnectionError) as exc:
                if attempt == attempts:
                    logger.error("Embedding failed after %d attempts: %s", attempts, exc)
                    raise
                logger.warning(
                    "Transient embedding error (attempt %d/%d): %s - retrying in %.0fs",
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= 2

    def delete_file(self, file_id: str) -> None:
        """Remove every chunk belonging to one file."""
        self.vector_db.delete(where={"file_id": file_id})

    def prune_file(self, file_id: str, keep_ids: Sequence[str]) -> None:
        """Delete this file's chunks whose ids are not in `keep_ids`.

        Used after a successful re-add so a failed/partial previous attempt
        cannot leave stale chunks behind (idempotent replace).
        """
        keep = set(keep_ids)
        try:
            existing = self.vector_db._collection.get(where={"file_id": file_id})
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list existing chunks for %s: %s", file_id, exc)
            return
        stale = [cid for cid in (existing or {}).get("ids", []) if cid not in keep]
        if stale:
            logger.debug("Pruning %d stale chunks for %s", len(stale), file_id)
            self.vector_db.delete(ids=stale)

    def delete_collection(self) -> None:
        """Drop the whole collection (used for embedding-model rebuilds)."""
        if self._vector_db is not None:
            logger.info("Deleting Chroma collection %r", self.COLLECTION_NAME)
            self._vector_db.delete_collection()
            self._vector_db = None

    # ------------------------------------------------------------------ reads

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Cosine similarity search; returns (doc, score) with score = 1 - distance."""
        results = self.vector_db.similarity_search_with_score(query, k=k, filter=filter)
        return [(doc, 1.0 - distance) for doc, distance in results]

    def count(self) -> int:
        """Number of chunks stored in the collection."""
        try:
            return int(self.vector_db._collection.count())
        except Exception:
            return 0

    def collection_count(self) -> int:
        """Number of collections persisted in VECTOR_DB_DIR (backs /health)."""
        import chromadb

        path = str(self.persist_directory)
        if not Path(path).exists():
            return 0
        try:
            client = chromadb.PersistentClient(path=path)
            return len(client.list_collections())
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("collection_count failed: %s", exc)
            return 0
