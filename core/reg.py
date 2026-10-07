"""RAG pipeline implementation: retrieve -> threshold/dedupe -> generate."""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .llm import LLMManager, REFUSAL_TEXT
from ..app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Answer plus the retrieved chunks (with scores) and observability metadata."""
    text: str
    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def format_context(hits: Sequence[Tuple[Document, float]]) -> str:
    """Render chunks as numbered blocks so `[n]` citations are verifiable.

    [1] (file: reports/q3.pdf, page: 4, type: pdf)
    <chunk text>
    """
    blocks = []
    for position, (doc, _score) in enumerate(hits, start=1):
        meta = doc.metadata
        header = (
            f"[{position}] (file: {meta.get('source_path') or meta.get('filename', 'unknown')}, "
            f"page: {meta.get('page')}, type: {meta.get('file_type', '')})"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _normalized_prefix(text: str, length: int = 200) -> str:
    """Whitespace-collapsed, lowercased prefix used for duplicate detection."""
    return re.sub(r"\s+", " ", text[:length]).strip().lower()


class RAGPipeline:
    """Manages the RAG (Retrieval Augmented Generation) pipeline.

    Plain cosine similarity (not MMR), top-k from settings, cosine-threshold
    filtering, metadata filtering via `pdf_ids`, duplicate collapsing, then a
    single grounded LLM call that answers with `[n]` citations.
    """

    def __init__(
        self,
        vector_db: Any,
        llm_manager: LLMManager,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ):
        self.vector_db = vector_db
        self.llm_manager = llm_manager
        self.top_k = top_k if top_k is not None else settings.TOP_K
        self.min_similarity = (
            min_similarity if min_similarity is not None else settings.MIN_SIMILARITY
        )

    # -------------------------------------------------------------- retrieval

    @staticmethod
    def _build_filter(pdf_ids: Optional[Sequence[str]]) -> Optional[Dict[str, Any]]:
        """Consume QueryRequest.pdf_ids against the `file_id` metadata key."""
        if not pdf_ids:
            return None
        if len(pdf_ids) == 1:
            return {"file_id": pdf_ids[0]}
        return {"file_id": {"$in": list(pdf_ids)}}

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        pdf_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[List[Tuple[Document, float]], Dict[str, Any]]:
        """Search, dedupe, apply the relevance floor. Returns hits + metadata."""
        started = time.perf_counter()
        k = max(1, min(12, top_k if top_k is not None else self.top_k))
        floor = self.min_similarity if min_score is None else min_score
        floor = max(0.0, floor)  # clamped to 0 to permit forcing retrieval

        raw_hits = self.vector_db.similarity_search_with_score(
            question, k=k, filter=self._build_filter(pdf_ids)
        )
        hits = sorted(raw_hits, key=lambda hit: hit[1], reverse=True)

        # Dedupe: (a) identical chunk ids; (b) identical first-200-char prefix
        # (byte-identical copies of one file in two directories) - keep the
        # higher score, which the sort already puts first.
        seen_ids: set = set()
        seen_prefixes: set = set()
        deduped: List[Tuple[Document, float]] = []
        for doc, score in hits:
            meta = doc.metadata
            chunk_key = (meta.get("file_id"), meta.get("chunk_index"))
            prefix = _normalized_prefix(doc.page_content)
            if chunk_key in seen_ids or prefix in seen_prefixes:
                continue
            seen_ids.add(chunk_key)
            seen_prefixes.add(prefix)
            deduped.append((doc, score))

        above = [(doc, score) for doc, score in deduped if score >= floor]
        metadata = {
            "top_score": round(above[0][1], 4) if above else None,
            "chunks_retrieved": len(above),
            "min_score": round(floor, 4),
            "min_score_hit": bool(above),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        return above, metadata

    # ------------------------------------------------------------- generation

    def answer(
        self,
        question: str,
        pdf_ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> RAGResult:
        """Retrieve then generate; returns text + scored sources + metadata."""
        hits, metadata = self.retrieve(
            question, top_k=top_k, min_score=min_score, pdf_ids=pdf_ids
        )
        if not hits:
            # Nothing above the floor: the model never sees the context.
            metadata["model"] = self.llm_manager.model_name
            logger.info("No chunks above threshold for question: %.80s", question)
            return RAGResult(
                text=REFUSAL_TEXT,
                documents=[],
                scores=[],
                metadata=metadata,
            )

        context = format_context(hits)
        chain = self.llm_manager.get_rag_prompt() | self.llm_manager.llm | StrOutputParser()
        text = chain.invoke({"context": context, "question": question})
        metadata["model"] = self.llm_manager.model_name
        logger.info(
            "Answered question with %d sources (top_score=%s)",
            len(hits),
            metadata.get("top_score"),
        )
        return RAGResult(
            text=text,
            documents=[doc for doc, _ in hits],
            scores=[score for _, score in hits],
            metadata=metadata,
        )

    def get_response(self, question: str) -> str:
        """Backward-compatible string-only accessor."""
        return self.answer(question).text
