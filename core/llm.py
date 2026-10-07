"""LLM configuration and prompts."""
import logging
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama

from ..app.config import settings

logger = logging.getLogger(__name__)

#: Returned verbatim when no chunk clears the relevance threshold.
REFUSAL_TEXT = (
    "I don't have enough information in the indexed documents to answer that."
)

_SYSTEM_PROMPT = """You answer questions STRICTLY from the numbered context below.
Rules:
- Use only facts present in the context. No outside knowledge.
- Cite supporting chunks inline as [1], [2] etc. matching the context numbering.
- If the context does not contain the answer, reply exactly:
  "I don't have enough information in the indexed documents to answer that."
  Do not guess.
- Be concise (<= 5 sentences unless asked otherwise)."""

_HUMAN_PROMPT = """Context:
{context}

Question: {question}"""


class LLMManager:
    """Manages LLM configuration and prompts."""

    def __init__(self, model_name: Optional[str] = None, llm: Any = None):
        # `llama2` default removed: model comes from settings (or an injected fake in tests).
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.llm = (
            llm
            if llm is not None
            else ChatOllama(
                model=self.model_name,
                temperature=0.1,  # grounded extractive QA wants near-determinism
                base_url=settings.OLLAMA_HOST,
            )
        )

    def get_rag_prompt(self) -> ChatPromptTemplate:
        """Grounded, citation-required prompt with an explicit refusal path."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", _HUMAN_PROMPT),
            ]
        )
