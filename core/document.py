
"""Document processing: loading (per format), cleaning and chunking."""
import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    BSHTMLLoader,
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..app.config import settings

logger = logging.getLogger(__name__)

# Loader metadata keys kept on documents/chunks (everything else is dropped as noise).
_LOADER_METADATA_KEYS = ("source", "title", "page", "row")

# C0 control chars except \t (0x09) and \n (0x0a); \x7f (DEL) included.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# Runs of 3+ blank lines collapse to a single blank line.
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Conservative text normalization applied before splitting.

    - normalize line endings (\\r\\n / \\r -> \\n);
    - strip NUL bytes and other C0 control chars except \\n/\\t;
    - collapse runs of >=3 blank lines to exactly one blank line;
    - strip the document's edges.
    Nothing else: over-cleaning hurts retrieval.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


class DocumentProcessor:
    """Loads documents of every supported format and splits them into chunks."""

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # == defaults, pinned
            keep_separator=True,                 # don't eat paragraph/newline delimiters
            add_start_index=True,                # char offset of chunk within its page/row
        )

    @staticmethod
    def is_supported(file_path: Path) -> bool:
        """True when the file's extension is on the configured allow-list."""
        ext = Path(file_path).suffix.lower()
        return any(ext == allowed.lower() for allowed in settings.INGEST_EXTENSIONS)

    def load(self, file_path: Path) -> List[Document]:
        """Load a file into per-page/per-row Documents using the right loader."""
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))  # default mode: one Document per page
        elif ext in {".txt", ".md", ".markdown", ".log"}:
            # Markdown is read as plain text on purpose: degrades gracefully and the
            # only structured alternative needs the excluded `unstructured` package.
            loader = TextLoader(str(path), autodetect_encoding=True)
        elif ext == ".csv":
            loader = CSVLoader(file_path=str(path), encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(str(path))
        elif ext in {".html", ".htm"}:
            loader = BSHTMLLoader(str(path), open_encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext or '<no extension>'}")

        logger.debug("Loading %s with %s", path, type(loader).__name__)
        documents = loader.load()
        for doc in documents:
            doc.metadata = {
                key: value
                for key, value in doc.metadata.items()
                if key in _LOADER_METADATA_KEYS
            }
        return documents

    def load_pdf(self, file_path: Path) -> List[Document]:
        """Backward-compatible wrapper around load() for PDF files."""
        return self.load(Path(file_path))

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Clean each document, split it per source, drop tiny/whitespace-only chunks.

        Documents are split individually (never concatenated first) so per-page /
        per-row metadata is inherited by every chunk.
        """
        chunks: List[Document] = []
        for doc in documents:
            text = clean_text(doc.page_content)
            if not text:
                continue
            cleaned = Document(page_content=text, metadata=dict(doc.metadata))
            for chunk in self.splitter.split_documents([cleaned]):
                if len(chunk.page_content.strip()) < 20:
                    continue  # whitespace-only / trivial fragments
                chunks.append(chunk)
        return chunks
