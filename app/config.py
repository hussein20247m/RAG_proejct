
"""Configuration settings for FastAPI application.

Every path is anchored to the repository root (never the process CWD).
Values come from (in order): constructor kwargs, process environment,
the absolute `backend/app/.env` file, then the defaults below.
"""
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings (single source of truth for paths and tunables)."""

    # Paths — None means "derive from PROJECT_ROOT / other paths" (see _derive_paths).
    # All of them may be overridden via environment variables or the constructor.
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Optional[Path] = None       # top-level user drop zone (recursively scanned)
    STORAGE_DIR: Optional[Path] = None    # internal storage: api.db, vectors/, legacy pdfs/
    UPLOAD_DIR: Optional[Path] = None     # API uploads land in the drop zone -> one pipeline
    VECTOR_DB_DIR: Optional[Path] = None  # Chroma persistence directory

    # Database — derived from STORAGE_DIR when not explicitly set
    DATABASE_URL: str = ""

    # Ollama
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_MODEL: str = "qwen2.5:0.5b"

    # Chunking / retrieval / ingestion tunables
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 6
    MIN_SIMILARITY: float = 0.30
    EMBED_BATCH_SIZE: int = 32
    MAX_FILE_SIZE_MB: int = 50
    INDEX_ON_STARTUP: bool = True
    INGEST_EXTENSIONS: List[str] = [
        ".pdf", ".txt", ".md", ".markdown", ".log",
        ".csv", ".docx", ".html", ".htm",
    ]
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / ".env",  # absolute: not CWD-relative
        extra="ignore",
    )

    @field_validator("INGEST_EXTENSIONS", mode="before")
    @classmethod
    def _split_extensions(cls, value: object) -> object:
        """Allow a comma-separated string from the environment."""
        if isinstance(value, str):
            parts = [p.strip().lower() for p in value.split(",") if p.strip()]
            return [p if p.startswith(".") else f".{p}" for p in parts]
        return value

    @model_validator(mode="after")
    def _derive_paths(self) -> "Settings":
        """Fill in derived paths so nothing ever resolves against the CWD."""
        root = self.PROJECT_ROOT
        if self.DATA_DIR is None:
            self.DATA_DIR = root / "data"
        if self.STORAGE_DIR is None:
            self.STORAGE_DIR = root / "backend" / "data"
        if self.UPLOAD_DIR is None:
            self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        if self.VECTOR_DB_DIR is None:
            self.VECTOR_DB_DIR = self.STORAGE_DIR / "vectors"
        if not self.DATABASE_URL:
            self.DATABASE_URL = f"sqlite:///{self.STORAGE_DIR}/api.db"
        return self


settings = Settings()
