"""Database models and session management."""
import logging

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

# Create engine from the single-sourced settings (never CWD-relative)
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # Needed for SQLite
    echo=False,  # Set to True for SQL debugging
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """WAL + busy_timeout so scans and API requests don't hit 'database is locked'."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
    finally:
        cursor.close()


# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class PDFMetadata(Base):
    """PDF metadata table."""
    __tablename__ = "pdfs"

    pdf_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    # Was UNIQUE (one collection per PDF, defect #19). The plan moves everything
    # into the single constant "documents" collection, so uniqueness is dropped.
    collection_name = Column(String, nullable=False)
    upload_timestamp = Column(DateTime, nullable=False)
    doc_count = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    is_sample = Column(Boolean, default=False)
    file_path = Column(String)


class ChatSession(Base):
    """Chat session table."""
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    last_active = Column(DateTime, nullable=False)


class ChatMessage(Base):
    """Chat message table."""
    __tablename__ = "messages"

    message_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    sources = Column(JSON)
    timestamp = Column(DateTime, nullable=False)


class IndexedFile(Base):
    """Incremental-indexing manifest: one row per file in the drop zone."""
    __tablename__ = "indexed_files"

    file_id = Column(String, primary_key=True)   # sha256(rel_path)[:16]
    rel_path = Column(String, unique=True, nullable=False)
    content_hash = Column(String, nullable=False)  # sha256 of file bytes
    size = Column(Integer, nullable=False)
    mtime = Column(Float, nullable=False)
    file_type = Column(String, nullable=False)
    status = Column(String, nullable=False)        # indexed | error
    error = Column(String)                         # nullable, <=500 chars
    chunk_count = Column(Integer, nullable=False, default=0)
    page_count = Column(Integer, nullable=False, default=0)
    indexed_at = Column(DateTime, nullable=False)


class IndexMeta(Base):
    """Key/value index metadata (embedding_model, last_scan_at)."""
    __tablename__ = "index_meta"

    key = Column(String, primary_key=True)
    value = Column(String)


def _drop_legacy_collection_uniqueness() -> None:
    """One-time migration: remove the per-PDF UNIQUE on ``pdfs.collection_name``.

    SQLite keeps that implicit unique index inside the table definition, so an
    existing database needs a table rebuild (all rows are copied; the table's
    columns and name are unchanged). Fresh databases have no such index and
    this is a no-op.
    """
    if engine.url.get_backend_name() != "sqlite":
        return
    with engine.begin() as conn:
        legacy = None
        for index in conn.exec_driver_sql("PRAGMA index_list(pdfs)").fetchall():
            # (seq, name, unique, origin, partial)
            _, name, is_unique, origin, _ = index
            if is_unique and origin == "u":
                columns = [
                    row[2]
                    for row in conn.exec_driver_sql(
                        f'PRAGMA index_info("{name}")'
                    ).fetchall()
                ]
                if columns == ["collection_name"]:
                    legacy = name
                    break
        if legacy is None:
            return
        logger.info("Migrating pdfs table: dropping UNIQUE on collection_name")
        conn.exec_driver_sql("ALTER TABLE pdfs RENAME TO pdfs_legacy_unique")
        conn.exec_driver_sql(
            "CREATE TABLE pdfs ("
            "pdf_id VARCHAR NOT NULL, name VARCHAR NOT NULL, "
            "collection_name VARCHAR NOT NULL, upload_timestamp DATETIME NOT NULL, "
            "doc_count INTEGER NOT NULL, page_count INTEGER NOT NULL, "
            "is_sample BOOLEAN, file_path VARCHAR, PRIMARY KEY (pdf_id))"
        )
        conn.exec_driver_sql(
            "INSERT INTO pdfs (pdf_id, name, collection_name, upload_timestamp, "
            "doc_count, page_count, is_sample, file_path) "
            "SELECT pdf_id, name, collection_name, upload_timestamp, doc_count, "
            "page_count, is_sample, file_path FROM pdfs_legacy_unique"
        )
        conn.exec_driver_sql("DROP TABLE pdfs_legacy_unique")


# Create all tables
Base.metadata.create_all(bind=engine)
_drop_legacy_collection_uniqueness()
