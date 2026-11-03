"""
database/connection.py — Database Connection Management

Reads DATABASE_URL from the environment. Defaults to SQLite for local dev.
The async-compatible Session is provided via a FastAPI dependency.
"""
import os
import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from src.database.models import Base

logger = logging.getLogger(__name__)

# Read connection string from env; fall back to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./guardian.db")

# SQLite requires connect_args for thread safety in a multi-threaded server context
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

_engine_kwargs = {}
if DATABASE_URL in ("sqlite://", "sqlite:///:memory:"):
    # A single shared connection, otherwise each pooled connection would see
    # its own empty in-memory database (used by the test-suite).
    _engine_kwargs["poolclass"] = StaticPool
elif not DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["pool_pre_ping"] = True

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # set to True for SQL query debugging
    **_engine_kwargs,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _redact(url: str) -> str:
    """Hide credentials in a database URL before logging it."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


def _add_missing_columns(bind) -> None:
    """
    Minimal additive migration: create_all() never alters existing tables, so
    databases created by an older release would be missing new nullable
    columns. Add them in place. Destructive changes are never attempted.
    """
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    with bind.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present or not column.nullable:
                    continue
                col_type = column.type.compile(dialect=bind.dialect)
                logger.info("Migrating: adding column %s.%s", table.name, column.name)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {column.name} {col_type}'))


def init_db(bind=None) -> None:
    """Create all tables if they do not already exist and add new columns."""
    bind = bind or engine
    logger.info("Initializing database at: %s", _redact(str(bind.url)))
    Base.metadata.create_all(bind=bind)
    _add_missing_columns(bind)
    logger.info("Database tables created/verified.")


def get_db():
    """
    FastAPI dependency that provides a database session.
    Usage: db: Session = Depends(get_db)
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
