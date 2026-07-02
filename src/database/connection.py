"""
database/connection.py — Database Connection Management

Reads DATABASE_URL from the environment. Defaults to SQLite for local dev.
The async-compatible Session is provided via a FastAPI dependency.
"""
import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.database.models import Base

logger = logging.getLogger(__name__)

# Read connection string from env; fall back to local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./guardian.db")

# SQLite requires connect_args for thread safety in a multi-threaded server context
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # set to True for SQL query debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    logger.info("Initializing database at: %s", DATABASE_URL)
    Base.metadata.create_all(bind=engine)
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
