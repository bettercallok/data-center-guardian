"""
tests/conftest.py — Shared pytest fixtures

Uses an in-memory SQLite database so tests run without Docker or Postgres.
"""
import pytest
import numpy as np
import xgboost as xgb
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base
from src.database.connection import get_db
from src.api.main import app

# In-memory SQLite — fast, no disk I/O
TEST_DB_URL = "sqlite:///:memory:"

test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="session")
def client():
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def healthy_telemetry():
    """A perfectly healthy drive — all SMART values at zero."""
    return {
        "smart_5_raw": 0,
        "smart_187_raw": 0,
        "smart_188_raw": 0,
        "smart_197_raw": 0,
        "smart_198_raw": 0,
    }


@pytest.fixture(scope="session")
def critical_telemetry():
    """A drive in catastrophic failure — all SMART values maxed."""
    return {
        "smart_5_raw": 500,
        "smart_187_raw": 200,
        "smart_188_raw": 1000,
        "smart_197_raw": 500,
        "smart_198_raw": 500,
    }


@pytest.fixture(scope="session")
def ood_telemetry():
    """Out-of-distribution data — extreme values far outside training range."""
    return {
        "smart_5_raw": 65535,
        "smart_187_raw": 65535,
        "smart_188_raw": 65535,
        "smart_197_raw": 65535,
        "smart_198_raw": 65535,
    }
