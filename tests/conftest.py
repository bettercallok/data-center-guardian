"""
tests/conftest.py — Shared pytest fixtures

Environment is configured *before* the app is imported so settings pick it up:
  - an in-memory SQLite database (shared StaticPool connection)
  - a known admin API key
  - rate limiting disabled (tests hammer endpoints far faster than any limit)
"""
import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["GUARDIAN_ADMIN_API_KEY"] = "test-admin-key"
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ.setdefault("PROBE_SAMPLES", "300")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

ADMIN_KEY = os.environ["GUARDIAN_ADMIN_API_KEY"]


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_headers():
    return {"X-API-Key": ADMIN_KEY}


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
