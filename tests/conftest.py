"""Shared fixtures.

Database tests run on TEST_DATABASE_URL when it is set (CI uses a pgvector service container).
Otherwise they start an embedded Postgres with pgvector through `pgserver` (dev dependency), so
no Docker is needed locally. The inventory is seeded once per test session with the hashing embedder.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from propsearch.config import Settings


def _database_url() -> tuple[str, object | None, str | None]:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url, None, None
    pgserver = pytest.importorskip("pgserver", reason="set TEST_DATABASE_URL or install pgserver")
    tmp = tempfile.mkdtemp(prefix="propsearch-pg-")
    srv = pgserver.get_server(tmp, cleanup_mode="stop")
    return srv.get_uri().replace("postgresql://", "postgresql+psycopg://"), srv, tmp


@pytest.fixture(scope="session")
def db_url():
    url, srv, tmp = _database_url()
    from propsearch.db import get_engine
    from propsearch.seed import ensure_seeded

    ensure_seeded(get_engine(url), "hashing", reset=True)
    yield url
    get_engine(url).dispose()
    if srv is not None:
        srv.cleanup()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def settings(db_url) -> Settings:
    return Settings(
        database_url=db_url,
        parser="rules",
        embedder="hashing",
        reranker="none",
        langfuse_public_key="",
        langfuse_secret_key="",
    )


@pytest.fixture(scope="session")
def Session(db_url):
    from propsearch.db import get_engine, session_factory

    return session_factory(get_engine(db_url))


@pytest.fixture(scope="session")
def engine(settings, Session):
    from propsearch.pipeline import SearchEngine

    return SearchEngine(settings, Session)


@pytest.fixture(scope="session")
def client(settings):
    from fastapi.testclient import TestClient

    from propsearch.api import create_app

    with TestClient(create_app(settings)) as c:
        yield c
