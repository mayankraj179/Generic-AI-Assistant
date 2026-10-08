"""LIVE checks of the three knowledge-source adapters against the local
stack: the sample handbook on disk, MinIO, and the office_directory table.

Skipped unless RUN_LIVE_SOURCE_TESTS=1. Needs `docker compose up -d
postgres minio` and `python scripts/seed_kb_demo.py`. Embeds with the
offline hash embedder into a throwaway tenant, so it costs nothing and
never touches the demo assistant's real index.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import boto3
import pytest
from dotenv import dotenv_values
from sqlalchemy.engine import make_url

from app.config.knowledge_sources import DatabaseSourceConfig, ObjectStorageSourceConfig
from app.config.loader import load_all_assistant_configs
from app.db.models import EMBEDDING_DIM
from app.ingestion.sources import DatabaseSource, ObjectStorageSource, SourceError, build_source
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.source_sync import sync_source

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_SOURCE_TESTS") != "1",
    reason="live source test requires RUN_LIVE_SOURCE_TESTS=1 and the seeded local stack",
)

BACKEND = Path(__file__).resolve().parent.parent
CONFIG = load_all_assistant_configs(BACKEND / "configs")["kb_demo_assistant"]
SOURCES = {source.name: source for source in CONFIG.knowledge_sources}


def _env(name: str, default: str | None = None) -> str:
    return os.environ.get(name) or dotenv_values(BACKEND / ".env").get(name) or default or ""


@pytest.fixture
def run():
    embedder = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=embedder)
    tenant = f"tenant-live-src-{uuid.uuid4().hex[:10]}"

    async def sync(source):
        return await sync_source(
            source,
            source_type=source.source_type,
            ingestion=service,
            tenant_id=tenant,
            assistant_id="kb_demo_live_test",
            embedder=embedder,
        )

    return sync


@pytest.mark.parametrize(
    ("name", "expected_uris"),
    [
        ("it_handbook", ["fs://it_handbook/it_support_handbook.md"]),
        ("policy_bucket", ["s3://policy_bucket/policies/travel_expense_policy.md"]),
        (
            "office_db",
            [f"db://office_db/office_directory/{c}" for c in ("BLR", "MUM", "NYC", "PUN")],
        ),
    ],
)
async def test_live_source_syncs_then_resync_is_a_noop(run, name, expected_uris):
    source = build_source(SOURCES[name], base_dir=BACKEND)
    first = await run(source)
    assert first.status == "ok", first.error
    assert sorted(item.uri for item in await source.discover()) == expected_uris
    assert (first.created, first.failed) == (len(expected_uris), 0)

    second = await run(source)
    assert (second.created, second.updated, second.unchanged) == (0, 0, len(expected_uris))
    assert second.chunks_embedded == 0


async def test_live_database_row_text_and_inactive_row_excluded():
    source = build_source(SOURCES["office_db"], base_dir=BACKEND)
    discovered = {item.uri: item for item in await source.discover()}
    assert "db://office_db/office_directory/LDN" not in discovered  # is_active = false
    pune = await source.fetch(
        source.resolve_access(discovered["db://office_db/office_directory/PUN"])
    )
    assert pune.title == "office_directory PUN"
    assert "it_helpdesk_extension: 4410" in pune.raw_text


async def test_live_minio_new_object_is_ingested_and_deleted_object_retired(run):
    admin = boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        region_name="us-east-1",
        aws_access_key_id=_env("MINIO_ROOT_USER", "minio_admin"),
        aws_secret_access_key=_env("MINIO_ROOT_PASSWORD", "minio_dev_password"),
    )
    prefix = f"live-test/{uuid.uuid4().hex[:8]}/"
    config = ObjectStorageSourceConfig.model_validate(
        {**SOURCES["policy_bucket"].model_dump(), "prefix": prefix}
    )
    source = ObjectStorageSource(config)
    admin.put_object(Bucket="kb-demo", Key=f"{prefix}a.md", Body=b"First policy.")
    try:
        assert (await run(source)).created == 1
        admin.put_object(Bucket="kb-demo", Key=f"{prefix}b.md", Body=b"Second policy.")
        result = await run(source)
        assert (result.created, result.unchanged) == (1, 1)
        admin.delete_object(Bucket="kb-demo", Key=f"{prefix}a.md")
        result = await run(source)
        assert result.retired_items == [f"s3://policy_bucket/{prefix}a.md"]
    finally:
        for key in ("a.md", "b.md"):
            admin.delete_object(Bucket="kb-demo", Key=f"{prefix}{key}")


async def test_live_minio_reader_cannot_write():
    source = build_source(SOURCES["policy_bucket"], base_dir=BACKEND)
    await source.discover()  # builds the client with the reader's credentials
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError, match="AccessDenied"):
        source._s3().put_object(Bucket="kb-demo", Key="policies/x.md", Body=b"x")


def _admin_db_source(monkeypatch, sql: str, **overrides) -> DatabaseSource:
    # The admin role CAN write, so this proves the READ ONLY transaction
    # itself blocks writes, independent of the reader role's grants.
    admin_url = make_url(_env("DATABASE_URL")).set(database="kb_demo_source")
    monkeypatch.setenv("KB_LIVE_ADMIN_DB_URL", admin_url.render_as_string(hide_password=False))
    config = DatabaseSourceConfig.model_validate(
        {
            "name": "admin_db",
            "type": "database",
            "url": "${KB_LIVE_ADMIN_DB_URL}",
            "access_labels": ["scope:development"],
            "queries": [{"name": "q", "key_column": "office_code", "sql": sql}],
            **overrides,
        }
    )
    return DatabaseSource(config)


async def test_live_database_write_through_a_cte_is_blocked_by_read_only(monkeypatch):
    source = _admin_db_source(
        monkeypatch,
        "WITH d AS (DELETE FROM office_directory RETURNING office_code) SELECT * FROM d",
    )
    with pytest.raises(SourceError, match="read-only transaction"):
        await source.discover()
    rows = await _admin_db_source(
        monkeypatch, "SELECT office_code FROM office_directory"
    ).discover()
    assert len(rows) == 5  # nothing was deleted


async def test_live_database_statement_timeout(monkeypatch):
    source = _admin_db_source(
        monkeypatch,
        "SELECT office_code, pg_sleep(2) FROM office_directory",
        statement_timeout_ms=300,
    )
    with pytest.raises(SourceError, match="statement timeout"):
        await source.discover()


async def test_live_database_row_cap(monkeypatch):
    source = _admin_db_source(monkeypatch, "SELECT office_code FROM office_directory", max_rows=3)
    with pytest.raises(SourceError, match="more than max_rows=3"):
        await source.discover()
