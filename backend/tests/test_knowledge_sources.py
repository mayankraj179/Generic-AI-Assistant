"""knowledge_sources config validation and the three adapters, with their
external systems mocked (a fake S3 client, a fake query runner). No
database or network needed."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import yaml
from botocore.exceptions import ClientError

from app.config.knowledge_sources import (
    DatabaseSourceConfig,
    FilesystemSourceConfig,
    ObjectStorageSourceConfig,
)
from app.config.loader import ConfigLoadError, load_all_assistant_configs, load_assistant_config
from app.ingestion.pipeline import DiscoveredSource
from app.ingestion.sources import (
    DatabaseSource,
    FilesystemSource,
    ObjectStorageSource,
    SourceError,
)
from app.ingestion.sources import base as sources_base

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

_BASE: dict[str, Any] = {
    "assistant_id": "ks_test",
    "display_name": "KS test",
    "description": "test",
    "tenant_id": "t",
    "model": {"provider": "azure_ai", "model_name": "m"},
    "system_prompt": "x",
}

_S3 = {
    "name": "bucket",
    "type": "object_storage",
    "bucket": "b",
    "access_key_id": "${S3_KEY}",
    "secret_access_key": "${S3_SECRET}",
    "access_labels": ["role:authenticated"],
}

_DB = {
    "name": "db",
    "type": "database",
    "url": "${DB_URL}",
    "access_labels": ["role:authenticated"],
    "queries": [{"name": "q", "key_column": "id", "sql": "SELECT id, v FROM t"}],
}


def _load(tmp_path: Path, sources: list[dict[str, Any]]):
    path = tmp_path / "a.yaml"
    path.write_text(yaml.safe_dump({**_BASE, "knowledge_sources": sources}), encoding="utf-8")
    return load_assistant_config(path)


# ---------- config ----------


def test_shipped_configs_load_and_existing_assistants_have_no_sources():
    configs = load_all_assistant_configs(CONFIGS_DIR)
    assert configs["hr_assistant"].knowledge_sources == []
    assert configs["finance_assistant"].knowledge_sources == []
    demo = configs["kb_demo_assistant"].knowledge_sources
    assert [(s.name, s.type) for s in demo] == [
        ("it_handbook", "filesystem"),
        ("policy_bucket", "object_storage"),
        ("office_db", "database"),
    ]


def test_valid_sources_of_every_type(tmp_path):
    fs = {"name": "docs", "type": "filesystem", "path": "x", "access_labels": ["dept:hr"]}
    config = _load(tmp_path, [fs, _S3, _DB])
    assert isinstance(config.knowledge_sources[0], FilesystemSourceConfig)
    assert isinstance(config.knowledge_sources[1], ObjectStorageSourceConfig)
    assert isinstance(config.knowledge_sources[2], DatabaseSourceConfig)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({**_S3, "type": "ftp"}, "unknown type 'ftp'"),
        ({k: v for k, v in _S3.items() if k != "type"}, "has no 'type'"),
        ({**_S3, "secret_access_key": "minio_dev_password"}, "literal secrets are not allowed"),
        ({**_S3, "access_key_id": "$S3_KEY"}, "literal secrets are not allowed"),
        ({**_DB, "url": "postgresql://u:p@h/db"}, "literal secrets are not allowed"),
        ({**_S3, "endpoint_url": "http://u:p@localhost:9000"}, "must not be embedded in a URL"),
        # extra="forbid": a stray credential field can't be smuggled in.
        ({**_S3, "password": "hunter2"}, "Extra inputs are not permitted"),
        ({**_S3, "access_labels": []}, "at least one label"),
        ({**_S3, "access_labels": ["hr"]}, "namespace:value"),
        ({**_S3, "name": "Bad Name"}, "lowercase"),
    ],
)
def test_invalid_source_config_fails_loading_with_a_clear_error(tmp_path, source, message):
    with pytest.raises(ConfigLoadError) as exc:
        _load(tmp_path, [source])
    assert message in str(exc.value)


@pytest.mark.parametrize(
    "source",
    [
        {**_S3, "secret_access_key": "s3cr3t-value"},
        {**_DB, "url": "postgresql://reader:s3cr3t-value@db/x"},
        {**_S3, "endpoint_url": "http://reader:s3cr3t-value@localhost:9000"},
    ],
)
def test_rejected_secret_is_not_echoed_in_the_error(tmp_path, source):
    with pytest.raises(ConfigLoadError) as exc:
        _load(tmp_path, [source])
    assert "s3cr3t-value" not in str(exc.value)
    assert "knowledge_sources.0" in str(exc.value)  # still names the field


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("DELETE FROM t", "single SELECT"),
        ("UPDATE t SET v = 1", "single SELECT"),
        ("SELECT 1; DROP TABLE t", "single statement"),
    ],
)
def test_database_queries_must_be_a_single_select(tmp_path, sql, message):
    source = {**_DB, "queries": [{"name": "q", "key_column": "id", "sql": sql}]}
    with pytest.raises(ConfigLoadError) as exc:
        _load(tmp_path, [source])
    assert message in str(exc.value)


def test_source_names_must_be_unique(tmp_path):
    with pytest.raises(ConfigLoadError) as exc:
        _load(tmp_path, [_S3, {**_DB, "name": "bucket"}])
    assert "duplicate knowledge source name(s): bucket" in str(exc.value)


def test_unset_env_var_fails_at_sync_not_at_load(tmp_path, monkeypatch):
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.setattr(sources_base, "_dotenv", lambda: {})
    config = _load(tmp_path, [_DB])  # loads fine
    with pytest.raises(SourceError, match="env var DB_URL"):
        sources_base.resolve_env_ref(config.knowledge_sources[0].url, field="url", source="db")


# ---------- filesystem ----------


def _fs(tmp_path: Path) -> FilesystemSource:
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "leave.md").write_text("Leave policy: 15 days.", encoding="utf-8")
    (root / "sub" / "travel.txt").write_text("Travel policy.", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")  # unsupported: not discovered
    (tmp_path / "secret.txt").write_text("outside the root", encoding="utf-8")
    config = FilesystemSourceConfig(
        name="docs", type="filesystem", path="docs", access_labels=["dept:hr"]
    )
    return FilesystemSource(config, base_dir=tmp_path)


async def test_filesystem_discovers_supported_files_with_root_relative_uris(tmp_path):
    source = _fs(tmp_path)
    uris = [d.uri for d in await source.discover()]
    assert uris == ["fs://docs/leave.md", "fs://docs/sub/travel.txt"]

    with_access = source.resolve_access(DiscoveredSource(uris[0], "filesystem"))
    assert with_access.access_labels == frozenset({"dept:hr"})
    parsed = await source.fetch(with_access)
    assert (parsed.title, parsed.raw_text) == ("leave", "Leave policy: 15 days.")


async def test_filesystem_fetch_cannot_escape_the_root(tmp_path):
    source = _fs(tmp_path)
    escape = source.resolve_access(DiscoveredSource("fs://docs/../secret.txt", "filesystem"))
    with pytest.raises(FileNotFoundError):
        await source.fetch(escape)


async def test_filesystem_missing_root_is_a_source_error(tmp_path):
    config = FilesystemSourceConfig(
        name="docs", type="filesystem", path="nope", access_labels=["dept:hr"]
    )
    with pytest.raises(SourceError, match="path does not exist"):
        await FilesystemSource(config, base_dir=tmp_path).discover()


# ---------- object storage (fake S3 client) ----------


class _FakeS3:
    def __init__(self, objects: dict[str, bytes], page_size: int = 2) -> None:
        self.objects = objects
        self.page_size = page_size
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_paginator(self, operation: str):
        assert operation == "list_objects_v2"
        fake = self

        class _Paginator:
            def paginate(self, **kwargs):
                fake.calls.append(("list", kwargs))
                keys = sorted(k for k in fake.objects if k.startswith(kwargs["Prefix"]))
                for i in range(0, len(keys), fake.page_size):
                    yield {"Contents": [{"Key": k} for k in keys[i : i + fake.page_size]]}

        return _Paginator()

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803 - boto3's casing
        self.calls.append(("get", {"Bucket": Bucket, "Key": Key}))
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        data = self.objects[Key]
        return {"ContentLength": len(data), "Body": io.BytesIO(data)}


def _s3_source(client: _FakeS3, **overrides: Any) -> ObjectStorageSource:
    config = ObjectStorageSourceConfig.model_validate(
        {**_S3, "name": "pol", "bucket": "kb", "prefix": "policies/", **overrides}
    )
    return ObjectStorageSource(config, client=client)


async def test_object_storage_lists_prefix_across_pages_and_reads_objects():
    client = _FakeS3(
        {
            "policies/travel.md": b"Claims within 30 days.",
            "policies/a/leave.txt": b"15 days leave.",
            "policies/a/": b"",  # folder marker
            "policies/logo.png": b"\x89PNG",  # unsupported
            "other/ignored.md": b"not under the prefix",
        }
    )
    source = _s3_source(client)
    discovered = await source.discover()
    assert [d.uri for d in discovered] == [
        "s3://pol/policies/a/leave.txt",
        "s3://pol/policies/travel.md",
    ]
    assert client.calls[0] == ("list", {"Bucket": "kb", "Prefix": "policies/"})

    parsed = await source.fetch(source.resolve_access(discovered[1]))
    assert (parsed.title, parsed.raw_text) == ("travel", "Claims within 30 days.")
    assert client.calls[-1] == ("get", {"Bucket": "kb", "Key": "policies/travel.md"})
    # Only list and get calls: the adapter never writes.
    assert {name for name, _ in client.calls} == {"list", "get"}


async def test_object_storage_over_max_objects_fails_instead_of_truncating():
    client = _FakeS3({f"policies/{i}.md": b"x" for i in range(4)})
    with pytest.raises(SourceError, match="max_objects=3"):
        await _s3_source(client, max_objects=3).discover()


async def test_object_storage_missing_object_is_file_not_found():
    source = _s3_source(_FakeS3({}))
    gone = source.resolve_access(DiscoveredSource("s3://pol/policies/gone.md", "object_storage"))
    with pytest.raises(FileNotFoundError):
        await source.fetch(gone)


# ---------- database (fake query runner) ----------


def _db_source(columns, rows, **overrides: Any) -> tuple[DatabaseSource, list]:
    seen: list = []

    async def runner(config, url):
        seen.append((config.queries[0].sql, config.queries[0].params, url))
        return {"office": (columns, rows)}

    config = DatabaseSourceConfig.model_validate(
        {
            **_DB,
            "name": "hr_db",
            "queries": [
                {
                    "name": "office",
                    "key_column": "code",
                    "sql": "SELECT code, city, ext FROM offices WHERE active = :active",
                    "params": {"active": True},
                }
            ],
            **overrides,
        }
    )
    return DatabaseSource(config, runner=runner), seen


@pytest.fixture
def db_url(monkeypatch):
    monkeypatch.setenv("DB_URL", "postgresql+asyncpg://reader:pw@localhost/db")


async def test_database_renders_one_document_per_row(db_url):
    source, seen = _db_source(
        ["code", "city", "ext"], [("PUN", "Pune", "4410"), ("NY/C", "New York", None)]
    )
    discovered = await source.discover()
    # The reviewed SQL and bound params pass through untouched.
    assert seen == [
        (
            "SELECT code, city, ext FROM offices WHERE active = :active",
            {"active": True},
            "postgresql+asyncpg://reader:pw@localhost/db",
        )
    ]
    assert [d.uri for d in discovered] == ["db://hr_db/office/PUN", "db://hr_db/office/NY%2FC"]

    pune = await source.fetch(source.resolve_access(discovered[0]))
    assert pune.title == "office PUN"
    assert pune.raw_text == "code: PUN\ncity: Pune\next: 4410"
    nyc = await source.fetch(source.resolve_access(discovered[1]))
    assert nyc.raw_text == "code: NY/C\ncity: New York"  # null column omitted


@pytest.mark.parametrize(
    ("columns", "rows", "message"),
    [
        (["code", "city"], [("A", "x"), ("A", "y")], "duplicate code 'A'"),
        (["code", "city"], [(None, "x")], "empty code"),
        (["city"], [("x",)], "no key_column 'code'"),
        (["code"], [("A",), ("B",), ("C",)], "more than max_rows=2"),
    ],
)
async def test_database_rejects_unusable_results(db_url, columns, rows, message):
    source, _ = _db_source(columns, rows, max_rows=2)
    with pytest.raises(SourceError, match=message):
        await source.discover()


async def test_database_only_supports_postgres(monkeypatch):
    from app.ingestion.sources.database import run_read_only_queries

    config = DatabaseSourceConfig.model_validate(_DB)
    with pytest.raises(SourceError, match="only PostgreSQL"):
        await run_read_only_queries(config, "mysql+aiomysql://u:p@h/db")
