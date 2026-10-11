"""LIVE database-source checks against real, disposable engines: PostgreSQL
(the compose stack's), MySQL 8.4, SQL Server 2022 and Oracle 23ai Free.

Skipped unless RUN_LIVE_DB_ENGINE_TESTS=1. Each engine is tested only when
its admin URL is reachable; LIVE_DB_ENGINES=mysql,oracle narrows the set.
Start throwaway containers (dev-only passwords, never real ones):

    docker run -d --name kbsrc-mysql -e MYSQL_ROOT_PASSWORD=root_dev_pw -p 13306:3306 mysql:8.4
    docker run -d --name kbsrc-mssql -e ACCEPT_EULA=Y -e MSSQL_SA_PASSWORD=Sa_dev_Passw0rd! \
        -p 11433:1433 mcr.microsoft.com/mssql/server:2022-latest
    docker run -d --name kbsrc-oracle -e ORACLE_PASSWORD=sys_dev_pw -p 11521:1521 \
        gvenzl/oracle-free:23-slim-faststart

Override the admin URLs with LIVE_{POSTGRESQL,MYSQL,MSSQL,ORACLE}_ADMIN_URL.
Every run recreates a small employee_directory table and a login that can
only SELECT it; sync results go into a throwaway tenant with the offline
hash embedder.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.config.knowledge_sources import DatabaseQueryConfig, DatabaseSourceConfig
from app.db.models import EMBEDDING_DIM
from app.ingestion.sources import DatabaseSource, SourceError
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.source_sync import sync_source

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_DB_ENGINE_TESTS") != "1",
    reason="live engine test requires RUN_LIVE_DB_ENGINE_TESTS=1 and running databases",
)

BACKEND = Path(__file__).resolve().parent.parent
READER = "kb_engine_reader"
READER_PASSWORD = "Reader_dev_Passw0rd1"
ROWS = [
    (1, "HR", "Recruiter", True, "2024-01-15"),
    (2, "IT", "Engineer", True, "2023-06-01"),
    (3, "Finance", "Analyst", False, "2022-03-20"),
]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name) or dotenv_values(BACKEND / ".env").get(name) or default


@dataclass
class Engine:
    dialect: str
    admin_url: str
    reader_url: str
    table: str  # as the reader names it
    setup: list[str]  # run by the admin, in order; failures in "-" lines are ignored
    heavy_sql: str  # a SELECT that runs for well over a second
    active: Any = True  # the value bound for :active
    # Seconds to wait after recreating the table. Oracle refuses a READ ONLY
    # transaction on a table whose DDL is newer than its snapshot
    # (ORA-01466), and that check has coarse time granularity.
    settle_s: float = 0.0

    def recreate(self) -> None:
        self.run_admin(self.setup)
        time.sleep(self.settle_s)

    def run_admin(self, statements: list[str]) -> None:
        engine = create_engine(self.admin_url, isolation_level="AUTOCOMMIT")
        try:
            with engine.connect() as conn:
                for statement in statements:
                    tolerant = statement.startswith("-")
                    try:
                        conn.execute(text(statement.lstrip("-")))
                    except Exception:
                        if not tolerant:
                            raise
        finally:
            engine.dispose()

    def admin_count(self) -> int:
        engine = create_engine(self.admin_url)
        try:
            with engine.connect() as conn:
                return int(conn.execute(text(f"SELECT COUNT(*) FROM {self.table}")).scalar_one())
        finally:
            engine.dispose()


def _inserts(table: str, render: Callable[[tuple], str]) -> list[str]:
    return [f"INSERT INTO {table} VALUES ({render(row)})" for row in ROWS]


def _mysql() -> Engine:
    admin = _env("LIVE_MYSQL_ADMIN_URL", "mysql+pymysql://root:root_dev_pw@127.0.0.1:13306/mysql")
    url = make_url(admin)
    table = "kb_engine_test.employee_directory"
    return Engine(
        dialect="mysql",
        admin_url=admin,
        reader_url=url.set(
            username=READER, password=READER_PASSWORD, database="kb_engine_test"
        ).render_as_string(hide_password=False),
        table=table,
        setup=[
            "CREATE DATABASE IF NOT EXISTS kb_engine_test",
            f"DROP USER IF EXISTS '{READER}'@'%'",
            f"CREATE USER '{READER}'@'%' IDENTIFIED BY '{READER_PASSWORD}'",
            f"GRANT SELECT ON kb_engine_test.* TO '{READER}'@'%'",
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (employee_id INT PRIMARY KEY, department VARCHAR(50), "
            "job_title VARCHAR(80), active BOOLEAN, hired DATE)",
            *_inserts(table, lambda r: f"{r[0]}, '{r[1]}', '{r[2]}', {int(r[3])}, '{r[4]}'"),
        ],
        heavy_sql="SELECT 1 AS employee_id, COUNT(*) AS n FROM information_schema.columns a "
        "CROSS JOIN information_schema.columns b CROSS JOIN information_schema.columns c",
    )


def _mssql() -> Engine:
    admin = _env(
        "LIVE_MSSQL_ADMIN_URL", "mssql+pymssql://sa:Sa_dev_Passw0rd!@127.0.0.1:11433/master"
    )
    url = make_url(admin)
    return Engine(
        dialect="mssql",
        admin_url=url.set(database="kb_engine_test").render_as_string(hide_password=False),
        reader_url=url.set(
            username=READER, password=READER_PASSWORD, database="kb_engine_test"
        ).render_as_string(hide_password=False),
        table="dbo.employee_directory",
        setup=[
            f"IF USER_ID('{READER}') IS NOT NULL DROP USER {READER}",
            "DROP TABLE IF EXISTS dbo.employee_directory",
            "CREATE TABLE dbo.employee_directory (employee_id INT PRIMARY KEY, "
            "department NVARCHAR(50), job_title NVARCHAR(80), active BIT, hired DATE)",
            *_inserts(
                "dbo.employee_directory",
                lambda r: f"{r[0]}, N'{r[1]}', N'{r[2]}', {int(r[3])}, '{r[4]}'",
            ),
            f"CREATE USER {READER} FOR LOGIN {READER}",
            f"ALTER ROLE db_datareader ADD MEMBER {READER}",
        ],
        heavy_sql="SELECT 1 AS employee_id, COUNT_BIG(*) AS n FROM sys.all_objects a "
        "CROSS JOIN sys.all_objects b CROSS JOIN sys.all_objects c "
        # Evaluated per row, so the optimizer can't shortcut the count.
        "WHERE CHECKSUM(a.object_id, b.object_id, c.object_id) = 7",
    )


def _mssql_server_setup(engine: Engine) -> None:
    """Database and login live at server level, so they're created from master."""
    master = make_url(engine.admin_url).set(database="master").render_as_string(hide_password=False)
    Engine("mssql", master, "", "", [], "").run_admin(
        [
            "IF DB_ID('kb_engine_test') IS NULL CREATE DATABASE kb_engine_test",
            f"IF SUSER_ID('{READER}') IS NOT NULL DROP LOGIN {READER}",
            f"CREATE LOGIN {READER} WITH PASSWORD = '{READER_PASSWORD}', "
            "DEFAULT_DATABASE = kb_engine_test",
        ]
    )


def _oracle() -> Engine:
    admin = _env(
        "LIVE_ORACLE_ADMIN_URL",
        "oracle+oracledb://system:sys_dev_pw@127.0.0.1:11521/?service_name=FREEPDB1",
    )
    url = make_url(admin)
    owner, table = "kb_engine_owner", "kb_engine_owner.employee_directory"
    return Engine(
        dialect="oracle",
        admin_url=admin,
        reader_url=url.set(username=READER, password=READER_PASSWORD).render_as_string(
            hide_password=False
        ),
        table=table,
        setup=[
            f"-DROP USER {READER} CASCADE",
            f"-DROP USER {owner} CASCADE",
            f'CREATE USER {owner} IDENTIFIED BY "Owner_dev_Passw0rd1" QUOTA UNLIMITED ON USERS',
            f"CREATE TABLE {table} (employee_id NUMBER(10) PRIMARY KEY, department VARCHAR2(50), "
            "job_title VARCHAR2(80), active NUMBER(1), hired DATE)",
            *_inserts(table, lambda r: f"{r[0]}, '{r[1]}', '{r[2]}', {int(r[3])}, DATE '{r[4]}'"),
            f'CREATE USER {READER} IDENTIFIED BY "{READER_PASSWORD}"',
            f"GRANT CREATE SESSION TO {READER}",
            f"GRANT SELECT ON {table} TO {READER}",
        ],
        heavy_sql="SELECT 1 AS employee_id, COUNT(*) AS n FROM all_objects a, all_objects b, "
        "all_objects c",
        active=1,  # NUMBER(1) column
        settle_s=5.0,
    )


class _PostgresEngine(Engine):
    """asyncpg is the only PostgreSQL driver installed, so admin work is async."""

    def _run(self, coro_fn):
        return asyncio.run(coro_fn())

    def run_admin(self, statements: list[str]) -> None:
        async def go():
            engine = create_async_engine(self.admin_url, isolation_level="AUTOCOMMIT")
            try:
                async with engine.connect() as conn:
                    for statement in statements:
                        try:
                            await conn.execute(text(statement.lstrip("-")))
                        except Exception:
                            if not statement.startswith("-"):
                                raise
            finally:
                await engine.dispose()

        self._run(go)

    def admin_count(self) -> int:
        async def go():
            engine = create_async_engine(self.admin_url)
            try:
                async with engine.connect() as conn:
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {self.table}"))
                    return int(result.scalar_one())
            finally:
                await engine.dispose()

        return self._run(go)


def _postgres() -> Engine:
    admin = _env("LIVE_POSTGRESQL_ADMIN_URL") or _env("DATABASE_URL")
    url = make_url(admin)
    table = "employee_directory"
    db_admin = url.set(database="kb_engine_test").render_as_string(hide_password=False)
    return _PostgresEngine(
        dialect="postgresql",
        admin_url=db_admin,
        reader_url=url.set(
            username=READER, password=READER_PASSWORD, database="kb_engine_test"
        ).render_as_string(hide_password=False),
        table=table,
        setup=[
            f"DROP TABLE IF EXISTS {table}",
            f"CREATE TABLE {table} (employee_id INT PRIMARY KEY, department TEXT, "
            "job_title TEXT, active BOOLEAN, hired DATE)",
            *_inserts(table, lambda r: f"{r[0]}, '{r[1]}', '{r[2]}', {r[3]}, '{r[4]}'"),
            f"GRANT SELECT ON {table} TO {READER}",
        ],
        heavy_sql="SELECT 1 AS employee_id, pg_sleep(3) AS slept",
    )


def _postgres_server_setup(engine: Engine) -> None:
    server = _PostgresEngine(
        "postgresql",
        make_url(engine.admin_url).set(database="postgres").render_as_string(hide_password=False),
        "", "", [], "",
    )  # fmt: skip
    server.run_admin(
        [
            "-CREATE DATABASE kb_engine_test",
            f"-CREATE ROLE {READER} LOGIN",
            f"ALTER ROLE {READER} LOGIN PASSWORD '{READER_PASSWORD}'",
        ]
    )


BUILDERS = {"postgresql": _postgres, "mysql": _mysql, "mssql": _mssql, "oracle": _oracle}
SERVER_SETUP = {"postgresql": _postgres_server_setup, "mssql": _mssql_server_setup}
WANTED = [
    d for d in (os.getenv("LIVE_DB_ENGINES") or ",".join(BUILDERS)).split(",") if d in BUILDERS
]


@pytest.fixture(scope="module", params=WANTED)
def engine(request) -> Engine:
    spec = BUILDERS[request.param]()
    try:
        if request.param in SERVER_SETUP:
            SERVER_SETUP[request.param](spec)
        spec.recreate()
    except Exception as exc:  # unreachable engine: skip it, not fail the run
        pytest.skip(f"{request.param} not available: {type(exc).__name__}: {exc}"[:300])
    return spec


def _config(engine: Engine, monkeypatch, *, sql: str | None = None, url: str | None = None,
            params: dict | None = None, **overrides) -> DatabaseSourceConfig:  # fmt: skip
    env = f"LIVE_SRC_URL_{uuid.uuid4().hex[:6].upper()}"
    monkeypatch.setenv(env, url or engine.reader_url)
    if sql is None:
        sql = (
            f"SELECT employee_id, department, job_title, hired FROM {engine.table} "
            "WHERE active = :active ORDER BY employee_id"
        )
        params = {"active": engine.active} if params is None else params
    return DatabaseSourceConfig.model_validate(
        {
            "name": f"live_{engine.dialect}",
            "type": "database",
            "dialect": engine.dialect,
            "url": f"${{{env}}}",
            "access_labels": ["role:authenticated"],
            "statement_timeout_ms": 5000,
            "queries": [
                {
                    "name": "employees",
                    "key_column": "employee_id",
                    "sql": sql,
                    "params": params or {},
                }
            ],
            **overrides,
        }  # fmt: skip
    )


def _unchecked(engine: Engine, monkeypatch, sql: str, url: str) -> DatabaseSourceConfig:
    """Skips config validation, to prove the database's own layer holds
    when a write somehow got past it."""
    env = f"LIVE_SRC_URL_{uuid.uuid4().hex[:6].upper()}"
    monkeypatch.setenv(env, url)
    query = DatabaseQueryConfig.model_construct(name="w", sql=sql, params={}, key_column="x")
    return DatabaseSourceConfig.model_construct(
        name=f"live_{engine.dialect}",
        type="database",
        dialect=engine.dialect,
        url=f"${{{env}}}",
        password=None,
        access_labels=frozenset({"role:authenticated"}),
        queries=[query],
        max_rows=10,
        statement_timeout_ms=5000,
        fetch_batch_size=500,
    )


async def test_rows_become_documents_with_bound_params(engine, monkeypatch):
    source = DatabaseSource(_config(engine, monkeypatch))
    discovered = await source.discover()
    assert [d.uri for d in discovered] == [
        f"db://live_{engine.dialect}/employees/1",
        f"db://live_{engine.dialect}/employees/2",
    ]  # row 3 is inactive
    doc = await source.fetch(source.resolve_access(discovered[0]))
    assert doc.title == "employees 1"
    text_ = doc.raw_text.lower()
    assert "department: hr" in text_ and "job_title: recruiter" in text_
    assert "hired: 2024-01-15" in text_  # DATE (Oracle: a datetime at midnight)


async def test_resync_skips_unchanged_updates_changed_and_retires_deleted(engine, monkeypatch):
    embedder = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=embedder)
    tenant = f"tenant-live-engine-{uuid.uuid4().hex[:10]}"
    source = DatabaseSource(_config(engine, monkeypatch))

    async def sync():
        return await sync_source(
            source, source_type="database", ingestion=service, tenant_id=tenant,
            assistant_id="kb_engine_live", embedder=embedder,
        )  # fmt: skip

    first = await sync()
    assert (first.status, first.created) == ("ok", 2), first.error
    second = await sync()
    assert (second.unchanged, second.chunks_embedded) == (2, 0)

    await asyncio.to_thread(
        engine.run_admin,
        [
            f"UPDATE {engine.table} SET job_title = 'Senior Engineer' WHERE employee_id = 2",
            f"DELETE FROM {engine.table} WHERE employee_id = 1",
        ],
    )
    try:
        third = await sync()
        assert (third.status, third.updated, third.unchanged) == ("ok", 1, 0)
        assert third.retired_items == [f"db://live_{engine.dialect}/employees/1"]
    finally:
        await asyncio.to_thread(engine.recreate)  # restore the table


async def test_the_select_only_login_cannot_write(engine, monkeypatch):
    before = await asyncio.to_thread(engine.admin_count)
    config = _unchecked(engine, monkeypatch, f"DELETE FROM {engine.table}", engine.reader_url)
    with pytest.raises(SourceError) as exc:
        await DatabaseSource(config).discover()
    assert f"source 'live_{engine.dialect}'" in str(exc.value)
    assert await asyncio.to_thread(engine.admin_count) == before


async def test_even_an_admin_login_cannot_write_through_a_source(engine, monkeypatch):
    """The admin CAN write, so this isolates the session-level protection:
    a READ ONLY transaction, or for SQL Server, the always-rolled-back one."""
    from app.ingestion.sources.dialects import DIALECTS

    before = await asyncio.to_thread(engine.admin_count)
    config = _unchecked(engine, monkeypatch, f"DELETE FROM {engine.table}", engine.admin_url)
    with pytest.raises(SourceError) as exc:
        await DatabaseSource(config).discover()
    if DIALECTS[engine.dialect].enforces_read_only:
        assert "tried to write and was refused" in str(exc.value)
    else:
        assert "returned no rows" in str(exc.value)  # ran, then rolled back
    assert await asyncio.to_thread(engine.admin_count) == before


async def test_statement_timeout(engine, monkeypatch):
    config = _config(engine, monkeypatch, sql=engine.heavy_sql, statement_timeout_ms=1000)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(SourceError, match="query 'employees' timed out"):
        await DatabaseSource(config).discover()
    assert loop.time() - started < 20


async def test_row_cap_and_empty_result(engine, monkeypatch):
    capped = _config(engine, monkeypatch, max_rows=1)
    with pytest.raises(SourceError, match="more than max_rows=1"):
        await DatabaseSource(capped).discover()
    none = _config(
        engine, monkeypatch,
        sql=f"SELECT employee_id FROM {engine.table} WHERE employee_id < :floor",
        params={"floor": 0},
    )  # fmt: skip
    assert await DatabaseSource(none).discover() == []


async def test_wrong_password_is_an_auth_failure_without_the_password(engine, monkeypatch):
    bad_password = "Wrong_dev_Passw0rd9"
    url = make_url(engine.reader_url).set(password=bad_password)
    config = _config(engine, monkeypatch, url=url.render_as_string(hide_password=False))
    with pytest.raises(SourceError) as exc:
        await DatabaseSource(config).discover()
    message = str(exc.value)
    assert "database authentication failed" in message, message
    assert bad_password not in message and READER not in message


async def test_unreachable_server_is_a_connect_failure(engine, monkeypatch):
    url = make_url(engine.reader_url).set(host="127.0.0.1", port=1)
    config = _config(engine, monkeypatch, url=url.render_as_string(hide_password=False))
    with pytest.raises(SourceError) as exc:
        await DatabaseSource(config).discover()
    assert "could not connect to the database" in str(exc.value)
    assert READER_PASSWORD not in str(exc.value)
