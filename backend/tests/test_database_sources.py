"""Database knowledge sources across engines, with the database mocked:
per-dialect config validation, URL/driver resolution, the engine-neutral
query runner against a fake connection, error classification and secret
redaction with each driver's real exception types, and sync behaviour
(change detection, retiring, labels, tenant isolation, a failing source
next to a working one) through the real pipeline and Postgres store.

Real engines are exercised in test_database_engines_live.py."""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from app.config.assistant_config import AssistantConfig
from app.config.knowledge_sources import DatabaseSourceConfig
from app.core.principal import PrincipalContext
from app.db.models import EMBEDDING_DIM
from app.ingestion.sources import DatabaseSource, SourceError
from app.ingestion.sources import database as db
from app.ingestion.sources.dialects import DIALECTS
from app.services.embedding import HashEmbeddingProvider
from app.services.ingestion_service import IngestionService
from app.services.retriever import RetrievalService
from app.services.source_sync import sync_knowledge_sources, sync_source

SECRET = "s3cr3t-Pa55word"

URLS = {
    "postgresql": f"postgresql+asyncpg://kb_reader:{SECRET}@pg.internal:5432/hr",
    "mysql": f"mysql+pymysql://kb_reader:{SECRET}@mysql.internal:3306/hr",
    "mssql": f"mssql+pymssql://kb_reader:{SECRET}@mssql.internal:1433/hr",
    "oracle": f"oracle+oracledb://kb_reader:{SECRET}@ora.internal:1521/?service_name=FREEPDB1",
}

# One idiomatic, valid query per engine.
VALID_SQL = {
    "postgresql": "SELECT employee_id, department::text FROM employees WHERE active = :active",
    "mysql": "SELECT `employee_id`, department FROM employees WHERE active = :active LIMIT 100",
    "mssql": "SELECT TOP 100 [employee_id], department FROM dbo.employees WITH (NOLOCK) "
    "WHERE active = :active",
    "oracle": "SELECT employee_id, department FROM employees WHERE active = :active "
    "FETCH FIRST 100 ROWS ONLY",
}


def _config(dialect: str = "postgresql", **overrides: Any) -> DatabaseSourceConfig:
    return DatabaseSourceConfig.model_validate(
        {
            "name": "employee_directory",
            "type": "database",
            "dialect": dialect,
            "url": "${HR_DATABASE_URL}",
            "access_labels": ["role:authenticated"],
            "max_rows": 1000,
            "statement_timeout_ms": 5000,
            "queries": [
                {
                    "name": "employee_records",
                    "key_column": "employee_id",
                    "sql": VALID_SQL.get(dialect, VALID_SQL["postgresql"]),
                    "params": {"active": True},
                }
            ],
            **overrides,
        }
    )


def _query(sql: str, **params: Any) -> dict[str, Any]:
    return {"name": "q", "key_column": "id", "sql": sql, "params": params}


# ---------- config validation, per dialect ----------


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_each_dialect_accepts_its_own_idiomatic_query(dialect):
    config = _config(dialect)
    assert config.dialect == dialect
    assert config.fetch_batch_size == 500 and config.password is None


def test_dialect_defaults_to_postgresql_for_existing_configs():
    config = DatabaseSourceConfig.model_validate(
        {
            "name": "office_db",
            "type": "database",
            "url": "${KB_DEMO_DB_URL}",
            "access_labels": ["scope:development"],
            "queries": [_query("SELECT id FROM t")],
        }
    )
    assert config.dialect == "postgresql"


def test_unknown_dialect_is_rejected_with_the_supported_list():
    with pytest.raises(ValueError) as exc:
        _config("sqlite")
    message = str(exc.value)
    assert "'postgresql', 'mysql', 'mssql' or 'oracle'" in message


@pytest.mark.parametrize(
    ("dialect", "sql", "message"),
    [
        # T-SQL needs no ';': sqlglot alone would read this as one SELECT.
        ("mssql", "SELECT id FROM t DELETE FROM t", "not allowed in a read query: DELETE"),
        ("mssql", "SELECT id INTO #copy FROM t", "INTO"),
        ("mssql", "SELECT id FROM t; EXEC xp_cmdshell 'dir'", "single statement"),
        ("mssql", "SELECT * FROM OPENROWSET('SQLNCLI', 'x', 'SELECT 1') AS r", "OPENROWSET"),
        ("mssql", "SELECT id FROM t WAITFOR DELAY '00:01'", "WAITFOR"),
        ("postgresql", "WITH d AS (DELETE FROM t RETURNING id) SELECT * FROM d", "DELETE"),
        ("postgresql", "SELECT id FROM t FOR UPDATE", "UPDATE"),
        ("postgresql", "SELECT * INTO copy_t FROM t", "INTO"),
        ("mysql", "SELECT id FROM t INTO OUTFILE '/tmp/x'", "INTO"),
        ("mysql", "WITH x AS (SELECT 1) SELECT * FROM t LOCK IN SHARE MODE", "LOCK"),
        ("oracle", "SELECT id FROM t FOR UPDATE NOWAIT", "UPDATE"),
        ("oracle", "WITH x AS (SELECT 1 FROM dual) SELECT id FROM x CONNECT BY ((((", "parsed"),
        # The dialect-independent first gate still applies.
        ("mysql", "SHOW TABLES", "single SELECT"),
        ("oracle", "SELECT 1 FROM dual; SELECT 2 FROM dual", "single statement"),
    ],
)
def test_write_or_unparseable_sql_is_rejected_for_its_dialect(dialect, sql, message):
    with pytest.raises(ValueError) as exc:
        _config(dialect, queries=[_query(sql)])
    assert message in str(exc.value)


def test_a_quoted_identifier_named_like_a_keyword_is_allowed():
    config = _config("postgresql", queries=[_query('SELECT id, "update", "comment" FROM t')])
    assert config.queries[0].sql.startswith("SELECT")
    config = _config("mssql", queries=[_query("SELECT id, [delete] FROM t")])
    assert config.queries[0].name == "q"


def test_keywords_inside_string_literals_and_column_names_are_fine():
    sql = "SELECT id, last_update, 'please do not DELETE' AS note FROM t WHERE kind = 'INSERT'"
    for dialect in DIALECTS:
        _config(dialect, queries=[_query(sql)])


@pytest.mark.parametrize(
    ("sql", "params", "message"),
    [
        ("SELECT id FROM t WHERE a = :a AND b = :b", {"a": 1}, "no params value for: b"),
        ("SELECT id FROM t", {"a": 1}, "params are not used by its sql: a"),
    ],
)
def test_params_must_match_the_sql_placeholders(sql, params, message):
    with pytest.raises(ValueError) as exc:
        _config("mysql", queries=[_query(sql, **params)])
    assert message in str(exc.value)


def test_postgres_casts_are_not_mistaken_for_placeholders():
    _config("postgresql", queries=[_query("SELECT id::text FROM t WHERE a = :a", a=1)])


@pytest.mark.parametrize(
    "overrides",
    [
        {"url": URLS["mysql"]},
        {"password": SECRET},
        {"password": "$HR_DB_PASSWORD"},
    ],
)
def test_credentials_in_yaml_are_rejected_without_echoing_them(overrides):
    with pytest.raises(ValueError) as exc:
        _config("mysql", **overrides)
    assert "literal secrets are not allowed" in str(exc.value)
    assert SECRET not in str(exc.value)


# ---------- resolving the URL against the dialect ----------


@pytest.mark.parametrize(
    ("dialect", "bare", "driver"),
    [
        ("postgresql", "postgresql://u:p@h/db", "asyncpg"),
        ("mysql", "mysql://u:p@h/db", "pymysql"),
        ("mssql", "mssql://u:p@h/db", "pymssql"),
        ("oracle", "oracle://u:p@h:1521/?service_name=X", "oracledb"),
    ],
)
def test_bare_scheme_gets_the_tested_driver(monkeypatch, dialect, bare, driver):
    monkeypatch.setenv("HR_DATABASE_URL", bare)
    connection = db.resolve_connection(_config(dialect))
    assert connection.url.drivername == f"{dialect}+{driver}"
    assert connection.is_async == (driver == "asyncpg")


@pytest.mark.parametrize(
    ("dialect", "url", "message"),
    [
        ("mysql", "mysql+aiomysql://u:p@h/db", "driver 'aiomysql' is not supported for mysql"),
        ("mssql", "mssql+pyodbc://u:p@dsn", "driver 'pyodbc' is not supported for mssql"),
        ("mysql", "mariadb+pymysql://u:p@h/db", "'mariadb' URL but the source's dialect"),
        (
            "oracle",
            "postgresql://u:p@h/db",
            "'postgresql' URL but the source's dialect is 'oracle'",
        ),
    ],
)
def test_unsupported_driver_or_mismatched_url_is_refused(monkeypatch, dialect, url, message):
    monkeypatch.setenv("HR_DATABASE_URL", url)
    with pytest.raises(SourceError, match=message):
        db.resolve_connection(_config(dialect))


def test_invalid_url_error_does_not_echo_the_url(monkeypatch):
    monkeypatch.setenv("HR_DATABASE_URL", f"not a url {SECRET}")
    with pytest.raises(SourceError) as exc:
        db.resolve_connection(_config("mysql"))
    assert "not a valid database URL" in str(exc.value)
    assert SECRET not in str(exc.value)


def test_separate_password_reference_is_merged_into_the_url(monkeypatch):
    monkeypatch.setenv("HR_DATABASE_URL", "mysql+pymysql://kb_reader@mysql.internal/hr")
    monkeypatch.setenv("HR_DB_PASSWORD", SECRET)
    connection = db.resolve_connection(_config("mysql", password="${HR_DB_PASSWORD}"))
    assert connection.url.password == SECRET
    assert SECRET not in repr(connection.url)  # SQLAlchemy masks it in repr/str


def test_redaction_removes_password_url_user_and_host():
    from sqlalchemy.engine import make_url

    resolved = db.ResolvedConnection(
        url=make_url(URLS["mysql"]), dialect=DIALECTS["mysql"], driver="pymysql"
    )
    message = (
        f"(1045, \"Access denied for user 'kb_reader'@'10.0.0.5' (using password: YES)\") "
        f"connecting to {URLS['mysql']} host mysql.internal pw={SECRET} enc=s3cr3t-Pa55word"
    )
    redacted = resolved.redact(message)
    for leaked in (SECRET, "kb_reader", "mysql.internal", URLS["mysql"]):
        assert leaked not in redacted
    assert "Access denied for user" in redacted


# ---------- the engine-neutral runner, on a fake connection ----------


class _FakeResult:
    def __init__(self, columns, rows, returns_rows=True):
        self._columns, self._rows, self.returns_rows = columns, list(rows), returns_rows
        self.fetch_sizes: list[int] = []
        self.closed = False

    def keys(self):
        return self._columns

    def fetchmany(self, size):
        self.fetch_sizes.append(size)
        batch, self._rows = self._rows[:size], self._rows[size:]
        return batch

    def close(self):
        self.closed = True


class _FakeConnection:
    """Records every statement; answers a query's SQL from ``results`` or
    raises the exception mapped to it in ``errors``."""

    def __init__(self, results=None, errors=None):
        self.results, self.errors = results or {}, errors or {}
        self.statements: list[tuple[str, Any]] = []
        self.options: list[dict] = []
        self.rolled_back = False
        self.last_result: _FakeResult | None = None

    def execution_options(self, **options):
        self.options.append(options)
        return self

    def execute(self, clause, params=None):
        sql = str(clause)
        self.statements.append((sql, params))
        for marker, exc in self.errors.items():
            if marker in sql:
                raise exc
        columns, rows = self.results.get(sql, ([], []))
        self.last_result = _FakeResult(columns, rows)
        return self.last_result

    def rollback(self):
        self.rolled_back = True

    def commit(self):  # pragma: no cover - must never happen
        raise AssertionError("a database source must never commit")


class _FakeEngine:
    def __init__(self, conn):
        self.conn, self.disposed = conn, False

    @contextmanager
    def connect(self):
        if isinstance(self.conn, BaseException):
            raise self.conn
        yield self.conn

    def dispose(self):
        self.disposed = True


@pytest.mark.parametrize(
    ("dialect", "expected_setup"),
    [
        (
            "postgresql",
            ["SET TRANSACTION READ ONLY", "SELECT set_config('statement_timeout', :ms, true)"],
        ),
        (
            "mysql",
            [
                "SET SESSION max_execution_time = 5000",
                "SET SESSION TRANSACTION READ ONLY",
                "START TRANSACTION READ ONLY",
            ],
        ),
        ("mssql", ["SET LOCK_TIMEOUT 5000"]),
        ("oracle", ["SET TRANSACTION READ ONLY"]),
    ],
)
def test_every_dialect_starts_read_only_runs_the_bound_query_and_rolls_back(
    dialect, expected_setup
):
    config = _config(dialect)
    sql = config.queries[0].sql
    conn = _FakeConnection(results={sql: (["employee_id", "department"], [(1, "HR"), (2, "IT")])})
    results = db.execute_queries(conn, config, DIALECTS[dialect], db._Progress())

    statements = [s for s, _ in conn.statements]
    assert statements[: len(expected_setup)] == expected_setup
    # The reviewed SQL runs unchanged, with values only as bound parameters.
    assert conn.statements[-1] == (sql, {"active": True})
    assert results == {"employee_records": (["employee_id", "department"], [(1, "HR"), (2, "IT")])}
    assert conn.rolled_back and conn.last_result.closed
    assert conn.options[-1] == {"stream_results": True, "max_row_buffer": 500}


def test_postgres_timeout_is_bound_not_formatted():
    conn = _FakeConnection()
    db.execute_queries(conn, _config("postgresql"), DIALECTS["postgresql"], db._Progress())
    assert conn.statements[1][1] == {"ms": "5000"}


def test_rows_are_fetched_in_batches_and_stop_one_past_max_rows():
    config = _config("mysql", max_rows=5, fetch_batch_size=2)
    sql = config.queries[0].sql
    conn = _FakeConnection(results={sql: (["employee_id"], [(i,) for i in range(50)])})
    columns, rows = db.execute_queries(conn, config, DIALECTS["mysql"], db._Progress())[
        "employee_records"
    ]
    assert len(rows) == 6  # max_rows + 1: enough to tell "over the cap"
    assert conn.last_result.fetch_sizes == [2, 2, 2]


def test_empty_result_is_an_empty_snapshot():
    config = _config("oracle")
    conn = _FakeConnection(results={config.queries[0].sql: (["employee_id"], [])})
    result = db.execute_queries(conn, config, DIALECTS["oracle"], db._Progress())
    assert result == {"employee_records": (["employee_id"], [])}


def test_a_statement_that_returns_no_rows_fails_the_source():
    config = _config("mssql")
    conn = _FakeConnection()
    conn.execute = lambda clause, params=None: _FakeResult([], [], returns_rows=False)
    with pytest.raises(SourceError, match="query 'employee_records' returned no rows"):
        db.execute_queries(conn, config, DIALECTS["mssql"], db._Progress())


# ---------- failures: real driver exception types, redacted ----------


def _driver_errors() -> dict[str, dict[str, BaseException]]:
    import asyncpg
    import oracledb
    import pymssql
    import pymysql

    def pg(cls, message):
        return cls(message)

    return {
        "postgresql": {
            "auth": pg(
                asyncpg.InvalidPasswordError, 'password authentication failed for user "kb_reader"'
            ),
            "timeout": pg(
                asyncpg.QueryCanceledError, "canceling statement due to statement timeout"
            ),
            "read_only": pg(
                asyncpg.ReadOnlySQLTransactionError,
                "cannot execute DELETE in a read-only transaction",
            ),
        },
        "mysql": {
            "auth": pymysql.err.OperationalError(
                1045,
                f"Access denied for user 'kb_reader'@'10.0.0.5' (using password: YES) {SECRET}",
            ),
            "timeout": pymysql.err.OperationalError(
                3024, "Query execution was interrupted, maximum statement execution time exceeded"
            ),
            "read_only": pymysql.err.OperationalError(
                1792, "Cannot execute statement in a READ ONLY transaction."
            ),
        },
        "mssql": {
            "auth": pymssql.OperationalError(
                18456, b"Login failed for user 'kb_reader'.DB-Lib error message 20018"
            ),
            "timeout": pymssql.OperationalError(20003, b"Adaptive Server connection timed out"),
        },
        "oracle": {
            "auth": oracledb.DatabaseError("ORA-01017: invalid credential or not authorized"),
            "timeout": oracledb.DatabaseError("DPY-4024: call timeout of 5000 ms exceeded"),
            "read_only": oracledb.DatabaseError(
                "ORA-01456: may not perform insert/delete/update operation inside a READ ONLY "
                "transaction"
            ),
        },
    }


def _wrapped(exc: BaseException) -> BaseException:
    """As SQLAlchemy raises it: a DBAPIError with the driver error as .orig."""
    from sqlalchemy.exc import DBAPIError

    return DBAPIError.instance("SELECT ...", {"active": True}, exc, Exception)


async def _run_with(monkeypatch, dialect, conn_or_exc, config=None, *, disposed=True):
    monkeypatch.setenv("HR_DATABASE_URL", URLS[dialect])
    config = config or _config(dialect)
    engine = _FakeEngine(conn_or_exc)
    monkeypatch.setattr(db, "create_source_engine", lambda connection, cfg: engine)
    # The async (asyncpg) path needs a real AsyncEngine; the fake engine
    # always takes the thread path, whose error handling is the same.
    try:
        return await db.run_read_only_queries(config, db.resolve_connection(config))
    finally:
        assert engine.disposed or not disposed


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
async def test_auth_failure_while_connecting_is_reported_without_secrets(monkeypatch, dialect):
    pytest.importorskip({"postgresql": "asyncpg", "mysql": "pymysql", "mssql": "pymssql",
                         "oracle": "oracledb"}[dialect])  # fmt: skip
    error = _wrapped(_driver_errors()[dialect]["auth"])
    with pytest.raises(SourceError) as exc:
        await _run_with(monkeypatch, dialect, error)
    message = str(exc.value)
    assert message.startswith("source 'employee_directory': database authentication failed")
    assert SECRET not in message and "kb_reader" not in message
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
async def test_unreachable_database_is_a_connect_failure(monkeypatch, dialect):
    refused = ConnectionRefusedError(10061, f"connect to {URLS[dialect]} refused")
    with pytest.raises(SourceError) as exc:
        await _run_with(monkeypatch, dialect, refused)
    assert "could not connect to the database" in str(exc.value)
    assert SECRET not in str(exc.value)


@pytest.mark.parametrize(
    ("dialect", "kind", "expected"),
    [
        (d, k, msg)
        for d, kinds in [
            ("postgresql", ("timeout", "read_only")),
            ("mysql", ("timeout", "read_only")),
            ("mssql", ("timeout",)),
            ("oracle", ("timeout", "read_only")),
        ]
        for k in kinds
        for msg in [{"timeout": "timed out", "read_only": "tried to write and was refused"}[k]]
    ],
)
async def test_query_errors_are_classified_per_engine(monkeypatch, dialect, kind, expected):
    config = _config(dialect)
    conn = _FakeConnection(errors={"employees": _wrapped(_driver_errors()[dialect][kind])})
    with pytest.raises(SourceError) as exc:
        await _run_with(monkeypatch, dialect, conn, config)
    assert f"query 'employee_records' {expected}" in str(exc.value)
    assert not conn.rolled_back  # failed before the end; NullPool discards the connection


async def test_sql_server_refused_connection_with_a_stale_login_failed_number(monkeypatch):
    # pymssql reuses the previous message number: a refused connection after
    # a failed login arrives as 18456 (seen live). The text decides.
    import pymssql

    refused = pymssql.OperationalError(
        (18456, b"DB-Lib error message 20009, severity 9:\nUnable to connect: Adaptive "
                b"Server is unavailable or does not exist (mssql.internal)\n")
    )  # fmt: skip
    with pytest.raises(SourceError) as exc:
        await _run_with(monkeypatch, "mssql", _wrapped(refused))
    assert str(exc.value).startswith(
        "source 'employee_directory': could not connect to the database: 18456 DB-Lib error"
    )
    assert "mssql.internal" not in str(exc.value)


async def test_unclassified_query_error_says_failed_and_keeps_the_driver_reason(monkeypatch):
    import pymysql

    conn = _FakeConnection(
        errors={
            "employees": _wrapped(
                pymysql.err.ProgrammingError(1146, "Table 'hr.employees' doesn't exist")
            )
        }
    )
    with pytest.raises(SourceError) as exc:
        await _run_with(monkeypatch, "mysql", conn)
    assert str(exc.value) == (
        "source 'employee_directory': query 'employee_records' failed: "
        "(1146, \"Table 'hr.employees' doesn't exist\")"
    )
    assert "SELECT" not in str(exc.value)  # SQLAlchemy's SQL/params echo is dropped


async def test_session_setup_failure_is_reported_as_such(monkeypatch):
    import pymysql

    conn = _FakeConnection(
        errors={
            "max_execution_time": _wrapped(
                pymysql.err.OperationalError(1193, "Unknown system variable 'max_execution_time'")
            )
        }
    )
    with pytest.raises(SourceError, match="could not start a read-only session: .*1193"):
        await _run_with(monkeypatch, "mysql", conn)


async def test_missing_driver_names_the_extra_to_install(monkeypatch):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["oracle"])
    config = _config("oracle")

    def no_driver(connection, cfg):
        raise ModuleNotFoundError("No module named 'oracledb'")

    monkeypatch.setattr(db, "create_source_engine", no_driver)
    with pytest.raises(SourceError, match=r"oracledb driver is not installed .*\[oracle\]"):
        await db.run_read_only_queries(config, db.resolve_connection(config))


async def test_overall_deadline_backs_up_a_hung_driver(monkeypatch):
    import time

    class _Hung(_FakeConnection):
        def execute(self, clause, params=None):
            if "employees" in str(clause):
                time.sleep(1.0)
            return super().execute(clause, params)

    # deadline = CONNECT_TIMEOUT_S + statement timeouts + 10s grace = 0.15s
    monkeypatch.setattr(db, "CONNECT_TIMEOUT_S", -9.85)
    config = _config("mssql", statement_timeout_ms=1)
    with pytest.raises(SourceError, match="query 'employee_records' timed out: no result within"):
        # The worker thread is still sleeping when the deadline fires.
        await _run_with(monkeypatch, "mssql", _Hung(), config, disposed=False)


def test_real_engines_are_built_without_a_pool_and_with_driver_timeouts(monkeypatch):
    from sqlalchemy.pool import NullPool

    for dialect in ("mysql", "mssql", "oracle"):
        pytest.importorskip({"mysql": "pymysql", "mssql": "pymssql", "oracle": "oracledb"}[dialect])
        monkeypatch.setenv("HR_DATABASE_URL", URLS[dialect])
        config = _config(dialect)
        engine = db.create_source_engine(db.resolve_connection(config), config)
        assert isinstance(engine.pool, NullPool)  # nothing outlives the sync
        engine.dispose()
    assert DIALECTS["mysql"].connect_args(5000) == {"connect_timeout": 10, "read_timeout": 10}
    assert DIALECTS["mssql"].connect_args(1500) == {"login_timeout": 10, "timeout": 2}
    assert DIALECTS["oracle"].connect_args(5000) == {"tcp_connect_timeout": 10.0}


def test_oracle_call_timeout_is_set_on_each_new_connection():
    class _Conn:
        call_timeout = 0

    conn = _Conn()
    DIALECTS["oracle"].on_connect(conn, 5000)
    assert conn.call_timeout == 5000


def test_only_sql_server_lacks_engine_enforced_read_only():
    assert {name for name, d in DIALECTS.items() if not d.enforces_read_only} == {"mssql"}


# ---------- DatabaseSource over any engine ----------


def _source(dialect, columns, rows, **overrides) -> DatabaseSource:
    async def runner(config, connection):
        assert connection.dialect is DIALECTS[dialect]
        return {"employee_records": (columns, rows)}

    return DatabaseSource(_config(dialect, **overrides), runner=runner)


async def test_key_column_matches_case_insensitively_when_unique(monkeypatch):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["oracle"])
    source = _source("oracle", ["EMPLOYEE_ID", "DEPARTMENT"], [(7, "HR")])
    [item] = await source.discover()
    assert item.uri == "db://employee_directory/employee_records/7"
    parsed = await source.fetch(source.resolve_access(item))
    assert parsed.raw_text == "EMPLOYEE_ID: 7\nDEPARTMENT: HR"


async def test_ambiguous_case_insensitive_key_is_refused(monkeypatch):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["mssql"])
    source = _source("mssql", ["Employee_Id", "EMPLOYEE_ID"], [(1, 1)])
    with pytest.raises(SourceError, match="no key_column 'employee_id'"):
        await source.discover()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1, "a"), (1, "b")], "duplicate employee_id '1'"),
        ([(None, "a")], "empty employee_id"),
        ([("  ", "a")], "empty employee_id"),
    ],
)
async def test_duplicate_or_missing_keys_fail_the_source(monkeypatch, rows, message):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["mysql"])
    with pytest.raises(SourceError, match=message):
        await _source("mysql", ["employee_id", "department"], rows).discover()


async def test_rows_render_the_same_for_every_engine(monkeypatch):
    from datetime import date, datetime, time
    from decimal import Decimal

    row = (Decimal("42"), date(2026, 1, 2), datetime(2026, 1, 2, 3, 4), time(9, 30), True, None)
    columns = ["employee_id", "start", "updated", "shift", "active", "notes"]
    texts = set()
    for dialect in DIALECTS:
        monkeypatch.setenv("HR_DATABASE_URL", URLS[dialect])
        source = _source(dialect, columns, [row])
        [item] = await source.discover()
        texts.add((item.uri, (await source.fetch(source.resolve_access(item))).raw_text))
    assert texts == {
        (
            "db://employee_directory/employee_records/42",
            "employee_id: 42\nstart: 2026-01-02\nupdated: 2026-01-02T03:04:00\n"
            "shift: 09:30:00\nactive: True",
        )
    }


# ---------- sync through the real pipeline and Postgres store ----------


class _Rows:
    """A mutable table behind a fake runner; ``error`` makes the next sync fail."""

    def __init__(self, rows):
        self.rows = rows
        self.error: SourceError | None = None

    async def __call__(self, config, connection):
        if self.error:
            raise self.error
        return {"employee_records": (["employee_id", "department"], list(self.rows))}


@pytest.fixture
def store():
    embedder = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = IngestionService(embedder=embedder)

    async def sync(source, tenant, assistant="kb_db_test"):
        return await sync_source(
            source,
            source_type="database",
            ingestion=service,
            tenant_id=tenant,
            assistant_id=assistant,
            embedder=embedder,
        )

    async def titles(tenant, labels, assistant="kb_db_test"):
        retriever = RetrievalService(embedder=embedder, vector_store=service.vector_store)
        principal = PrincipalContext(tenant_id=tenant, principal_id="u", labels=labels, clearance=1)
        chunks = await retriever.search(
            query="department",
            assistant_id=assistant,
            principal=principal,
            top_k=20,
            min_similarity=-1.0,
        )
        return sorted({c.document_title for c in chunks})

    store = type("Store", (), {})()
    store.sync, store.titles, store.service = sync, titles, service
    return store


def _tenant() -> str:
    return f"tenant-dbsrc-{uuid.uuid4().hex[:10]}"


async def test_sync_detects_changes_skips_unchanged_and_retires_removed_rows(monkeypatch, store):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["mysql"])
    table = _Rows([(1, "HR"), (2, "IT"), (3, "Finance")])
    source = DatabaseSource(_config("mysql"), runner=table)
    tenant = _tenant()

    first = await store.sync(source, tenant)
    assert (first.status, first.created, first.discovered) == ("ok", 3, 3)

    again = await store.sync(source, tenant)
    assert (again.created, again.updated, again.unchanged, again.chunks_embedded) == (0, 0, 3, 0)

    table.rows = [(1, "HR"), (2, "Engineering")]  # row 2 changed, row 3 deleted
    changed = await store.sync(source, tenant)
    assert (changed.updated, changed.unchanged) == (1, 1)
    assert changed.retired_items == ["db://employee_directory/employee_records/3"]
    labels = frozenset({"role:authenticated"})
    assert await store.titles(tenant, labels) == ["employee_records 1", "employee_records 2"]


async def test_failed_query_retires_nothing_but_an_empty_result_retires_all(monkeypatch, store):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["mssql"])
    table = _Rows([(1, "HR"), (2, "IT")])
    source = DatabaseSource(_config("mssql"), runner=table)
    tenant = _tenant()
    labels = frozenset({"role:authenticated"})
    await store.sync(source, tenant)

    table.error = SourceError("source 'employee_directory': query 'employee_records' timed out: x")
    failed = await store.sync(source, tenant)
    assert (failed.status, failed.retired) == ("failed", 0)
    assert await store.titles(tenant, labels) == ["employee_records 1", "employee_records 2"]

    # A query that succeeds with no rows is the source saying "nothing".
    table.error, table.rows = None, []
    emptied = await store.sync(source, tenant)
    assert (emptied.status, emptied.retired) == ("ok", 2)
    assert await store.titles(tenant, labels) == []


async def test_access_labels_from_config_gate_retrieval(monkeypatch, store):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["oracle"])
    source = DatabaseSource(_config("oracle", access_labels=["dept:hr"]), runner=_Rows([(1, "HR")]))
    tenant = _tenant()
    await store.sync(source, tenant)
    assert await store.titles(tenant, frozenset({"role:authenticated"})) == []
    assert await store.titles(tenant, frozenset({"dept:hr"})) == ["employee_records 1"]


async def test_tenants_and_assistants_are_isolated(monkeypatch, store):
    monkeypatch.setenv("HR_DATABASE_URL", URLS["postgresql"])
    labels = frozenset({"role:authenticated"})
    tenant_a, tenant_b = _tenant(), _tenant()
    table_a, table_b = _Rows([(1, "HR")]), _Rows([(2, "IT")])
    await store.sync(DatabaseSource(_config("postgresql"), runner=table_a), tenant_a)
    await store.sync(DatabaseSource(_config("postgresql"), runner=table_b), tenant_b)
    await store.sync(
        DatabaseSource(_config("postgresql"), runner=_Rows([(3, "Ops")])), tenant_a, "other_bot"
    )
    assert await store.titles(tenant_a, labels) == ["employee_records 1"]
    assert await store.titles(tenant_b, labels) == ["employee_records 2"]
    assert await store.titles(tenant_a, labels, "other_bot") == ["employee_records 3"]

    # Emptying tenant A's table retires only tenant A's rows for this assistant.
    table_a.rows = []
    result = await store.sync(DatabaseSource(_config("postgresql"), runner=table_a), tenant_a)
    assert result.retired_items == ["db://employee_directory/employee_records/1"]
    assert await store.titles(tenant_b, labels) == ["employee_records 2"]
    assert await store.titles(tenant_a, labels, "other_bot") == ["employee_records 3"]


async def test_one_failing_database_does_not_stop_the_other_sources(
    monkeypatch, store, tmp_path: Path, caplog
):
    """A real assistant config: a filesystem source, a MySQL source whose
    server rejects the login, and a working Oracle source (fake engine).
    The failure is reported for its source only, without the password, and
    retires nothing of what that source ingested earlier."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "handbook.md").write_text("The IT department handbook.", encoding="utf-8")
    monkeypatch.setenv("HR_MYSQL_URL", URLS["mysql"])
    monkeypatch.setenv("HR_ORACLE_URL", URLS["oracle"])

    def source(name, dialect, env):
        return {
            "name": name,
            "type": "database",
            "dialect": dialect,
            "url": f"${{{env}}}",
            "access_labels": ["role:authenticated"],
            "queries": [
                {"name": "employee_records", "key_column": "employee_id", "sql": VALID_SQL[dialect],
                 "params": {"active": True}}
            ],
        }  # fmt: skip

    config = AssistantConfig.model_validate(
        {
            "assistant_id": "kb_db_multi",
            "display_name": "x",
            "description": "x",
            "tenant_id": "t",
            "model": {"provider": "azure_ai", "model_name": "m"},
            "system_prompt": "x",
            "knowledge_sources": [
                {
                    "name": "docs",
                    "type": "filesystem",
                    "path": str(docs),
                    "access_labels": ["role:authenticated"],
                },
                source("hr_mysql", "mysql", "HR_MYSQL_URL"),
                source("hr_oracle", "oracle", "HR_ORACLE_URL"),
            ],
        }  # fmt: skip
    )
    login_ok = {"mysql": True}

    def engine_for(connection, cfg):
        if connection.dialect.name == "mysql" and not login_ok["mysql"]:
            return _FakeEngine(_wrapped(_driver_errors()["mysql"]["auth"]))
        sql = cfg.queries[0].sql
        return _FakeEngine(_FakeConnection(results={sql: (["employee_id"], [(cfg.name,)])}))

    monkeypatch.setattr(db, "create_source_engine", engine_for)
    tenant = _tenant()

    async def run():
        return await sync_knowledge_sources(
            config,
            tenant_id=tenant,
            ingestion=store.service,
            embedder=HashEmbeddingProvider(dim=EMBEDDING_DIM),
            base_dir=tmp_path,
        )

    first = await run()
    assert [(r.name, r.status, r.created) for r in first] == [
        ("docs", "ok", 1), ("hr_mysql", "ok", 1), ("hr_oracle", "ok", 1)
    ]  # fmt: skip

    login_ok["mysql"] = False
    with caplog.at_level(logging.INFO):
        second = await run()
    by_name = {r.name: r for r in second}
    assert (by_name["docs"].status, by_name["hr_oracle"].status) == ("ok", "ok")
    assert by_name["hr_oracle"].unchanged == 1
    failed = by_name["hr_mysql"]
    assert (failed.status, failed.retired) == ("failed", 0)
    assert failed.error.startswith("source 'hr_mysql': database authentication failed")
    assert SECRET not in failed.error and SECRET not in caplog.text
    assert "kb_reader" not in failed.error
    titles = await store.titles(tenant, frozenset({"role:authenticated"}), "kb_db_multi")
    assert "employee_records hr_mysql" in titles  # still searchable


def test_the_example_config_validates():
    from app.config.loader import load_assistant_config

    path = Path(__file__).resolve().parent.parent / "configs/examples/database_sources.yaml"
    config = load_assistant_config(path)
    assert [(s.name, s.dialect) for s in config.knowledge_sources] == [
        ("pg_employees", "postgresql"),
        ("mysql_employees", "mysql"),
        ("mssql_employees", "mssql"),
        ("oracle_employees", "oracle"),
    ]
