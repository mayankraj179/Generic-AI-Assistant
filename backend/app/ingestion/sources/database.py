from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from urllib.parse import quote, quote_plus

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL, Connection, Engine, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.config.knowledge_sources import DatabaseQueryConfig, DatabaseSourceConfig
from app.ingestion.pipeline import (
    DiscoveredSource,
    ParsedDocument,
    SourceWithAccess,
    parsed_document,
)
from app.ingestion.sources.base import KnowledgeSource, SourceError, resolve_env_ref
from app.ingestion.sources.dialects import (
    CONNECT_TIMEOUT_S,
    DIALECTS,
    DatabaseDialect,
    FailureKind,
)

QueryResult = tuple[Sequence[str], Sequence[Sequence[Any]]]


@dataclass(frozen=True)
class ResolvedConnection:
    """A source's connection, checked against its dialect. ``url`` holds the
    password: it never goes into a message (use ``redact``)."""

    url: URL
    dialect: DatabaseDialect
    driver: str

    @property
    def is_async(self) -> bool:
        return self.driver in self.dialect.async_drivers

    def redact(self, message: str) -> str:
        """Removes the password, the URL itself, and the user and host names
        from a driver message before it reaches a log or a response."""
        url = self.url
        secrets = {url.render_as_string(hide_password=False), str(url)}
        if url.password:
            secrets |= {str(url.password), quote_plus(str(url.password)), quote(str(url.password))}
        for secret in sorted((s for s in secrets if s), key=len, reverse=True):
            message = message.replace(secret, "***")
        for name in (url.username, url.host):
            if name and len(name) >= 3:
                message = re.sub(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", "***", message)
        return message


# Runs the configured queries and returns {query name: (columns, rows)}.
QueryRunner = Callable[
    [DatabaseSourceConfig, ResolvedConnection], Awaitable[dict[str, QueryResult]]
]


def resolve_connection(config: DatabaseSourceConfig) -> ResolvedConnection:
    """Resolves the url (and optional password) env refs and checks the URL
    is for the configured dialect with an allowed driver."""
    dialect = DIALECTS[config.dialect]
    raw = resolve_env_ref(config.url, field="url", source=config.name)
    try:
        url = make_url(raw)
    except ArgumentError:
        # SQLAlchemy's message quotes the whole string, password included.
        raise SourceError(f"source '{config.name}': url is not a valid database URL") from None
    backend, _, driver = url.drivername.partition("+")
    if backend != dialect.name:
        raise SourceError(
            f"source '{config.name}': url is a '{backend}' URL but the source's "
            f"dialect is '{dialect.name}'"
        )
    driver = driver or dialect.drivers[0]
    if driver not in dialect.drivers:
        raise SourceError(
            f"source '{config.name}': driver '{driver}' is not supported for "
            f"{dialect.name} (supported: {', '.join(dialect.drivers)})"
        )
    url = url.set(drivername=f"{backend}+{driver}")
    if config.password is not None:
        url = url.set(
            password=resolve_env_ref(config.password, field="password", source=config.name)
        )
    return ResolvedConnection(url=url, dialect=dialect, driver=driver)


def create_source_engine(
    connection: ResolvedConnection, config: DatabaseSourceConfig
) -> Engine | AsyncEngine:
    """A pool-less engine for one sync. Module-level so tests can swap it."""
    dialect, timeout_ms = connection.dialect, config.statement_timeout_ms
    kwargs: dict[str, Any] = {
        "poolclass": NullPool,
        "connect_args": dialect.connect_args(timeout_ms),
    }
    engine: Engine | AsyncEngine
    if connection.is_async:
        engine = create_async_engine(connection.url, **kwargs)
        sync_engine = engine.sync_engine
    else:
        engine = sync_engine = create_engine(connection.url, **kwargs)

    @event.listens_for(sync_engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        dialect.on_connect(dbapi_connection, timeout_ms)

    return engine


_SETUP_STEPS = frozenset({"connect", "session", "rollback"})


class _Progress:
    """Which step a failure happened in: one of _SETUP_STEPS or a query name."""

    step = "connect"


def execute_queries(
    conn: Connection,
    config: DatabaseSourceConfig,
    dialect: DatabaseDialect,
    progress: _Progress,
) -> dict[str, QueryResult]:
    """Runs each reviewed query in one read-only transaction (as far as the
    engine supports one), fetching at most max_rows + 1 rows in batches (one
    extra row tells "exactly at the cap" from "over it"), then rolls back.
    Nothing is ever committed. The same code runs for every engine: an
    async driver gets here through AsyncConnection.run_sync."""
    progress.step = "session"
    dialect.begin_read_only(conn, config.statement_timeout_ms)
    results: dict[str, QueryResult] = {}
    limit = config.max_rows + 1
    for query in config.queries:
        progress.step = query.name
        result = conn.execution_options(
            stream_results=True, max_row_buffer=config.fetch_batch_size
        ).execute(text(query.sql), query.params)
        try:
            if not result.returns_rows:
                raise SourceError(f"source '{config.name}': query '{query.name}' returned no rows")
            columns = list(result.keys())
            rows: list[tuple[Any, ...]] = []
            while len(rows) < limit:
                batch = result.fetchmany(min(config.fetch_batch_size, limit - len(rows)))
                if not batch:
                    break
                rows.extend(tuple(row) for row in batch)
        finally:
            result.close()
        results[query.name] = (columns, rows)
    progress.step = "rollback"
    conn.rollback()
    return results


def _failure(
    config: DatabaseSourceConfig,
    connection: ResolvedConnection,
    exc: BaseException,
    step: str,
) -> SourceError:
    dialect = connection.dialect
    kind = dialect.classify(exc)
    if isinstance(exc, ImportError):
        extra = f"[{dialect.extra}]" if dialect.extra else ""
        return SourceError(
            f"source '{config.name}': the {connection.driver} driver is not installed "
            f"(pip install 'generic-ai-assistant-framework{extra}')"
        )
    if step == "connect":
        kind = FailureKind.AUTH if kind is FailureKind.AUTH else FailureKind.CONNECT
    elif step == "session" and kind is None:
        kind = FailureKind.SESSION
    elif kind in (None, FailureKind.AUTH):
        kind = FailureKind.QUERY
    reason = " ".join(connection.redact(dialect.describe(exc)).split())[:500]
    reason = reason or type(exc).__name__  # e.g. a bare TimeoutError while connecting
    if kind in (FailureKind.CONNECT, FailureKind.AUTH, FailureKind.SESSION):
        return SourceError(f"source '{config.name}': {kind}: {reason}")
    return SourceError(f"source '{config.name}': query '{step}' {kind}: {reason}")


async def run_read_only_queries(
    config: DatabaseSourceConfig, connection: ResolvedConnection
) -> dict[str, QueryResult]:
    """Connects, runs execute_queries, and turns every driver or SQLAlchemy
    error into a SourceError with the credentials redacted. The driver's
    own timeouts are the primary limit; an overall deadline backs them up
    so a hung connection can't hold a sync forever."""
    progress = _Progress()
    deadline_s = CONNECT_TIMEOUT_S + len(config.queries) * config.statement_timeout_ms / 1000 + 10

    async def run_async(engine: AsyncEngine) -> dict[str, QueryResult]:
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(execute_queries, config, connection.dialect, progress)
        finally:
            await engine.dispose()

    def run_sync(engine: Engine) -> dict[str, QueryResult]:
        try:
            with engine.connect() as conn:
                return execute_queries(conn, config, connection.dialect, progress)
        finally:
            engine.dispose()

    deadline = asyncio.timeout(deadline_s)
    try:
        async with deadline:
            engine = create_source_engine(connection, config)
            if isinstance(engine, AsyncEngine):
                return await run_async(engine)
            return await asyncio.to_thread(run_sync, engine)
    except SourceError:
        raise
    # Any driver, SQLAlchemy or socket error. "from None": the original's
    # traceback can carry the URL or SQL parameters into a log.
    except Exception as exc:
        if deadline.expired():
            step = progress.step
            where = step if step in _SETUP_STEPS else f"query '{step}'"
            raise SourceError(
                f"source '{config.name}': {where} {FailureKind.TIMEOUT}: "
                f"no result within {deadline_s:.0f}s"
            ) from None
        raise _failure(config, connection, exc, progress.step) from None


def _format(value: Any) -> str:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    return str(value)


def render_row(columns: Sequence[str], row: Sequence[Any]) -> str:
    """One ``column: value`` line per non-null column."""
    return "\n".join(
        f"{column}: {_format(value)}"
        for column, value in zip(columns, row, strict=True)
        if value is not None
    )


class DatabaseSource(KnowledgeSource):
    """A snapshot of reviewed, parameterised SELECTs, one document per row,
    from any engine in dialects.DIALECTS.

    discover() runs the queries once and keeps the rendered rows; fetch()
    reads from that snapshot, so a sync sees one consistent result set.
    A row's URI is db://<name>/<query>/<key value> whatever the engine, so
    an unchanged row is skipped on re-sync and a deleted row is retired.
    """

    source_type = "db"

    def __init__(
        self, config: DatabaseSourceConfig, *, runner: QueryRunner = run_read_only_queries
    ) -> None:
        super().__init__(config.name, config.access_labels)
        self.config = config
        self._runner = runner
        self._snapshot: dict[str, tuple[str, str]] = {}

    def _key_index(self, query: DatabaseQueryConfig, columns: Sequence[str]) -> int:
        names = list(columns)
        if query.key_column in names:
            return names.index(query.key_column)
        # Engines differ in the case they report unquoted names in (Oracle:
        # upper; SQLAlchemy lowers it again), so fall back to a unique
        # case-insensitive match.
        folded = [i for i, c in enumerate(names) if c.lower() == query.key_column.lower()]
        if len(folded) == 1:
            return folded[0]
        raise SourceError(
            f"source '{self.name}': query '{query.name}' has no key_column "
            f"'{query.key_column}' (columns: {', '.join(names)})"
        )

    def _documents(
        self, query: DatabaseQueryConfig, result: QueryResult
    ) -> dict[str, tuple[str, str]]:
        columns, rows = result
        if len(rows) > self.config.max_rows:
            raise SourceError(
                f"source '{self.name}': query '{query.name}' returned more than "
                f"max_rows={self.config.max_rows} rows"
            )
        key_index = self._key_index(query, columns)
        key_name = columns[key_index]
        documents: dict[str, tuple[str, str]] = {}
        for row in rows:
            key = row[key_index]
            if key is None or str(key).strip() == "":
                raise SourceError(
                    f"source '{self.name}': query '{query.name}' returned a row "
                    f"with an empty {key_name}"
                )
            uri = self.uri_for(f"{query.name}/{quote(str(key), safe='')}")
            if uri in documents:
                raise SourceError(
                    f"source '{self.name}': query '{query.name}' returned duplicate "
                    f"{key_name} '{key}'"
                )
            documents[uri] = (f"{query.name} {key}", render_row(columns, row))
        return documents

    async def discover(self) -> list[DiscoveredSource]:
        connection = resolve_connection(self.config)
        results = await self._runner(self.config, connection)
        snapshot: dict[str, tuple[str, str]] = {}
        for query in self.config.queries:
            snapshot.update(self._documents(query, results[query.name]))
        self._snapshot = snapshot
        return [DiscoveredSource(uri=uri, source_type="database") for uri in snapshot]

    async def fetch(self, source: SourceWithAccess) -> ParsedDocument:
        try:
            title, body = self._snapshot[source.source.uri]
        except KeyError:
            raise FileNotFoundError(
                f"row is not in this sync's snapshot: {source.source.uri}"
            ) from None
        return parsed_document(source, title=title, raw_text=body)
