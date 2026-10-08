from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime
from typing import Any
from urllib.parse import quote

import asyncpg
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.config.knowledge_sources import DatabaseQueryConfig, DatabaseSourceConfig
from app.ingestion.pipeline import (
    DiscoveredSource,
    ParsedDocument,
    SourceWithAccess,
    parsed_document,
)
from app.ingestion.sources.base import KnowledgeSource, SourceError, resolve_env_ref

QueryResult = tuple[Sequence[str], Sequence[Sequence[Any]]]
# Runs the configured queries and returns {query name: (columns, rows)}.
QueryRunner = Callable[[DatabaseSourceConfig, str], Awaitable[dict[str, QueryResult]]]


async def run_read_only_queries(config: DatabaseSourceConfig, url: str) -> dict[str, QueryResult]:
    """Runs each reviewed query inside one READ ONLY transaction with a
    statement timeout, fetching at most max_rows + 1 rows (one extra, to
    tell "exactly at the cap" from "over it"). PostgreSQL only."""
    if url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    if not url.startswith("postgresql+asyncpg://"):
        raise SourceError(f"source '{config.name}': only PostgreSQL databases are supported")

    engine = create_async_engine(url, poolclass=NullPool, connect_args={"timeout": 10})
    results: dict[str, QueryResult] = {}
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            await conn.execute(
                text("SELECT set_config('statement_timeout', :ms, true)"),
                {"ms": str(config.statement_timeout_ms)},
            )
            for query in config.queries:
                result = await conn.stream(text(query.sql), query.params)
                rows = await result.fetchmany(config.max_rows + 1)
                columns = list(result.keys())
                await result.close()
                results[query.name] = (columns, [tuple(row) for row in rows])
            await conn.rollback()
    # Streamed (server-side cursor) results can raise the driver's own
    # exceptions unwrapped, e.g. a statement timeout or a write refused by
    # the READ ONLY transaction, so catch asyncpg's as well as SQLAlchemy's.
    except (SQLAlchemyError, asyncpg.PostgresError, asyncpg.InterfaceError) as exc:
        # The driver's own message (e.g. "canceling statement due to
        # statement timeout"), not SQLAlchemy's wrapper with the SQL echoed.
        reason = getattr(exc, "orig", None) or exc
        raise SourceError(f"source '{config.name}': query failed: {reason}") from exc
    except OSError as exc:
        raise SourceError(f"source '{config.name}': could not connect: {exc}") from exc
    finally:
        await engine.dispose()
    return results


def _format(value: Any) -> str:
    if isinstance(value, datetime | date):
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
    """A snapshot of reviewed, parameterised SELECTs, one document per row.

    discover() runs the queries once and keeps the rendered rows; fetch()
    reads from that snapshot, so a sync sees one consistent result set.
    A row's URI is db://<name>/<query>/<key value>, so an unchanged row is
    skipped on re-sync and a deleted row is retired.
    """

    source_type = "db"

    def __init__(
        self, config: DatabaseSourceConfig, *, runner: QueryRunner = run_read_only_queries
    ) -> None:
        super().__init__(config.name, config.access_labels)
        self.config = config
        self._runner = runner
        self._snapshot: dict[str, tuple[str, str]] = {}

    def _documents(
        self, query: DatabaseQueryConfig, result: QueryResult
    ) -> dict[str, tuple[str, str]]:
        columns, rows = result
        if len(rows) > self.config.max_rows:
            raise SourceError(
                f"source '{self.name}': query '{query.name}' returned more than "
                f"max_rows={self.config.max_rows} rows"
            )
        if query.key_column not in columns:
            raise SourceError(
                f"source '{self.name}': query '{query.name}' has no key_column "
                f"'{query.key_column}' (columns: {', '.join(columns)})"
            )
        key_index = list(columns).index(query.key_column)
        documents: dict[str, tuple[str, str]] = {}
        for row in rows:
            key = row[key_index]
            if key is None or str(key).strip() == "":
                raise SourceError(
                    f"source '{self.name}': query '{query.name}' returned a row "
                    f"with an empty {query.key_column}"
                )
            uri = self.uri_for(f"{query.name}/{quote(str(key), safe='')}")
            if uri in documents:
                raise SourceError(
                    f"source '{self.name}': query '{query.name}' returned duplicate "
                    f"{query.key_column} '{key}'"
                )
            documents[uri] = (f"{query.name} {key}", render_row(columns, row))
        return documents

    async def discover(self) -> list[DiscoveredSource]:
        url = resolve_env_ref(self.config.url, field="url", source=self.name)
        results = await self._runner(self.config, url)
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
