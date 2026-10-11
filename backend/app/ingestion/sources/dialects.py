"""The engine-specific half of a database knowledge source.

DatabaseSource (database.py) owns everything engine-neutral: resolving the
URL, running the reviewed queries through SQLAlchemy, the row cap, turning
rows into documents and keeping secrets out of errors. A dialect supplies
only what genuinely differs per engine, all through SQLAlchemy's own
dialect/driver support (no protocol code here):

- which drivers are allowed (only the ones this framework has tested),
- connect arguments: connect timeout and any client-side query timeout,
- how a session is made read-only and given its statement timeout,
- how the driver's errors map to auth / timeout / read-only refusals.

Adding an engine means adding a subclass here, registering it in DIALECTS,
adding its name to DatabaseDialectName (config/knowledge_sources.py) and
sqlglot's name to SQLGLOT_DIALECTS (config/sql_safety.py), plus its driver
to an optional extra in pyproject.toml.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, ClassVar

from sqlalchemy import text
from sqlalchemy.engine import Connection

CONNECT_TIMEOUT_S = 10


class FailureKind(StrEnum):
    CONNECT = "could not connect to the database"
    AUTH = "database authentication failed"
    SESSION = "could not start a read-only session"
    TIMEOUT = "timed out"
    READ_ONLY = "tried to write and was refused"
    QUERY = "failed"


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """exc, then what it wraps: SQLAlchemy's ``.orig``, then ``__cause__``
    and ``__context__``, so a driver error is found however it's wrapped."""
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack and len(seen) < 10:
        current = stack.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.extend([getattr(current, "orig", None), current.__cause__, current.__context__])


def _driver_exception(exc: BaseException) -> BaseException:
    """The innermost non-SQLAlchemy exception: the driver's own message
    rather than SQLAlchemy's wrapper, which echoes the SQL and params."""
    for candidate in exception_chain(exc):
        if not type(candidate).__module__.startswith("sqlalchemy"):
            return candidate
    return exc


class DatabaseDialect:
    name: ClassVar[str]
    # Allowed SQLAlchemy drivers; the first is used for a bare scheme
    # ("mysql://..." means "mysql+pymysql://...").
    drivers: ClassVar[tuple[str, ...]]
    async_drivers: ClassVar[frozenset[str]] = frozenset()
    # pip extra that installs the drivers (None: a core dependency).
    extra: ClassVar[str | None] = None
    # Whether the engine itself refuses writes for the whole session.
    enforces_read_only: ClassVar[bool] = True

    def connect_args(self, timeout_ms: int) -> dict[str, Any]:
        return {}

    def on_connect(self, dbapi_connection: Any, timeout_ms: int) -> None:
        """Runs once per new DBAPI connection, before anything else."""

    def begin_read_only(self, conn: Connection, timeout_ms: int) -> None:
        raise NotImplementedError

    def classify(self, exc: BaseException) -> FailureKind | None:
        """AUTH / TIMEOUT / READ_ONLY when recognisable, else None."""
        return None

    def describe(self, exc: BaseException) -> str:
        return str(_driver_exception(exc))


class PostgreSQLDialect(DatabaseDialect):
    """READ ONLY transaction plus a server-side statement_timeout, both
    scoped to the one transaction the sync runs in."""

    name = "postgresql"
    drivers = ("asyncpg",)
    async_drivers = frozenset({"asyncpg"})

    _CODES: ClassVar[dict[str, FailureKind]] = {
        "28P01": FailureKind.AUTH,  # invalid_password
        "28000": FailureKind.AUTH,  # invalid_authorization_specification
        "57014": FailureKind.TIMEOUT,  # query_canceled (statement_timeout)
        "25006": FailureKind.READ_ONLY,  # read_only_sql_transaction
    }

    def connect_args(self, timeout_ms: int) -> dict[str, Any]:
        return {"timeout": CONNECT_TIMEOUT_S}

    def begin_read_only(self, conn: Connection, timeout_ms: int) -> None:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(
            text("SELECT set_config('statement_timeout', :ms, true)"), {"ms": str(timeout_ms)}
        )

    def classify(self, exc: BaseException) -> FailureKind | None:
        for candidate in exception_chain(exc):
            code = getattr(candidate, "sqlstate", None) or getattr(candidate, "pgcode", None)
            if code in self._CODES:
                return self._CODES[code]
        return None


class MySQLDialect(DatabaseDialect):
    """START TRANSACTION READ ONLY plus max_execution_time. MySQL applies
    max_execution_time to SELECT statements only (fine: only a SELECT gets
    this far), and READ ONLY still lets a session write TEMPORARY tables.
    pymysql's read_timeout is a client-side backstop. MariaDB is not
    supported: it names the timeout variable differently."""

    name = "mysql"
    drivers = ("pymysql",)
    extra = "mysql"

    _CODES: ClassVar[dict[int, FailureKind]] = {
        1044: FailureKind.AUTH,  # access denied to the database
        1045: FailureKind.AUTH,  # access denied for user
        1698: FailureKind.AUTH,
        3024: FailureKind.TIMEOUT,  # max_execution_time exceeded
        1792: FailureKind.READ_ONLY,  # can't execute in a READ ONLY transaction
    }

    def connect_args(self, timeout_ms: int) -> dict[str, Any]:
        return {
            "connect_timeout": CONNECT_TIMEOUT_S,
            "read_timeout": math.ceil(timeout_ms / 1000) + 5,
        }

    def begin_read_only(self, conn: Connection, timeout_ms: int) -> None:
        conn.execute(text(f"SET SESSION max_execution_time = {int(timeout_ms)}"))
        conn.execute(text("SET SESSION TRANSACTION READ ONLY"))
        conn.execute(text("START TRANSACTION READ ONLY"))

    def classify(self, exc: BaseException) -> FailureKind | None:
        for candidate in exception_chain(exc):
            args = getattr(candidate, "args", ())
            if args and isinstance(args[0], int):
                if args[0] in self._CODES:
                    return self._CODES[args[0]]
                if args[0] == 2013 and "timed out" in str(candidate):
                    return FailureKind.TIMEOUT  # pymysql's read_timeout
        return None


class SQLServerDialect(DatabaseDialect):
    """SQL Server has no read-only transaction or session mode. The sync
    runs inside a transaction that is always rolled back and never
    committed, the SQL is validated at config load, and the login must hold
    only SELECT (db_datareader or narrower). pymssql's query timeout counts
    whole seconds, so statement_timeout_ms rounds up."""

    name = "mssql"
    drivers = ("pymssql",)
    extra = "mssql"
    enforces_read_only = False

    def connect_args(self, timeout_ms: int) -> dict[str, Any]:
        return {"login_timeout": CONNECT_TIMEOUT_S, "timeout": max(1, math.ceil(timeout_ms / 1000))}

    def begin_read_only(self, conn: Connection, timeout_ms: int) -> None:
        conn.execute(text(f"SET LOCK_TIMEOUT {int(timeout_ms)}"))

    def classify(self, exc: BaseException) -> FailureKind | None:
        # By message text, not number: pymssql reports the number of the
        # process's previous DB-Lib message, so after a failed login a
        # refused connection also arrives as 18456 (seen live).
        for candidate in exception_chain(exc):
            message = self.describe(candidate).lower()
            if "unable to connect" in message:
                return None  # unreachable server: a connect failure
            if "login failed" in message:
                return FailureKind.AUTH
            if "timed out" in message or "timeout expired" in message:
                return FailureKind.TIMEOUT
        return None

    def describe(self, exc: BaseException) -> str:
        inner = _driver_exception(exc)
        args: tuple[Any, ...] = getattr(inner, "args", ())
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]  # pymssql wraps (number, message bytes) in one tuple
        parts = [
            arg.decode("utf-8", "replace") if isinstance(arg, bytes) else str(arg) for arg in args
        ]
        return " ".join(parts) if parts else str(inner)


class OracleDialect(DatabaseDialect):
    """SET TRANSACTION READ ONLY plus oracledb's call_timeout (a per round
    trip limit, enforced by the driver: the call is aborted and the
    connection discarded). python-oracledb's thin mode needs no Oracle
    client install."""

    name = "oracle"
    drivers = ("oracledb",)
    extra = "oracle"

    _CODES: ClassVar[dict[str, FailureKind]] = {
        "ORA-01017": FailureKind.AUTH,  # invalid username/password
        "ORA-01005": FailureKind.AUTH,
        "ORA-28000": FailureKind.AUTH,  # account locked
        "DPY-4024": FailureKind.TIMEOUT,  # call timeout exceeded
        "DPY-4011": FailureKind.TIMEOUT,  # connection closed (after a call timeout)
        "ORA-03156": FailureKind.TIMEOUT,
        "ORA-01013": FailureKind.TIMEOUT,  # user requested cancel
        "ORA-01456": FailureKind.READ_ONLY,  # DML inside a READ ONLY transaction
        "ORA-16000": FailureKind.READ_ONLY,
    }
    _CODE_RE = re.compile(r"\b(ORA-\d{5}|DPY-\d{4})\b")

    def connect_args(self, timeout_ms: int) -> dict[str, Any]:
        return {"tcp_connect_timeout": float(CONNECT_TIMEOUT_S)}

    def on_connect(self, dbapi_connection: Any, timeout_ms: int) -> None:
        dbapi_connection.call_timeout = int(timeout_ms)

    def begin_read_only(self, conn: Connection, timeout_ms: int) -> None:
        # Must be the transaction's first statement; connecting runs only
        # SELECTs, which don't start one in Oracle.
        conn.execute(text("SET TRANSACTION READ ONLY"))

    def classify(self, exc: BaseException) -> FailureKind | None:
        for candidate in exception_chain(exc):
            for code in self._CODE_RE.findall(str(candidate)):
                if code in self._CODES:
                    return self._CODES[code]
        return None


DIALECTS: dict[str, DatabaseDialect] = {
    d.name: d for d in (PostgreSQLDialect(), MySQLDialect(), SQLServerDialect(), OracleDialect())
}
