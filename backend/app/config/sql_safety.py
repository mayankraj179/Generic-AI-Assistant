"""Config-time checks that a database source's reviewed SQL is a single
read query, in the grammar of the engine it will run on.

This is one layer, not the guarantee. At sync time the query also runs in
a read-only transaction where the engine has one, under a timeout, as a
login that should only hold SELECT grants (see README, "Database sources").
Two independent checks run here, because neither is enough alone:

1. Token deny-list. Any unquoted keyword that writes, changes the schema,
   controls transactions or runs code (INSERT, DELETE, INTO, EXEC, ...)
   fails the query. This catches what a lenient parse misses: T-SQL needs
   no ';' between statements, and sqlglot reads ``SELECT 1 DELETE FROM t``
   as one SELECT that aliases a column ``DELETE``. A column that really is
   named after one of these keywords has to be quoted.
2. Parse. The SQL must parse, in the engine's own grammar, to exactly one
   query (SELECT / UNION / ...), with no write, SELECT INTO or row lock
   anywhere in the tree (e.g. no ``WITH d AS (DELETE ...) SELECT``).
   SQL that sqlglot can't parse is rejected. Failing closed is the safe
   default for something an LLM never writes but an admin might get wrong.

Neither check can tell whether a function called by the query has side
effects (e.g. a user-defined function that writes). Only the database
role's privileges and the read-only transaction cover that.
"""

from __future__ import annotations

from typing import Any

# DatabaseSourceConfig.dialect -> sqlglot's dialect name.
SQLGLOT_DIALECTS = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "mssql": "tsql",
    "oracle": "oracle",
}

FORBIDDEN_KEYWORDS = frozenset(
    {
        # data changes
        "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "TRUNCATE", "INTO",
        "COPY", "LOAD", "OUTFILE", "DUMPFILE",
        # schema and privileges
        "CREATE", "ALTER", "DROP", "RENAME", "GRANT", "REVOKE", "DENY", "COMMENT",
        # procedures, dynamic SQL, external data
        "EXEC", "EXECUTE", "CALL", "DO", "PREPARE", "DEALLOCATE", "DECLARE",
        "OPENROWSET", "OPENQUERY", "OPENDATASOURCE",
        # transactions, sessions, locks
        "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "TRANSACTION", "SET", "USE",
        "LOCK", "UNLOCK", "HANDLER",
        # administration
        "KILL", "SHUTDOWN", "DBCC", "BACKUP", "RESTORE", "VACUUM", "ANALYZE",
        "EXPLAIN", "PRAGMA", "ATTACH", "DETACH", "WAITFOR",
    }
)  # fmt: skip


def _literal_token_types() -> frozenset[Any]:
    from sqlglot import TokenType

    names = (
        "STRING", "IDENTIFIER", "NATIONAL_STRING", "RAW_STRING", "HEREDOC_STRING",
        "BIT_STRING", "HEX_STRING", "BYTE_STRING", "UNICODE_STRING",
    )  # fmt: skip
    return frozenset(getattr(TokenType, n) for n in names if hasattr(TokenType, n))


def check_read_only_query(sql: str, dialect: str) -> None:
    """Raises ValueError unless ``sql`` is one read-only query for ``dialect``."""
    import sqlglot
    from sqlglot import exp
    from sqlglot.errors import SqlglotError

    read = SQLGLOT_DIALECTS[dialect]
    try:
        tokens = sqlglot.tokenize(sql, read=read)
    except SqlglotError as exc:
        raise ValueError(f"sql could not be tokenized as {dialect}: {exc}") from None

    literal = _literal_token_types()
    found = sorted(
        {
            token.text.upper()
            for token in tokens
            if token.token_type not in literal and token.text.upper() in FORBIDDEN_KEYWORDS
        }
    )
    if found:
        raise ValueError(
            f"sql uses keyword(s) not allowed in a read query: {', '.join(found)} "
            "(quote an identifier that really has one of these names)"
        )

    try:
        statements = [s for s in sqlglot.parse(sql, read=read) if s is not None]
    except SqlglotError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        raise ValueError(f"sql could not be parsed as {dialect}: {first_line}") from None
    if len(statements) != 1:
        raise ValueError(f"sql must be a single statement, found {len(statements)}")
    statement = statements[0]
    if not isinstance(statement, exp.Query):
        raise ValueError(f"sql must be a read query, not {type(statement).__name__.upper()}")
    writes = (
        exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
        exp.Command, exp.Into, exp.Lock, exp.TruncateTable,
    )  # fmt: skip
    for node in statement.walk():
        if isinstance(node, writes):
            raise ValueError(f"sql must not contain {type(node).__name__.upper()}")
