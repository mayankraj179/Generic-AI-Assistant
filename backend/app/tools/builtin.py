from __future__ import annotations

import json
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.tool_definition import ToolDefinition, ToolKind
from app.observability.audit import record_tool_call
from app.observability.tracing import start_span

# ---------------------------------------------------------------------------
# The framework's only two built-in tools — a deliberately small, local,
# in-process registry rather than a real MCP server/client round-trip
# (per this task's own scope decision: these tools are pure, dependency-free
# computations with no reason to run out-of-process). Both function names
# below must exactly equal their ToolDefinition.name: GeminiProvider wraps
# these same functions directly in google-adk's FunctionTool, which derives
# the tool name the model sees from `func.__name__` (verified against the
# installed SDK — FunctionTool.__init__ has no name-override parameter), and
# OpenRouterProvider dispatches by name via execute_tool() below. Keeping
# exactly one implementation of each tool's behavior — used by both
# providers, wrapped differently to suit each one's calling convention.
# ---------------------------------------------------------------------------


class ToolExecutionError(Exception):
    """Raised only when execute_tool() is asked for a tool name that isn't
    registered — a genuine caller bug (a provider proposing a name never in
    prompt.enabled_tools), not something a model's malformed *arguments* can
    trigger. Malformed arguments to a registered tool never raise — see
    current_datetime()/calculate() below, which return a structured
    {"error": ...} dict instead so it can be fed straight back to the model
    as a tool result for it to retry, exactly like google-adk's own
    FunctionTool already does for its own argument-validation failures.
    """


def current_datetime(timezone: str = "UTC", format: str | None = None) -> dict:
    """Returns the current date and time.

    Args:
        timezone: IANA timezone name (e.g. "America/New_York"). Defaults to
            "UTC" when omitted.
        format: Optional strftime-style format string. When given, an
            additional "formatted" field is included in the result; the
            ISO 8601 "iso" field is always present regardless.

    Returns:
        A dict with "iso", "timezone", and "day_of_week" on success (plus
        "formatted" if a format was given), or a single "error" key if the
        timezone or format string was invalid.
    """
    tz_name = timezone or "UTC"
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"unknown IANA timezone '{tz_name}'"}

    now = datetime.now(tz)
    result: dict[str, str] = {
        "iso": now.isoformat(),
        "timezone": tz_name,
        "day_of_week": now.strftime("%A"),
    }
    if format:
        try:
            result["formatted"] = now.strftime(format)
        except (ValueError, TypeError) as exc:
            return {"error": f"invalid format string: {exc}"}
    return result


# ---------------------------------------------------------------------------
# calculate() — hand-written recursive-descent parser, deliberately NOT
# eval()/exec() or any general Python expression evaluator: even a
# "read-only" arithmetic tool is a code-execution risk under eval() if a
# model is ever prompted (accidentally or adversarially) to pass something
# shaped like "__import__('os').system('ls')". The tokenizer below only
# ever recognizes digits, '.', '+', '-', '*', '/', '(', ')' and whitespace —
# letters, underscores, quotes, and brackets are rejected at the tokenizer
# stage, before any evaluation happens, so there is no code path that could
# ever reach a name lookup, attribute access, or function call.
# ---------------------------------------------------------------------------

_DATE_DIFF_PATTERN = re.compile(
    r"^\s*days?\s+between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\s*$",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"\s*(\d+\.\d+|\d+|[()+\-*/])")


class _ExpressionError(Exception):
    """Internal to calculate() — always caught below and turned into a
    structured {"error": ...} result, never allowed to propagate."""


def _tokenize(expression: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    length = len(expression)
    while pos < length:
        match = _TOKEN_PATTERN.match(expression, pos)
        if not match:
            if expression[pos:].strip() == "":
                break
            raise _ExpressionError(
                f"unsupported character {expression[pos]!r} at position {pos}"
            )
        pos = match.end()
        tokens.append(match.group(1))
    return tokens


class _ArithmeticParser:
    """Recursive-descent parser over a flat token list, standard precedence:

        expr   := term (('+' | '-') term)*
        term   := factor (('*' | '/') factor)*
        factor := ('+' | '-') factor | NUMBER | '(' expr ')'

    No name/identifier token exists in this grammar at all — there is
    nothing here that could ever resolve to a Python symbol.
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> float:
        value = self._expr()
        if self._pos != len(self._tokens):
            raise _ExpressionError(f"unexpected token {self._tokens[self._pos]!r}")
        return value

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _advance(self) -> str:
        token = self._peek()
        if token is None:
            raise _ExpressionError("unexpected end of expression")
        self._pos += 1
        return token

    def _expr(self) -> float:
        value = self._term()
        while self._peek() in ("+", "-"):
            op = self._advance()
            rhs = self._term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def _term(self) -> float:
        value = self._factor()
        while self._peek() in ("*", "/"):
            op = self._advance()
            rhs = self._factor()
            if op == "*":
                value *= rhs
            else:
                if rhs == 0:
                    raise _ExpressionError("division by zero")
                value /= rhs
        return value

    def _factor(self) -> float:
        token = self._peek()
        if token in ("+", "-"):
            self._advance()
            value = self._factor()
            return value if token == "+" else -value
        if token == "(":
            self._advance()
            value = self._expr()
            if self._advance() != ")":
                raise _ExpressionError("expected closing parenthesis")
            return value
        token = self._advance()
        try:
            return float(token)
        except ValueError as exc:
            raise _ExpressionError(f"expected a number, got {token!r}") from exc


def calculate(expression: str) -> dict:
    """Evaluates a simple arithmetic expression, or a date-difference phrase.

    Args:
        expression: Either an arithmetic expression using +, -, *, /, and
            parentheses (e.g. "(96.5 - 79.1) / 79.1 * 100"), or a phrase of
            the exact form "days between YYYY-MM-DD and YYYY-MM-DD".
            Nothing else is supported — no variables, function calls, or
            other syntax.

    Returns:
        {"result": <number>, "expression": <input>, "kind": "arithmetic" |
        "date_difference"} on success, or {"error": <message>} if the
        expression doesn't match either supported grammar.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "expression must be a non-empty string"}

    date_diff = _DATE_DIFF_PATTERN.match(expression)
    if date_diff:
        try:
            start = date.fromisoformat(date_diff.group(1))
            end = date.fromisoformat(date_diff.group(2))
        except ValueError as exc:
            return {"error": f"invalid date in expression: {exc}"}
        return {
            "result": (end - start).days,
            "expression": expression,
            "kind": "date_difference",
        }

    try:
        tokens = _tokenize(expression)
        if not tokens:
            return {"error": "expression is empty"}
        value = _ArithmeticParser(tokens).parse()
    except _ExpressionError as exc:
        return {"error": f"invalid expression: {exc}"}
    return {"result": value, "expression": expression, "kind": "arithmetic"}


CURRENT_DATETIME_TOOL = ToolDefinition(
    name="current_datetime",
    description=(
        "Returns the current date and time, optionally in a specific IANA "
        "timezone. Use this instead of guessing or assuming the current date."
    ),
    kind=ToolKind.QUERY,
    input_schema={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone name, e.g. 'America/New_York'. "
                    "Defaults to UTC if omitted."
                ),
            },
            "format": {
                "type": "string",
                "description": (
                    "Optional strftime-style format string for an "
                    "additional human-formatted value."
                ),
            },
        },
        "additionalProperties": False,
        "required": [],
    },
)

CALCULATE_TOOL = ToolDefinition(
    name="calculate",
    description=(
        "Evaluates a simple arithmetic expression (+, -, *, /, parentheses) "
        "or a date-difference phrase of the exact form 'days between "
        "YYYY-MM-DD and YYYY-MM-DD'. Use this for any calculation instead of "
        "computing it yourself — it is exact, you are not. Does not support "
        "variables, function calls, or any other syntax."
    ),
    kind=ToolKind.QUERY,
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression or date-difference phrase to evaluate.",
            },
        },
        "additionalProperties": False,
        "required": ["expression"],
    },
)

TOOL_REGISTRY: dict[str, ToolDefinition] = {
    CURRENT_DATETIME_TOOL.name: CURRENT_DATETIME_TOOL,
    CALCULATE_TOOL.name: CALCULATE_TOOL,
}

_TOOL_FUNCTIONS = {
    "current_datetime": current_datetime,
    "calculate": calculate,
}


async def execute_tool(name: str, arguments: dict) -> dict:
    """The single dispatch point both providers resolve tool calls through:
    GeminiProvider indirectly (FunctionTool wraps these same functions and
    google-adk calls them directly), OpenRouterProvider directly (its manual
    tool-call loop calls this). Async only to match this codebase's
    async-at-the-boundary convention — every implementation above is sync,
    pure, and instant.

    A mismatched-arguments call (e.g. the model passes an unexpected keyword)
    is caught here and turned into a structured error result rather than
    propagating as a TypeError — the same "never crash the turn, feed a
    structured error back to the model instead" discipline the tool
    implementations themselves already apply to malformed input values.
    """
    # Traced and audited here for the OpenRouter/xAI tool loops. Gemini's
    # tools are invoked by google-adk directly, which traces them itself
    # (execute_tool spans); GeminiProvider audits them from ADK events.
    with start_span(f"tool.{name}", tool=name, arguments=json.dumps(arguments)[:500]) as span:
        func = _TOOL_FUNCTIONS.get(name)
        if func is None:
            record_tool_call(name, ok=False, error="unknown tool")
            raise ToolExecutionError(f"unknown tool '{name}'")
        try:
            result = func(**arguments)
        except TypeError as exc:
            result = {"error": f"invalid arguments for '{name}': {exc}"}
        error = result.get("error") if isinstance(result, dict) else None
        span.set_attribute("tool.ok", error is None)
        record_tool_call(
            name,
            ok=error is None,
            error=str(error) if error else None,
            result=result if isinstance(result, dict) else None,
        )
        return result
