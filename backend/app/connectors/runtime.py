"""Executes connector operations for one chat turn.

ChatOrchestrator binds a TurnConnectors for every turn (bind_turn), built
from the validated Principal and the assistant's ``connectors`` config. The
existing tool path then reaches it: providers resolve tool definitions via
app.tools.builtin.tool_definition and dispatch calls via execute_tool, which
hands connector tool names to TurnConnectors.execute. The Principal lives
only here, never in anything the model sees or sends.

Order of checks in execute(), each failing closed:
  1. the tool is one of this assistant's connector operations;
  2. the caller holds every required label (no request is made otherwise);
  3. identity arguments the model sent are dropped (and noted), the rest are
     validated against the schema the model was shown;
  4. the identity value comes from the configured claim on the Principal,
     and is missing means no call;
  5. one GET, bounded by the operation's timeout and response-size cap.
A failure becomes a structured error result for the model, never an
exception that ends the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator

from app.config.connectors import ConnectorConfig
from app.connectors.auth import ConnectorCallError, auth_for
from app.connectors.base import ConnectorOperation, tool_name
from app.connectors.catalog import CONNECTOR_TYPES
from app.core.principal import PrincipalContext
from app.core.tool_definition import ToolDefinition, ToolKind
from app.observability.audit import record_connector_call, record_guardrail
from app.observability.tracing import mark_error, start_span

logger = logging.getLogger(__name__)

# What the model is told for each failure kind. Fixed text: never upstream
# content. Every one tells it not to guess.
_DO_NOT_GUESS = "Do not estimate or guess the value."
_FAILURE_MESSAGES = {
    "timeout": "{source} did not respond in time, so the live data could not be retrieved.",
    "unavailable": "{source} could not be reached, so the live data could not be retrieved.",
    "auth_failed": "{source} could not be reached (the service connection was rejected).",
    "not_configured": "{source} is not configured on this server, so it could not be reached.",
    "bad_response": "{source} returned a response that could not be read.",
    "response_too_large": "{source} returned more data than allowed, so it was discarded.",
    "not_found": "{source} has no record for the signed-in user.",
    "denied": "The signed-in user is not permitted to use this {source} lookup.",
    "identity_unavailable": (
        "The signed-in user's {source} id is not available, so their data cannot be looked up."
    ),
    "invalid_arguments": "The arguments were not valid for this lookup: {detail}",
    "unknown_tool": "This lookup is not available.",
}
# Kinds where the system itself couldn't be reached or read; used for the
# user-facing "couldn't be reached" reply/notice.
UNREACHABLE_KINDS = frozenset(
    {
        "timeout",
        "unavailable",
        "auth_failed",
        "not_configured",
        "bad_response",
        "response_too_large",
    }
)


@dataclass(frozen=True)
class LiveSource:
    """Provenance for a connector-backed answer. Not a document citation."""

    connector: str
    display_name: str
    operations: tuple[str, ...]
    fetched_at: datetime


@dataclass(frozen=True)
class ConnectorOutcome:
    connector: str
    display_name: str
    operation: str
    outcome: str
    fetched_at: datetime | None = None


@dataclass(frozen=True)
class _BoundOperation:
    connector: ConnectorConfig
    operation: ConnectorOperation
    required_labels: frozenset[str]
    timeout_seconds: float
    max_response_bytes: int
    definition: ToolDefinition


def _bind_operations(connectors: Sequence[ConnectorConfig]) -> dict[str, _BoundOperation]:
    bound: dict[str, _BoundOperation] = {}
    for connector in connectors:
        connector_type = CONNECTOR_TYPES[connector.type]
        for op_config in connector.operations:
            operation = connector_type.operations[op_config.name]
            name = tool_name(connector.name, operation.name)
            labels = operation.required_labels | op_config.required_labels
            # Config can tighten the operation's limits, never loosen them.
            timeout = operation.timeout_seconds
            if op_config.timeout_seconds is not None:
                timeout = min(timeout, op_config.timeout_seconds)
            max_bytes = operation.max_response_bytes
            if op_config.max_response_bytes is not None:
                max_bytes = min(max_bytes, op_config.max_response_bytes)
            bound[name] = _BoundOperation(
                connector=connector,
                operation=operation,
                required_labels=labels,
                timeout_seconds=timeout,
                max_response_bytes=max_bytes,
                definition=ToolDefinition(
                    name=name,
                    description=f"[{connector.display_name}, live] {operation.description}",
                    kind=ToolKind.QUERY,
                    input_schema=operation.model_input_schema(),
                    requires_labels=labels,
                ),
            )
    return bound


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Scalar leaves of a JSON value, keyed by path, for the tool-result
    check (it only compares top-level scalars)."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}{index}."))
        return out
    return {prefix.rstrip("."): value}


class TurnConnectors:
    """The connector operations available in one turn, bound to its caller."""

    def __init__(
        self,
        principal: PrincipalContext,
        connectors: Sequence[ConnectorConfig],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._principal = principal
        self._operations = _bind_operations(connectors)
        self._transport = transport
        self.outcomes: list[ConnectorOutcome] = []

    # --- what the model is offered ------------------------------------------

    def _authorized(self, bound: _BoundOperation) -> bool:
        # Fail closed: an operation with no required labels is callable by
        # no one (config validation already requires at least one).
        return bool(bound.required_labels) and bound.required_labels <= self._principal.labels

    def is_connector_tool(self, name: str) -> bool:
        return name in self._operations

    def authorized_tool_names(self) -> list[str]:
        return [name for name, bound in self._operations.items() if self._authorized(bound)]

    def definition(self, name: str) -> ToolDefinition | None:
        bound = self._operations.get(name)
        if bound is None or not self._authorized(bound):
            return None
        return bound.definition

    # --- after the model call -----------------------------------------------

    def live_sources(self) -> tuple[LiveSource, ...]:
        by_connector: dict[str, list[ConnectorOutcome]] = {}
        for outcome in self.outcomes:
            if outcome.outcome == "ok":
                by_connector.setdefault(outcome.connector, []).append(outcome)
        return tuple(
            LiveSource(
                connector=connector,
                display_name=calls[0].display_name,
                operations=tuple(dict.fromkeys(c.operation for c in calls)),
                fetched_at=max(c.fetched_at for c in calls if c.fetched_at is not None),
            )
            for connector, calls in by_connector.items()
        )

    def unreachable_sources(self) -> tuple[str, ...]:
        """Display names of connectors that failed to answer this turn and
        never succeeded in it."""
        succeeded = {o.connector for o in self.outcomes if o.outcome == "ok"}
        names = [
            o.display_name
            for o in self.outcomes
            if o.outcome in UNREACHABLE_KINDS and o.connector not in succeeded
        ]
        return tuple(dict.fromkeys(names))

    # --- execution ----------------------------------------------------------

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        bound = self._operations.get(name)
        if bound is None:
            return {"error": _FAILURE_MESSAGES["unknown_tool"], "error_kind": "unknown_tool"}

        connector, operation = bound.connector, bound.operation
        started = time.perf_counter()
        with start_span(
            "connector.call",
            connector=connector.name,
            connector_type=connector.type,
            operation=operation.name,
        ) as span:
            fetched_at: datetime | None = None
            status: int | None = None
            data: Any = None
            failure: ConnectorCallError | None = None
            try:
                data, status = await self._call(bound, arguments)
                fetched_at = datetime.now(UTC)
                outcome = "ok"
            except ConnectorCallError as exc:
                outcome, failure = exc.kind, exc
            latency_ms = int((time.perf_counter() - started) * 1000)

            span.set_attribute("connector.outcome", outcome)
            span.set_attribute("connector.latency_ms", latency_ms)
            if status is not None:
                span.set_attribute("http.status_code", status)
            if outcome != "ok":
                mark_error(span, outcome, outcome)

        self.outcomes.append(
            ConnectorOutcome(
                connector=connector.name,
                display_name=connector.display_name,
                operation=operation.name,
                outcome=outcome,
                fetched_at=fetched_at,
            )
        )
        logger.info(
            "connector %s.%s outcome=%s latency_ms=%d",
            connector.name,
            operation.name,
            outcome,
            latency_ms,
            extra={
                "event": "connector_call",
                "connector": connector.name,
                "operation": operation.name,
                "outcome": outcome,
                "latency_ms": latency_ms,
            },
        )

        if failure is not None:
            record_connector_call(
                name,
                connector=connector.name,
                operation=operation.name,
                outcome=outcome,
                latency_ms=latency_ms,
            )
            return self._error_result(bound, failure)

        assert fetched_at is not None
        result = {
            "source": f"{connector.display_name} (live)",
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "data": data,
        }
        record_connector_call(
            name,
            connector=connector.name,
            operation=operation.name,
            outcome="ok",
            latency_ms=latency_ms,
            values=_flatten(data),
        )
        return result

    def _error_result(self, bound: _BoundOperation, failure: ConnectorCallError) -> dict[str, Any]:
        template = _FAILURE_MESSAGES.get(failure.kind, "{source} returned an error.")
        message = template.format(source=bound.connector.display_name, detail=failure.message)
        if failure.kind in UNREACHABLE_KINDS:
            message = f"{message} Tell the user {bound.connector.display_name} couldn't be reached."
        return {
            "error": f"{message} {_DO_NOT_GUESS}",
            "error_kind": failure.kind,
            "source": bound.connector.display_name,
        }

    async def _call(self, bound: _BoundOperation, arguments: dict[str, Any]) -> tuple[Any, int]:
        connector, operation = bound.connector, bound.operation

        if not self._authorized(bound):
            raise ConnectorCallError("denied", "missing required labels")

        supplied_identity = sorted(set(arguments) & operation.identity_params)
        if supplied_identity:
            # Names only, never values: the model tried to choose whose data
            # to read. It is ignored; the server-side identity is used.
            record_guardrail(
                "connector_identity_argument_ignored",
                connector=connector.name,
                operation=operation.name,
                fields=supplied_identity,
            )
        model_args = {k: v for k, v in arguments.items() if k not in operation.identity_params}
        errors = sorted(
            Draft202012Validator(operation.model_input_schema()).iter_errors(model_args),
            key=lambda e: list(e.path),
        )
        if errors:
            raise ConnectorCallError("invalid_arguments", errors[0].message[:200])

        identity_value: str | None = None
        if operation.identity_params:
            assert connector.identity is not None  # enforced by config validation
            identity_value = self._principal.claims.get(connector.identity.claim) or None
            if identity_value is None:
                raise ConnectorCallError("identity_unavailable", "identity claim missing")

        values: dict[str, Any] = dict(model_args)
        for param in operation.identity_params:
            values[param] = identity_value
        path = operation.path.format(
            **{k: quote(str(v), safe="") for k, v in values.items() if f"{{{k}}}" in operation.path}
        )
        query = {k: str(values[k]) for k in operation.query_params if values.get(k) is not None}

        try:
            async with asyncio.timeout(bound.timeout_seconds):
                return await self._get(bound, f"{connector.base_url}{path}", query)
        except TimeoutError as exc:
            raise ConnectorCallError("timeout", "timed out") from exc

    @asynccontextmanager
    async def _client(self) -> AsyncIterator[httpx.AsyncClient]:
        if self._transport is not None:
            async with httpx.AsyncClient(transport=self._transport, follow_redirects=False) as c:
                yield c
        else:
            yield _shared_client()

    async def _get(
        self, bound: _BoundOperation, url: str, query: dict[str, str]
    ) -> tuple[Any, int]:
        auth = auth_for(bound.connector)
        async with self._client() as client:
            for attempt in (1, 2):
                headers = {"Accept": "application/json"}
                await auth.apply(headers, client=client, principal=self._principal)
                try:
                    async with client.stream(
                        "GET", url, params=query, headers=headers, timeout=bound.timeout_seconds
                    ) as response:
                        status = response.status_code
                        if status == 401 and attempt == 1:
                            # A cached token the system no longer accepts:
                            # drop it and retry once (safe, the call is a GET).
                            auth.invalidate()
                            continue
                        if status in (401, 403):
                            raise ConnectorCallError("auth_failed", f"http {status}")
                        if status == 404:
                            raise ConnectorCallError("not_found", "http 404")
                        if status != 200:
                            raise ConnectorCallError("unavailable", f"http {status}")
                        body = await _read_capped(response, bound.max_response_bytes)
                except httpx.TimeoutException as exc:
                    raise ConnectorCallError("timeout", "timed out") from exc
                except httpx.HTTPError as exc:
                    raise ConnectorCallError("unavailable", type(exc).__name__) from exc
                try:
                    payload = json.loads(body)
                except ValueError as exc:
                    raise ConnectorCallError("bad_response", "not JSON") from exc
                if not isinstance(payload, dict):
                    raise ConnectorCallError("bad_response", "not a JSON object")
                fields = bound.operation.result_fields
                data = {k: payload[k] for k in fields if k in payload} if fields else payload
                return data, status
        # Both attempts got 401 (the second one raises above); kept for the type checker.
        raise ConnectorCallError("auth_failed", "http 401")  # pragma: no cover


# One pooled client per event loop for real HTTP: building an AsyncClient
# loads a TLS context, which measured ~1s on a Windows dev machine, longer
# than the call itself. Per-request timeouts come from each operation.
_shared_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
    weakref.WeakKeyDictionary()
)


def _shared_client() -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _shared_clients.get(loop)
    if client is None or client.is_closed:
        client = _shared_clients[loop] = httpx.AsyncClient(follow_redirects=False, timeout=30.0)
    return client


async def _read_capped(response: httpx.Response, limit: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise ConnectorCallError("response_too_large", "content-length over cap")
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > limit:
            raise ConnectorCallError("response_too_large", "body over cap")
    return bytes(body)


# --- per-turn binding ---------------------------------------------------------

_current: ContextVar[TurnConnectors | None] = ContextVar("turn_connectors", default=None)


def bind_turn(turn: TurnConnectors) -> None:
    """Never reset, for the same reason as audit.bind: on the streaming path
    the rest of the turn runs in another task. Every turn binds afresh."""
    _current.set(turn)


def current_turn() -> TurnConnectors | None:
    return _current.get()
