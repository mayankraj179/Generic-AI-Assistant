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
  4. identity values come from the configured claims on the Principal; a
     missing claim means no call;
  5. the request is built from the operation's template only (the model
     supplies values, never a URL, host, path or header) and sent under the
     network rules in http.py, page by page, within the operation's timeout,
     payload cap, max_pages and max_items.
A failure becomes a structured error result for the model, never an
exception that ends the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
import time
import weakref
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import httpx
from jsonschema import Draft202012Validator

from app.config.connectors import ConnectorConfig
from app.connectors.auth import AuthContext, ConnectorCallError, auth_for
from app.connectors.http import (
    NetworkGuard,
    Request,
    Resolver,
    Response,
    send,
    system_resolver,
    within_base,
)
from app.connectors.spec import OperationSpec, tool_name
from app.connectors.templates import render as render_template
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
    "rate_limited": "{source} is limiting requests right now, so the live data could not be "
    "retrieved.",
    "auth_failed": "{source} could not be reached (the service connection was rejected).",
    "not_configured": "{source} is not configured on this server, so it could not be reached.",
    "blocked_address": "{source} is not reachable from this server (address not allowed).",
    "redirect_refused": "{source} pointed the request outside the configured system, so it was "
    "stopped.",
    "bad_response": "{source} returned a response that could not be read.",
    "response_too_large": "{source} returned more data than allowed, so it was discarded.",
    "not_found": "{source} has no matching record for the signed-in user.",
    "rejected": "{source} rejected the lookup (for example, the record does not exist or a "
    "value is not valid there).",
    "denied": "The signed-in user is not permitted to use this {source} lookup.",
    "identity_unavailable": (
        "The signed-in user's {source} id is not available, so their data cannot be looked up."
    ),
    "identity_unverified": (
        "The signed-in user's email address is not verified, so their {source} data cannot be "
        "looked up."
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
        "rate_limited",
        "auth_failed",
        "not_configured",
        "blocked_address",
        "redirect_refused",
        "bad_response",
        "response_too_large",
    }
)
# Kept literal in path values: legal in a path segment and unable to change
# its structure, but needed as-is by real ids (Graph site ids have commas,
# drive ids start "b!", /users/{upn} takes an "@"). "/", "?", "#", "%" and
# ":" (Graph's path-addressing syntax) are always encoded.
_PATH_SAFE = ",!@"
_LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;[^,]*\brel="?next"?', re.IGNORECASE)
_MISSING = object()


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
    spec: OperationSpec
    definition: ToolDefinition


def _bind_operations(connectors: Sequence[ConnectorConfig]) -> dict[str, _BoundOperation]:
    bound: dict[str, _BoundOperation] = {}
    for connector in connectors:
        for spec in connector.resolved_operations():
            name = tool_name(connector.name, spec.name)
            bound[name] = _BoundOperation(
                connector=connector,
                spec=spec,
                definition=ToolDefinition(
                    name=name,
                    description=f"[{connector.display_name}, live] {spec.description}",
                    kind=ToolKind.QUERY,
                    input_schema=spec.model_input_schema(),
                    requires_labels=spec.required_labels,
                ),
            )
    return bound


# --- response helpers ---------------------------------------------------------


def get_path(value: Any, path: str) -> Any:
    """Dotted-path lookup that also handles keys containing dots
    ("@odata.nextLink") and list indexes. Returns _MISSING if absent."""
    if isinstance(value, dict) and path in value:
        return value[path]
    parts = path.split(".")
    for i in range(1, len(parts)):
        head, rest = ".".join(parts[:i]), ".".join(parts[i:])
        if isinstance(value, dict) and head in value:
            found = get_path(value[head], rest)
            if found is not _MISSING:
                return found
        elif isinstance(value, list) and head.isdigit() and int(head) < len(value):
            found = get_path(value[int(head)], rest)
            if found is not _MISSING:
                return found
    if isinstance(value, list) and path.isdigit() and int(path) < len(value):
        return value[int(path)]
    return _MISSING


def _project(record: Any, fields: Sequence[str]) -> Any:
    if not fields or not isinstance(record, dict):
        return record
    out = {}
    for field in fields:
        found = get_path(record, field)
        if found is not _MISSING:
            out[field] = found
    return out


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


def _parse_json(response: Response) -> Any:
    try:
        payload = json.loads(response.body)
    except ValueError as exc:
        raise ConnectorCallError("bad_response", "not JSON") from exc
    if not isinstance(payload, dict | list):
        raise ConnectorCallError("bad_response", "not a JSON object or array")
    return payload


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --- per-turn execution -------------------------------------------------------


class TurnConnectors:
    """The connector operations available in one turn, bound to its caller."""

    def __init__(
        self,
        principal: PrincipalContext,
        connectors: Sequence[ConnectorConfig],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._principal = principal
        self._operations = _bind_operations(connectors)
        self._transport = transport
        self._resolver = resolver or system_resolver
        self._guards: dict[str, NetworkGuard] = {}
        self.outcomes: list[ConnectorOutcome] = []

    # --- what the model is offered ------------------------------------------

    def _authorized(self, bound: _BoundOperation) -> bool:
        # Fail closed: an operation with no required labels is callable by
        # no one (config validation already requires at least one).
        labels = bound.spec.required_labels
        return bool(labels) and labels <= self._principal.labels

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

        connector, spec = bound.connector, bound.spec
        started = time.perf_counter()
        with start_span(
            "connector.call",
            connector=connector.name,
            connector_type=connector.type,
            operation=spec.name,
            method=spec.method,
        ) as span:
            fetched_at: datetime | None = None
            data: Any = None
            pages = 0
            failure: ConnectorCallError | None = None
            try:
                data, pages = await self._call(bound, arguments)
                fetched_at = datetime.now(UTC)
                outcome = "ok"
            except ConnectorCallError as exc:
                outcome, failure = exc.kind, exc
            latency_ms = int((time.perf_counter() - started) * 1000)

            span.set_attribute("connector.outcome", outcome)
            span.set_attribute("connector.latency_ms", latency_ms)
            span.set_attribute("connector.pages", pages)
            if failure is not None:
                mark_error(span, outcome, outcome)

        self.outcomes.append(
            ConnectorOutcome(
                connector=connector.name,
                display_name=connector.display_name,
                operation=spec.name,
                outcome=outcome,
                fetched_at=fetched_at,
            )
        )
        logger.info(
            "connector %s.%s outcome=%s latency_ms=%d pages=%d",
            connector.name,
            spec.name,
            outcome,
            latency_ms,
            pages,
            extra={
                "event": "connector_call",
                "connector": connector.name,
                "operation": spec.name,
                "outcome": outcome,
                "latency_ms": latency_ms,
            },
        )

        if failure is not None:
            record_connector_call(
                name,
                connector=connector.name,
                operation=spec.name,
                outcome=outcome,
                latency_ms=latency_ms,
            )
            return self._error_result(bound, failure)

        assert fetched_at is not None
        record_connector_call(
            name,
            connector=connector.name,
            operation=spec.name,
            outcome="ok",
            latency_ms=latency_ms,
            values=_flatten(data),
        )
        return {
            "source": f"{connector.display_name} (live)",
            "fetched_at": fetched_at.isoformat(timespec="seconds"),
            "data": data,
        }

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
        connector, spec = bound.connector, bound.spec

        if not self._authorized(bound):
            raise ConnectorCallError("denied", "missing required labels")

        identity_params = spec.identity_params()
        supplied_identity = sorted(set(arguments) & identity_params.keys())
        if supplied_identity:
            # Names only, never values: the model tried to choose whose data
            # to read. It is ignored; the server-side identity is used.
            record_guardrail(
                "connector_identity_argument_ignored",
                connector=connector.name,
                operation=spec.name,
                fields=supplied_identity,
            )
        # Setting parameters are hidden too; a model-sent value is dropped.
        hidden = identity_params.keys() | spec.setting_params().keys()
        values = {k: v for k, v in arguments.items() if k not in hidden}
        errors = sorted(
            Draft202012Validator(spec.model_input_schema()).iter_errors(values),
            key=lambda e: list(e.path),
        )
        if errors:
            raise ConnectorCallError("invalid_arguments", errors[0].message[:200])

        for pname, param in identity_params.items():
            values[pname] = self._identity_value(param.claim or connector.identity.claim)
        for pname, param in spec.setting_params().items():
            values[pname] = connector.settings[param.setting or ""]

        loop = asyncio.get_running_loop()
        deadline = loop.time() + spec.timeout_seconds
        try:
            async with asyncio.timeout(spec.timeout_seconds):
                return await self._fetch(bound, values, deadline)
        except TimeoutError as exc:
            raise ConnectorCallError("timeout", "timed out") from exc

    def _identity_value(self, claim: str) -> str:
        """The caller's id from a validated claim. Fails closed: a missing or
        empty claim, or an email claim the IdP hasn't marked verified, means
        no call (an unverified email could be anyone's address)."""
        value = self._principal.claims.get(claim)
        if not value:
            raise ConnectorCallError("identity_unavailable", "identity claim missing")
        if claim == "email" and self._principal.claims.get("email_verified") != "true":
            raise ConnectorCallError("identity_unverified", "email not verified")
        return value

    def _first_request(self, bound: _BoundOperation, values: dict[str, Any]) -> Request:
        connector, spec = bound.connector, bound.spec
        path = spec.path
        query: dict[str, str] = {k: _query_value(v) for k, v in spec.fixed_query.items()}
        body: dict[str, Any] = dict(spec.fixed_body)
        for pname, param in spec.parameters.items():
            if pname not in values:
                continue
            value = values[pname]
            if param.location == "path":
                path = path.replace("{" + pname + "}", quote(str(value), safe=_PATH_SAFE))
            elif param.location == "query":
                query[pname] = _query_value(value)
            elif param.location == "body":
                body[pname] = value
        for tname, template in spec.templates.items():
            # Template parameters reach the request only here, escaped.
            raw = {n for n, p in spec.parameters.items() if p.raw}
            text = render_template(template, values, raw)
            if template.location == "query":
                query[tname] = text
            else:
                body[tname] = text
        pagination = spec.pagination
        if pagination.page_size_param:
            target = body if spec.method == "POST" else query
            target[pagination.page_size_param] = (
                pagination.page_size if spec.method == "POST" else str(pagination.page_size)
            )
        return Request(
            method=spec.method,
            url=f"{connector.base_url}{path}",
            params=query or None,
            json=body if spec.method == "POST" else None,
            headers=dict(spec.headers),
        )

    def _next_request(
        self,
        bound: _BoundOperation,
        current: Request,
        response: Response,
        payload: Any,
        records: list[Any],
        offset: int,
    ) -> Request | None:
        spec, pagination = bound.spec, bound.spec.pagination
        strategy = pagination.strategy

        if strategy in ("offset", "cursor"):
            if strategy == "offset":
                if not records or (pagination.page_size and len(records) < pagination.page_size):
                    return None
                key, value = pagination.offset_param, offset
            else:
                cursor = get_path(payload, pagination.cursor_path or "")
                if cursor is _MISSING or cursor in (None, ""):
                    return None
                key, value = pagination.cursor_param, cursor
            if spec.method == "POST":
                return replace(current, json={**(current.json or {}), key: value})
            return replace(current, params={**(current.params or {}), key: _query_value(value)})

        if strategy == "link_header":
            match = _LINK_NEXT_RE.search(response.headers.get("link", ""))
            link = match.group(1) if match else None
        else:  # odata_next_link
            found = get_path(payload, pagination.next_link_path)
            link = found if isinstance(found, str) and found else None
        if link is None:
            return None
        target = urljoin(response.url, link)
        if not within_base(bound.connector.base_url, target):
            # A link pointing anywhere else is never followed.
            raise ConnectorCallError("redirect_refused", "next-page link outside base_url")
        return replace(current, url=target, params=None)

    async def _fetch(
        self, bound: _BoundOperation, values: dict[str, Any], deadline: float
    ) -> tuple[Any, int]:
        connector, spec = bound.connector, bound.spec
        guard = self._guards.setdefault(
            connector.name,
            NetworkGuard(
                allow_private_network=connector.allow_private_network, resolver=self._resolver
            ),
        )
        auth = auth_for(connector)
        root, fields = spec.response.root, spec.response.fields
        max_items = spec.response.max_items
        request: Request | None = self._first_request(bound, values)

        async with self._client(connector) as client:
            auth_context = AuthContext(client=client, principal=self._principal, guard=guard.check)
            items: list[Any] = []
            pages = offset = 0
            truncated = False
            while request is not None:
                response = await send(
                    request,
                    client=client,
                    base_url=connector.base_url,
                    guard=guard,
                    auth=auth,
                    auth_context=auth_context,
                    timeout=spec.timeout_seconds,
                    max_bytes=spec.max_response_bytes,
                    deadline=deadline,
                )
                pages += 1
                payload = _parse_json(response)
                target = get_path(payload, root) if root else payload
                if target is _MISSING:
                    raise ConnectorCallError("bad_response", "response.root not in the response")

                if spec.pagination.strategy == "none":
                    if isinstance(target, list):
                        return {
                            "items": [_project(r, fields) for r in target[:max_items]],
                            "returned": min(len(target), max_items),
                            "truncated": len(target) > max_items,
                        }, pages
                    return _project(target, fields), pages

                if not isinstance(target, list):
                    raise ConnectorCallError("bad_response", "response.root is not a list")
                items.extend(target)
                offset += len(target)
                next_request = self._next_request(bound, request, response, payload, target, offset)
                if len(items) >= max_items or pages >= spec.pagination.max_pages:
                    truncated = len(items) > max_items or next_request is not None
                    break
                request = next_request

        return {
            "items": [_project(r, fields) for r in items[:max_items]],
            "returned": min(len(items), max_items),
            "truncated": truncated,
        }, pages

    @asynccontextmanager
    async def _client(self, connector: ConnectorConfig) -> AsyncIterator[httpx.AsyncClient]:
        if self._transport is not None:
            async with httpx.AsyncClient(transport=self._transport, follow_redirects=False) as c:
                yield c
        else:
            yield _shared_client(connector.tls_ca_bundle)


# One pooled client per event loop for real HTTP: building an AsyncClient
# loads a TLS context, which measured ~1s on a Windows dev machine, longer
# than the call itself. Per-request timeouts come from each operation.
# Redirects are never followed by httpx itself (see http.send).
# A connector with tls_ca_bundle gets its own pooled client per bundle.
_shared_clients: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str | None, httpx.AsyncClient]
] = weakref.WeakKeyDictionary()


def tls_context(ca_bundle: str) -> ssl.SSLContext:
    """The default (verifying) context plus the bundle's CAs."""
    path = Path(ca_bundle)
    if not path.is_file():
        raise ConnectorCallError("not_configured", "the connector's CA bundle file is missing")
    context = ssl.create_default_context()
    try:
        context.load_verify_locations(cafile=str(path))
    except (ssl.SSLError, OSError) as exc:
        raise ConnectorCallError("not_configured", "the connector's CA bundle is invalid") from exc
    return context


def _shared_client(ca_bundle: str | None = None) -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    clients = _shared_clients.setdefault(loop, {})
    client = clients.get(ca_bundle)
    if client is None or client.is_closed:
        verify: ssl.SSLContext | bool = tls_context(ca_bundle) if ca_bundle else True
        client = clients[ca_bundle] = httpx.AsyncClient(
            follow_redirects=False, timeout=30.0, verify=verify
        )
    return client


# --- per-turn binding ---------------------------------------------------------

_current: ContextVar[TurnConnectors | None] = ContextVar("turn_connectors", default=None)


def bind_turn(turn: TurnConnectors) -> None:
    """Never reset, for the same reason as audit.bind: on the streaming path
    the rest of the turn runs in another task. Every turn binds afresh."""
    _current.set(turn)


def current_turn() -> TurnConnectors | None:
    return _current.get()
