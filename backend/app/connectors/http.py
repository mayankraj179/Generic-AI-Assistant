"""One connector HTTP request, with the network rules applied.

- Origin: every URL (the first request, a redirect, a next-page link) must
  sit under the connector's base_url: same scheme, host and port, and a path
  under base_url's path. The model never supplies a URL; links the remote
  system returns are checked the same way.
- Address (SSRF): unless the connector sets allow_private_network, only https
  is allowed and the host must resolve to public addresses only. This is
  checked on every request, after DNS. Known gap: a DNS answer that changes
  between this check and httpx's own lookup (rebinding) isn't caught here;
  egress filtering at the network layer is the backstop for that.
- Redirects: followed only for GET, only within base_url, at most 3 hops.
- Throttling: a 429 or 503 with Retry-After is retried (at most twice) when
  the wait fits in the operation's remaining time; a 429 without it backs off
  1s then 2s on the same terms; otherwise rate_limited.
- 401: the cached credential is dropped and the request retried once.
- Size: the body is read up to the operation's payload cap, then refused.
"""

from __future__ import annotations

import asyncio
import ipaddress
import posixpath
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from app.connectors.auth import AuthContext, ConnectorAuth, ConnectorCallError

Resolver = Callable[[str, int], Awaitable[list[str]]]

MAX_REDIRECTS = 3
MAX_THROTTLE_RETRIES = 2
MAX_RETRY_AFTER_SECONDS = 10.0
BACKOFF_BASE_SECONDS = 1.0


async def system_resolver(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def within_base(base_url: str, url: str) -> bool:
    """True if ``url`` is on base_url's origin and under its path."""
    base, target = urlsplit(base_url), urlsplit(url)
    if (target.scheme, (target.hostname or "").lower()) != (
        base.scheme,
        (base.hostname or "").lower(),
    ):
        return False
    if (target.port or _default_port(target.scheme)) != (base.port or _default_port(base.scheme)):
        return False
    if target.username or target.password:
        return False
    path = posixpath.normpath(target.path or "/")
    base_path = base.path.rstrip("/")
    return path == base_path or path.startswith(base_path + "/") or base_path == ""


class NetworkGuard:
    """Per-turn, per-connector address check. DNS answers are cached for the
    turn so a paginated call resolves its host once."""

    def __init__(self, *, allow_private_network: bool, resolver: Resolver) -> None:
        self._allow_private = allow_private_network
        self._resolver = resolver
        self._checked: dict[tuple[str, int], None] = {}

    async def check(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ConnectorCallError("blocked_address", "not an http(s) URL")
        if self._allow_private:
            return
        if parts.scheme != "https":
            raise ConnectorCallError("blocked_address", "plain http to a public host")
        key = (parts.hostname.lower(), parts.port or 443)
        if key in self._checked:
            return
        try:
            addresses = await self._resolver(*key)
        except OSError as exc:
            raise ConnectorCallError("unavailable", "the host name could not be resolved") from exc
        if not addresses:
            raise ConnectorCallError("unavailable", "the host name could not be resolved")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address.split("%", 1)[0])
            except ValueError as exc:
                raise ConnectorCallError("blocked_address", "unparseable address") from exc
            if not ip.is_global:
                raise ConnectorCallError("blocked_address", "resolves to a non-public address")
        self._checked[key] = None


@dataclass(frozen=True)
class Request:
    method: str
    url: str
    params: dict[str, str] | None = None
    json: Any = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Response:
    status: int
    headers: httpx.Headers
    body: bytes
    url: str


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


async def send(
    request: Request,
    *,
    client: httpx.AsyncClient,
    base_url: str,
    guard: NetworkGuard,
    auth: ConnectorAuth,
    auth_context: AuthContext,
    timeout: float,
    max_bytes: int,
    deadline: float,
) -> Response:
    loop = asyncio.get_running_loop()
    redirects = throttle_retries = 0
    auth_retried = False
    while True:
        if not within_base(base_url, request.url):
            raise ConnectorCallError("redirect_refused", "URL outside the configured base_url")
        await guard.check(request.url)
        headers = {"Accept": "application/json", **request.headers}
        await auth.apply(headers, auth_context)

        wait: float | None = None
        try:
            async with client.stream(
                request.method,
                request.url,
                params=request.params,
                json=request.json,
                headers=headers,
                timeout=timeout,
            ) as response:
                status = response.status_code
                if status == 401 and not auth_retried:
                    # A cached token the system no longer accepts: drop it and
                    # retry once (safe: the call is read-only).
                    auth.invalidate()
                    auth_retried = True
                    continue
                if status in (429, 503):
                    delay = retry_after_seconds(response.headers.get("retry-after"))
                    if delay is None and status == 429:
                        # No Retry-After (Graph omits it sometimes): exponential
                        # backoff, as Microsoft's throttling guidance asks.
                        delay = BACKOFF_BASE_SECONDS * (2**throttle_retries)
                    fits = (
                        delay is not None
                        and delay <= MAX_RETRY_AFTER_SECONDS
                        and loop.time() + delay < deadline
                    )
                    if fits and throttle_retries < MAX_THROTTLE_RETRIES:
                        wait = delay
                    elif status == 429:
                        raise ConnectorCallError("rate_limited", "http 429")
                    else:
                        raise ConnectorCallError("unavailable", "http 503")
                elif 300 <= status < 400:
                    location = response.headers.get("location")
                    target = urljoin(str(response.url), location) if location else ""
                    if (
                        not location
                        or request.method != "GET"
                        or redirects >= MAX_REDIRECTS
                        or not within_base(base_url, target)
                    ):
                        raise ConnectorCallError("redirect_refused", f"http {status} redirect")
                    request = replace(request, url=target, params=None)
                    redirects += 1
                    continue
                elif status in (401, 403):
                    raise ConnectorCallError("auth_failed", f"http {status}")
                elif status == 404:
                    raise ConnectorCallError("not_found", "http 404")
                elif status in (400, 422):
                    # The system understood the request and refused it (e.g.
                    # JQL naming an issue that doesn't exist): not "unreachable".
                    raise ConnectorCallError("rejected", f"http {status}")
                elif not 200 <= status < 300:
                    raise ConnectorCallError("unavailable", f"http {status}")
                else:
                    body = await _read_capped(response, max_bytes)
                    return Response(status, response.headers, body, str(response.url))
        except httpx.TimeoutException as exc:
            raise ConnectorCallError("timeout", "timed out") from exc
        except httpx.HTTPError as exc:
            raise ConnectorCallError("unavailable", type(exc).__name__) from exc
        assert wait is not None
        throttle_retries += 1
        await asyncio.sleep(wait)


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
