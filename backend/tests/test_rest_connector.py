"""The generic declarative REST connector (type: rest): load-time rules
(read-only, paths, headers, network), auth strategies, pagination,
throttling, redirects, SSRF controls, and cross-user isolation.

Self-contained; the remote system is an httpx.MockTransport per test.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError

from app.config.connectors import ConnectorConfig
from app.connectors.auth import reset_auth_cache
from app.connectors.http import within_base
from app.connectors.runtime import TurnConnectors, bind_turn
from app.core.principal import PrincipalContext
from app.observability import audit
from app.observability.audit import TurnRecorder
from app.tools.builtin import execute_tool, tool_definition

LABELS = ["role:authenticated"]
BASE = "https://api.example.test/v2"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    reset_auth_cache()
    for name, value in {
        "T_KEY": "key-0123456789abcdef",
        "T_USER": "svc-reader",
        "T_PASS": "basic-password-0123456",
        "T_TOKEN": "pat-0123456789abcdef",
        "T_CLIENT_ID": "client-abc",
        "T_CLIENT_SECRET": "client-secret-0123456789",
        "T_SUBJECT": "integration@example.test",
    }.items():
        monkeypatch.setenv(name, value)
    yield
    reset_auth_cache()


async def public_dns(host: str, port: int) -> list[str]:
    return ["93.184.216.34"]


def _principal(user: str = "alice", labels=("role:authenticated",)) -> PrincipalContext:
    return PrincipalContext(
        tenant_id="t1",
        principal_id=f"sub-{user}",
        labels=frozenset(labels),
        clearance=1,
        claims={
            "preferred_username": user,
            "email": f"{user}@example.test",
            "email_verified": "true",
        },
    )


def _op(**overrides) -> dict:
    op = {
        "name": "list_tickets",
        "description": "Lists the signed-in user's tickets.",
        "method": "GET",
        "path": "/users/{user}/tickets",
        "parameters": {
            "user": {"in": "path", "identity": True},
            "status": {"in": "query", "enum": ["open", "closed"]},
        },
        "response": {"root": "data", "fields": ["id", "title", "status"]},
        "required_labels": LABELS,
    }
    op.update(overrides)
    return op


def _connector(ops: list[dict] | None = None, **overrides) -> dict:
    data = {
        "name": "tickets",
        "type": "rest",
        "display_name": "Ticket system",
        "base_url": BASE,
        "auth": {"method": "api_key", "header": "X-Api-Key", "key": "${T_KEY}"},
        "identity": {"claim": "preferred_username"},
        "operations": ops or [_op()],
    }
    data.update(overrides)
    return data


class Remote:
    """A scriptable remote system. ``route`` maps a request to a response."""

    def __init__(self, route: Callable[[httpx.Request], httpx.Response]) -> None:
        self.route = route
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.route(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def _bind(remote: Remote, principal=None, resolver=public_dns, **connector) -> TurnConnectors:
    config = ConnectorConfig.model_validate(_connector(**connector))
    turn = TurnConnectors(
        principal or _principal(), [config], transport=remote.transport, resolver=resolver
    )
    bind_turn(turn)
    return turn


def _ok(payload: Any, **kw) -> httpx.Response:
    return httpx.Response(200, json=payload, **kw)


TOOL = "tickets__list_tickets"


# ---------------------------------------------------------------------------
# Load-time rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_write_methods_are_rejected_at_load(method):
    with pytest.raises(ValidationError, match="connectors are read-only"):
        ConnectorConfig.model_validate(_connector([_op(method=method)]))


def test_post_requires_read_only_post():
    with pytest.raises(ValidationError, match="only allowed with read_only_post"):
        ConnectorConfig.model_validate(_connector([_op(method="POST")]))


def test_read_only_post_on_get_is_rejected():
    with pytest.raises(ValidationError, match="read_only_post is only valid with POST"):
        ConnectorConfig.model_validate(_connector([_op(read_only_post=True)]))


def test_body_params_need_post():
    params = {"user": {"in": "path", "identity": True}, "q": {"in": "body"}}
    with pytest.raises(ValidationError, match="needs a POST operation"):
        ConnectorConfig.model_validate(_connector([_op(parameters=params)]))


@pytest.mark.parametrize(
    "path",
    ["https://evil.test/x", "users/{user}", "/a/../admin", "/a?x=1", "//evil.test/{user}"],
)
def test_paths_cannot_name_a_host_or_escape(path):
    with pytest.raises(ValidationError):
        ConnectorConfig.model_validate(_connector([_op(path=path)]))


def test_path_placeholders_must_match_path_params():
    with pytest.raises(ValidationError, match="placeholders"):
        ConnectorConfig.model_validate(_connector([_op(path="/users/{user}/{extra}")]))


@pytest.mark.parametrize("header", ["Authorization", "X-Api-Key", "Cookie", "Host", "X-Token"])
def test_credential_headers_are_rejected(header):
    with pytest.raises(ValidationError, match="not allowed here"):
        ConnectorConfig.model_validate(_connector([_op(headers={header: "literal"})]))


def test_inline_operation_needs_required_labels():
    with pytest.raises(ValidationError, match="at least one label"):
        ConnectorConfig.model_validate(_connector([_op(required_labels=[])]))


def test_preset_operations_cannot_be_redefined():
    with pytest.raises(ValidationError, match="can't be redefined"):
        ConnectorConfig.model_validate(
            _connector(type="hr_api", ops=[_op(name="get_leave_balance")])
        )


def test_pagination_needs_a_record_root():
    with pytest.raises(ValidationError, match="response.root"):
        ConnectorConfig.model_validate(
            _connector([_op(response={}, pagination={"strategy": "link_header"})])
        )


@pytest.mark.parametrize(
    "base_url",
    ["http://api.example.test", "https://127.0.0.1", "https://10.1.2.3/api", "https://localhost"],
)
def test_plain_http_and_private_addresses_need_opt_in(base_url):
    with pytest.raises(ValidationError, match="allow_private_network"):
        ConnectorConfig.model_validate(_connector(base_url=base_url))
    ConnectorConfig.model_validate(_connector(base_url=base_url, allow_private_network=True))


@pytest.mark.parametrize(
    "auth",
    [
        {"method": "basic", "username": "${T_USER}", "password": "hunter2-literal"},
        {"method": "bearer", "token": "pat-literal-0123456789"},
        {
            "method": "oauth_jwt_bearer",
            "token_url": "https://login.example.test/token",
            "client_id": "${T_CLIENT_ID}",
            "subject": "${T_SUBJECT}",
            "audience": "https://login.example.test",
            "private_key": "-----BEGIN PRIVATE KEY-----literal",
        },
    ],
)
def test_literal_secrets_rejected_for_every_auth_method(auth):
    with pytest.raises(ValidationError, match="literal secrets are not allowed") as excinfo:
        ConnectorConfig.model_validate(_connector(auth=auth))
    assert "literal" not in str(excinfo.value).split("literal secrets")[0]
    assert "hunter2" not in str(excinfo.value) and "BEGIN PRIVATE" not in str(excinfo.value)


def test_secret_extra_params_rejected():
    auth = {
        "method": "oauth_client_credentials",
        "token_url": "https://login.example.test/token",
        "client_id": "${T_CLIENT_ID}",
        "client_secret": "${T_CLIENT_SECRET}",
        "extra_params": {"client_assertion": "abc"},
    }
    with pytest.raises(ValidationError, match="extra_params may not set"):
        ConnectorConfig.model_validate(_connector(auth=auth))


# ---------------------------------------------------------------------------
# Requests: identity, parameters, read_only_post, projection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_builds_request_from_template_and_projects_fields():
    remote = Remote(
        lambda r: _ok({"data": [{"id": 1, "title": "VPN", "status": "open", "secret_note": "x"}]})
    )
    _bind(remote)
    result = await execute_tool(TOOL, {"status": "open", "user": "mallory"})

    (request,) = remote.requests
    assert request.url.path == "/v2/users/alice/tickets"
    assert parse_qs(request.url.query.decode()) == {"status": ["open"]}
    assert request.headers["x-api-key"] == "key-0123456789abcdef"
    assert result["data"] == {
        "items": [{"id": 1, "title": "VPN", "status": "open"}],
        "returned": 1,
        "truncated": False,
    }
    assert "user" not in json.dumps(tool_definition(TOOL).input_schema)


@pytest.mark.asyncio
async def test_model_cannot_inject_a_host_through_a_path_parameter():
    params = {
        "user": {"in": "path", "identity": True},
        "ticket_id": {"in": "path"},
    }
    op = _op(
        name="get_ticket",
        path="/users/{user}/tickets/{ticket_id}",
        parameters=params,
        required=["ticket_id"],
        response={},
    )
    remote = Remote(lambda r: _ok({"id": 1}))
    _bind(remote, ops=[op])
    await execute_tool("tickets__get_ticket", {"ticket_id": "@evil.test/../../admin?x=1#"})

    (request,) = remote.requests
    assert request.url.host == "api.example.test"
    assert request.url.raw_path.decode().startswith("/v2/users/alice/tickets/%40evil.test%2F")
    assert request.url.query == b""


@pytest.mark.asyncio
async def test_read_only_post_sends_a_json_body_with_identity():
    op = _op(
        name="search",
        method="POST",
        read_only_post=True,
        path="/search",
        parameters={
            "text": {"in": "body", "max_length": 50},
            "reporter": {"in": "body", "identity": True, "claim": "email"},
        },
        required=["text"],
        fixed_body={"fields": ["id", "title"]},
        response={"root": "results"},
    )
    remote = Remote(lambda r: _ok({"results": [{"id": 7, "title": "laptop"}]}))
    _bind(remote, ops=[op])
    result = await execute_tool("tickets__search", {"text": "laptop", "reporter": "bob@x"})

    (request,) = remote.requests
    assert request.method == "POST"
    assert json.loads(request.content) == {
        "fields": ["id", "title"],
        "text": "laptop",
        "reporter": "alice@example.test",
    }
    assert result["data"]["items"] == [{"id": 7, "title": "laptop"}]


@pytest.mark.asyncio
async def test_arguments_validated_against_declared_schema():
    remote = Remote(lambda r: _ok({"data": []}))
    _bind(remote)
    for bad in ({"status": "deleted"}, {"status": "open", "url": "https://evil.test"}):
        result = await execute_tool(TOOL, bad)
        assert result["error_kind"] == "invalid_arguments"
    assert remote.requests == []


# ---------------------------------------------------------------------------
# Auth strategies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_auth():
    remote = Remote(lambda r: _ok({"data": []}))
    _bind(remote, auth={"method": "basic", "username": "${T_USER}", "password": "${T_PASS}"})
    await execute_tool(TOOL, {})
    expected = base64.b64encode(b"svc-reader:basic-password-0123456").decode()
    assert remote.requests[0].headers["authorization"] == f"Basic {expected}"


@pytest.mark.asyncio
async def test_bearer_and_api_key_prefix():
    remote = Remote(lambda r: _ok({"data": []}))
    _bind(remote, auth={"method": "bearer", "token": "${T_TOKEN}"})
    await execute_tool(TOOL, {})
    assert remote.requests[0].headers["authorization"] == "Bearer pat-0123456789abcdef"

    remote2 = Remote(lambda r: _ok({"data": []}))
    _bind(
        remote2,
        auth={"method": "api_key", "header": "Authorization", "prefix": "SSWS ", "key": "${T_KEY}"},
    )
    await execute_tool(TOOL, {})
    assert remote2.requests[0].headers["authorization"] == "SSWS key-0123456789abcdef"


def _token_remote(
    issued: list[str],
    *,
    api: Callable[[httpx.Request], httpx.Response] | None = None,
    form_log: list[dict] | None = None,
    expires_in: int | None = 300,
) -> Remote:
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "login.example.test":
            if form_log is not None:
                form_log.append(
                    {"headers": dict(request.headers), "form": parse_qs(request.content.decode())}
                )
            token = f"tok-{len(issued)}-0123456789"
            issued.append(token)
            body = {"access_token": token, "token_type": "Bearer"}
            if expires_in is not None:
                body["expires_in"] = expires_in
            return _ok(body)
        auth = request.headers.get("authorization", "")
        if auth.removeprefix("Bearer ") not in issued:
            return httpx.Response(401)
        return api(request) if api else _ok({"data": []})

    return Remote(route)


@pytest.mark.asyncio
async def test_client_credentials_post_style_and_cache():
    issued: list[str] = []
    forms: list[dict] = []
    remote = _token_remote(issued, form_log=forms)
    auth = {
        "method": "oauth_client_credentials",
        "token_url": "https://login.example.test/tenant/oauth2/v2.0/token",
        "client_id": "${T_CLIENT_ID}",
        "client_secret": "${T_CLIENT_SECRET}",
        "scope": "https://api.example.test/.default",
        "client_auth": "post",
    }
    for _ in range(3):
        _bind(remote, auth=auth)
        assert "data" in await execute_tool(TOOL, {})
    assert len(issued) == 1
    form = forms[0]["form"]
    assert form["grant_type"] == ["client_credentials"]
    assert form["client_id"] == ["client-abc"] and form["client_secret"] == [
        "client-secret-0123456789"
    ]
    assert form["scope"] == ["https://api.example.test/.default"]
    assert "authorization" not in forms[0]["headers"]


@pytest.mark.asyncio
async def test_jwt_bearer_signs_assertion_and_caches_token(monkeypatch: pytest.MonkeyPatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    monkeypatch.setenv("T_PRIVATE_KEY", pem)
    issued: list[str] = []
    forms: list[dict] = []
    remote = _token_remote(issued, form_log=forms, expires_in=None)
    auth = {
        "method": "oauth_jwt_bearer",
        "token_url": "https://login.example.test/services/oauth2/token",
        "client_id": "${T_CLIENT_ID}",
        "subject": "${T_SUBJECT}",
        "audience": "https://login.example.test",
        "private_key": "${T_PRIVATE_KEY}",
    }
    _bind(remote, auth=auth)
    await execute_tool(TOOL, {})
    _bind(remote, auth=auth)
    await execute_tool(TOOL, {})

    assert len(issued) == 1  # no expires_in: default_token_ttl_seconds applies
    form = forms[0]["form"]
    assert form["grant_type"] == ["urn:ietf:params:oauth:grant-type:jwt-bearer"]
    claims = jwt.decode(
        form["assertion"][0],
        key.public_key(),
        algorithms=["RS256"],
        audience="https://login.example.test",
    )
    assert claims["iss"] == "client-abc" and claims["sub"] == "integration@example.test"
    assert claims["exp"] - claims["iat"] == 180 and claims["exp"] > time.time()


@pytest.mark.asyncio
async def test_bad_signing_key_is_not_configured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("T_PRIVATE_KEY", "not a pem key at all")
    remote = _token_remote([])
    auth = {
        "method": "oauth_jwt_bearer",
        "token_url": "https://login.example.test/token",
        "client_id": "${T_CLIENT_ID}",
        "subject": "${T_SUBJECT}",
        "audience": "https://login.example.test",
        "private_key": "${T_PRIVATE_KEY}",
    }
    _bind(remote, auth=auth)
    assert (await execute_tool(TOOL, {}))["error_kind"] == "not_configured"
    assert remote.requests == []


@pytest.mark.asyncio
async def test_token_endpoint_on_private_address_is_blocked():
    async def dns(host: str, port: int) -> list[str]:
        return ["169.254.169.254"] if host == "login.example.test" else ["93.184.216.34"]

    remote = _token_remote([])
    auth = {
        "method": "oauth_client_credentials",
        "token_url": "https://login.example.test/token",
        "client_id": "${T_CLIENT_ID}",
        "client_secret": "${T_CLIENT_SECRET}",
    }
    _bind(remote, resolver=dns, auth=auth)
    assert (await execute_tool(TOOL, {}))["error_kind"] == "blocked_address"
    assert remote.requests == []


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

ROWS = [{"id": i, "title": f"t{i}", "status": "open"} for i in range(1, 8)]


def _page_op(pagination: dict, root: str = "data", **kw) -> dict:
    return _op(
        pagination=pagination, response={"root": root, "fields": ["id"], "max_items": 50}, **kw
    )


@pytest.mark.asyncio
async def test_offset_pagination():
    def route(r):
        q = parse_qs(r.url.query.decode())
        start, size = int(q.get("startAt", ["0"])[0]), int(q["maxResults"][0])
        return _ok({"data": ROWS[start : start + size]})

    remote = Remote(route)
    op = _page_op(
        {
            "strategy": "offset",
            "offset_param": "startAt",
            "page_size": 3,
            "page_size_param": "maxResults",
            "max_pages": 5,
        }
    )
    _bind(remote, ops=[op])
    result = await execute_tool(TOOL, {})
    assert [i["id"] for i in result["data"]["items"]] == [1, 2, 3, 4, 5, 6, 7]
    assert result["data"]["truncated"] is False
    assert [parse_qs(r.url.query.decode()).get("startAt") for r in remote.requests] == [
        None,
        ["3"],
        ["6"],
    ]


@pytest.mark.asyncio
async def test_cursor_pagination_stops_at_max_pages_and_reports_truncation():
    def route(r):
        cursor = int(parse_qs(r.url.query.decode()).get("after", ["0"])[0])
        return _ok({"data": ROWS[cursor : cursor + 2], "meta": {"next": str(cursor + 2)}})

    remote = Remote(route)
    op = _page_op(
        {"strategy": "cursor", "cursor_param": "after", "cursor_path": "meta.next", "max_pages": 2}
    )
    _bind(remote, ops=[op])
    result = await execute_tool(TOOL, {})
    assert [i["id"] for i in result["data"]["items"]] == [1, 2, 3, 4]
    assert result["data"]["truncated"] is True
    assert len(remote.requests) == 2


@pytest.mark.asyncio
async def test_link_header_pagination():
    def route(r):
        page = int(parse_qs(r.url.query.decode()).get("page", ["1"])[0])
        headers = {}
        if page < 3:
            headers["Link"] = f'<{BASE}/users/alice/tickets?page={page + 1}>; rel="next"'
        return _ok({"data": ROWS[(page - 1) * 2 : page * 2]}, headers=headers)

    remote = Remote(route)
    _bind(remote, ops=[_page_op({"strategy": "link_header", "max_pages": 5})])
    result = await execute_tool(TOOL, {})
    assert [i["id"] for i in result["data"]["items"]] == [1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
async def test_odata_next_link_pagination_v4_and_v2():
    def route_v4(r):
        skip = int(parse_qs(r.url.query.decode()).get("$skip", ["0"])[0])
        body: dict[str, Any] = {"value": ROWS[skip : skip + 3]}
        if skip + 3 < len(ROWS):
            body["@odata.nextLink"] = f"{BASE}/users/alice/tickets?$skip={skip + 3}"
        return _ok(body)

    remote = Remote(route_v4)
    _bind(remote, ops=[_page_op({"strategy": "odata_next_link"}, root="value")])
    result = await execute_tool(TOOL, {})
    assert [i["id"] for i in result["data"]["items"]] == [1, 2, 3, 4, 5, 6, 7]

    def route_v2(r):
        first = "skiptoken" not in r.url.query.decode()
        body = {"d": {"results": ROWS[:2] if first else ROWS[2:3]}}
        if first:
            body["d"]["__next"] = "/v2/users/alice/tickets?$skiptoken=2"  # relative
        return _ok(body)

    remote2 = Remote(route_v2)
    _bind(
        remote2,
        ops=[
            _page_op(
                {"strategy": "odata_next_link", "next_link_path": "d.__next"}, root="d.results"
            )
        ],
    )
    result = await execute_tool(TOOL, {})
    assert [i["id"] for i in result["data"]["items"]] == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "link",
    [
        "https://evil.test/v2/users/alice/tickets?page=2",
        "https://api.example.test/other/secrets",
        "http://api.example.test/v2/users/alice/tickets",
        "https://api.example.test:8443/v2/users/alice/tickets",
    ],
)
async def test_next_links_outside_base_url_are_never_followed(link):
    remote = Remote(lambda r: _ok({"value": ROWS[:1], "@odata.nextLink": link}))
    _bind(remote, ops=[_page_op({"strategy": "odata_next_link"}, root="value")])
    result = await execute_tool(TOOL, {})
    assert result["error_kind"] == "redirect_refused"
    assert len(remote.requests) == 1


def test_within_base():
    assert within_base(BASE, f"{BASE}/x?y=1")
    assert within_base(BASE, BASE)
    assert not within_base(BASE, "https://api.example.test/v2x/a")
    assert not within_base(BASE, "https://api.example.test/v2/../admin")
    assert not within_base(BASE, "https://user:pw@api.example.test/v2/a")


# ---------------------------------------------------------------------------
# Throttling, redirects, SSRF
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_after_is_honoured_on_429():
    calls = {"n": 0}

    def route(r):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return _ok({"data": ROWS[:1]})

    remote = Remote(route)
    _bind(remote)
    result = await execute_tool(TOOL, {})
    assert result["data"]["returned"] == 1 and calls["n"] == 2


@pytest.mark.asyncio
async def test_429_with_long_retry_after_is_rate_limited_without_waiting():
    remote = Remote(lambda r: httpx.Response(429, headers={"Retry-After": "120"}))
    _bind(remote)
    started = time.perf_counter()
    result = await execute_tool(TOOL, {})
    assert result["error_kind"] == "rate_limited"
    assert time.perf_counter() - started < 1
    assert len(remote.requests) == 1


@pytest.mark.asyncio
async def test_persistent_429_gives_up_after_bounded_retries():
    remote = Remote(lambda r: httpx.Response(429, headers={"Retry-After": "0"}))
    _bind(remote)
    assert (await execute_tool(TOOL, {}))["error_kind"] == "rate_limited"
    assert len(remote.requests) == 3


@pytest.mark.asyncio
async def test_same_host_redirect_is_followed_foreign_redirect_refused():
    def route(r):
        if r.url.path.endswith("/tickets"):
            return httpx.Response(302, headers={"Location": "/v2/users/alice/tickets-v2"})
        return _ok({"data": ROWS[:1]})

    remote = Remote(route)
    _bind(remote)
    assert (await execute_tool(TOOL, {}))["data"]["returned"] == 1

    remote2 = Remote(lambda r: httpx.Response(302, headers={"Location": "https://evil.test/steal"}))
    _bind(remote2)
    assert (await execute_tool(TOOL, {}))["error_kind"] == "redirect_refused"
    assert [r.url.host for r in remote2.requests] == ["api.example.test"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.5",
        "169.254.169.254",
        "::1",
        "fd00::1",
        "100.64.0.1",
        "::ffff:192.168.1.1",
    ],
)
async def test_host_resolving_to_private_address_is_blocked(address):
    async def dns(host: str, port: int) -> list[str]:
        return ["93.184.216.34", address]

    remote = Remote(lambda r: _ok({"data": []}))
    _bind(remote, resolver=dns)
    assert (await execute_tool(TOOL, {}))["error_kind"] == "blocked_address"
    assert remote.requests == []


@pytest.mark.asyncio
async def test_allow_private_network_permits_on_prem_hosts():
    async def dns(host: str, port: int) -> list[str]:
        return ["10.20.30.40"]

    remote = Remote(lambda r: _ok({"data": ROWS[:1]}))
    _bind(
        remote,
        resolver=dns,
        base_url="http://sap.corp.internal:8000/sap/opu/odata",
        allow_private_network=True,
    )
    assert (await execute_tool(TOOL, {}))["data"]["returned"] == 1


@pytest.mark.asyncio
async def test_unresolvable_host_is_unavailable():
    async def dns(host: str, port: int) -> list[str]:
        raise OSError("no such host")

    remote = Remote(lambda r: _ok({"data": []}))
    _bind(remote, resolver=dns)
    assert (await execute_tool(TOOL, {}))["error_kind"] == "unavailable"


# ---------------------------------------------------------------------------
# Cross-user isolation and audit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_users_only_ever_get_their_own_records():
    data = {
        "alice": [{"id": 1, "title": "alice-laptop", "status": "open"}],
        "bob": [{"id": 2, "title": "bob-salary-query", "status": "open"}],
    }
    remote = Remote(lambda r: _ok({"data": data[r.url.path.split("/")[3]]}))
    recorder = TurnRecorder(
        operation="chat",
        tenant_id="t1",
        assistant_id="a",
        principal_ref="r",
        provider="p",
        model_name="m",
    )
    audit.bind(recorder)

    _bind(remote, principal=_principal("alice"))
    as_alice = await execute_tool(TOOL, {"user": "bob"})
    _bind(remote, principal=_principal("bob"))
    as_bob = await execute_tool(TOOL, {})

    assert [r.url.path for r in remote.requests] == [
        "/v2/users/alice/tickets",
        "/v2/users/bob/tickets",
    ]
    assert "bob" not in json.dumps(as_alice["data"])
    assert "alice" not in json.dumps(as_bob["data"])
    assert {
        "action": "connector_identity_argument_ignored",
        "connector": "tickets",
        "operation": "list_tickets",
        "fields": ["user"],
    } in recorder.guardrail_actions
    dumped = json.dumps(recorder.as_row(), default=str)
    for value in ("alice-laptop", "bob-salary", "key-0123456789abcdef"):
        assert value not in dumped


@pytest.mark.asyncio
async def test_label_denial_makes_no_request():
    remote = Remote(lambda r: _ok({"data": []}))
    turn = _bind(remote, principal=_principal(labels=("scope:development",)))
    assert turn.authorized_tool_names() == []
    assert (await execute_tool(TOOL, {}))["error_kind"] == "denied"
    assert remote.requests == []


def test_page_size_must_be_below_max_items():
    op = _op(
        response={"root": "data", "max_items": 20},
        pagination={"strategy": "offset", "page_size": 20, "page_size_param": "limit"},
    )
    with pytest.raises(ValidationError, match="no second page"):
        ConnectorConfig.model_validate(_connector([op]))
