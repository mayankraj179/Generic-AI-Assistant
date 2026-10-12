"""Phase 3: the microsoft_graph preset (Graph v1.0 + SharePoint), against a
mock of Graph and the Entra ID token endpoint.

Covers Entra client credentials, settings injection (site/list/drive ids the
model can't choose), OData paging, Graph throttling (Retry-After and backoff),
and that users only ever get their own profile, manager, OneDrive items and
list rows.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, unquote

import httpx
import pytest
from pydantic import ValidationError

import app.connectors.http as connector_http
from app.config.connectors import ConnectorConfig
from app.connectors.auth import reset_auth_cache
from app.connectors.runtime import TurnConnectors, bind_turn
from app.core.principal import PrincipalContext
from app.observability import audit
from app.observability.audit import TurnRecorder
from app.tools.builtin import execute_tool, tool_definition

TENANT = "11111111-2222-3333-4444-555555555555"
TOKEN_URL = f"https://login.microsoftonline.com/{TENANT}/oauth2/v2.0/token"
GRAPH = "https://graph.microsoft.com/v1.0"
SITE_ID = (
    "contoso.sharepoint.com,2c712604-1370-44e7-a1f5-426573a2b7a2,"
    "2d2244a5-7a59-4d79-8d6f-3fb6a1bd6c1b"
)
LIST_ID = "d7689e2b-941a-4cd3-bb24-55cddee54294"
DRIVE_ID = "b!BhRxLHATQ0Sh9UJlc6K3oqVEIi1ZenlNjW8_tqG9bBs"
FOLDER_ID = "01BYE5RZ6QN3ZWBTUFOFD3GSPGOHDJD36K"
ALICE, BOB = "alice@contoso.test", "bob@contoso.test"

USERS = {
    ALICE: {
        "displayName": "Alice A",
        "jobTitle": "Engineer",
        "department": "IT",
        "mail": ALICE,
        "officeLocation": "Pune",
        "employeeId": "E100",
        "passwordProfile": "never-returned",
    },
    BOB: {
        "displayName": "Bob B",
        "jobTitle": "Payroll Lead",
        "department": "Finance",
        "mail": BOB,
        "officeLocation": "Mumbai",
        "employeeId": "E200",
    },
}
MANAGERS = {
    ALICE: {"displayName": "Mgr Alice", "mail": "mgr.a@contoso.test"},
    BOB: {"displayName": "Mgr Bob", "mail": "mgr.b@contoso.test"},
}
DRIVES = {
    ALICE: [
        {
            "id": "A1",
            "name": "alice-cv.docx",
            "webUrl": "https://x/a1",
            "size": 10,
            "file": {"mimeType": "application/msword"},
        }
    ],
    BOB: [
        {
            "id": "B1",
            "name": "bob-salary.xlsx",
            "webUrl": "https://x/b1",
            "size": 20,
            "file": {"mimeType": "application/vnd.ms-excel"},
        }
    ],
}
LIST_ROWS = [
    {"id": "1", "webUrl": "https://x/1", "fields": {"Title": "Laptop", "EmployeeEmail": ALICE}},
    {"id": "2", "webUrl": "https://x/2", "fields": {"Title": "Phone", "EmployeeEmail": BOB}},
]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    reset_auth_cache()
    monkeypatch.setenv("G_CLIENT_ID", "app-client-id")
    monkeypatch.setenv("G_CLIENT_SECRET", "app-client-secret-0123456789")
    monkeypatch.setattr(connector_http, "BACKOFF_BASE_SECONDS", 0.0)
    yield
    reset_auth_cache()


async def public_dns(host: str, port: int) -> list[str]:
    return ["20.190.160.1"]


def _principal(email: str = ALICE, verified: bool = True, labels=("role:employee",)):
    return PrincipalContext(
        tenant_id="t1",
        principal_id=f"oid-{email}",
        labels=frozenset(labels),
        clearance=1,
        claims={"email": email, "email_verified": "true" if verified else "false"},
    )


def _graph(**overrides) -> dict:
    data = {
        "name": "m365",
        "type": "microsoft_graph",
        "display_name": "Microsoft 365",
        "base_url": GRAPH,
        "auth": {
            "method": "oauth_client_credentials",
            "token_url": TOKEN_URL,
            "client_id": "${G_CLIENT_ID}",
            "client_secret": "${G_CLIENT_SECRET}",
            "scope": "https://graph.microsoft.com/.default",
            "client_auth": "post",
        },
        "settings": {
            "site_id": SITE_ID,
            "list_id": LIST_ID,
            "owner_column": "EmployeeEmail",
            "drive_id": DRIVE_ID,
            "folder_id": FOLDER_ID,
        },
        "operations": [
            {"name": n, "required_labels": ["role:employee"]}
            for n in (
                "my_profile",
                "my_manager",
                "my_drive_items",
                "list_items",
                "my_list_items",
                "library_folder_items",
                "get_library_item",
            )
        ],
    }
    data.update(overrides)
    return data


class Graph:
    """Mock Graph + Entra. Filters by the user in the path / $filter, as
    Graph would."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.tokens: list[str] = []
        self.token_forms: list[dict[str, list[str]]] = []
        self.throttle: list[httpx.Response] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "login.microsoftonline.com":
            self.token_forms.append(parse_qs(request.content.decode()))
            token = f"graph-token-{len(self.tokens)}-0123456789"
            self.tokens.append(token)
            return httpx.Response(
                200, json={"token_type": "Bearer", "expires_in": 3599, "access_token": token}
            )
        if request.headers.get("authorization", "").removeprefix("Bearer ") not in self.tokens:
            return httpx.Response(401, json={"error": {"code": "InvalidAuthenticationToken"}})
        if self.throttle:
            return self.throttle.pop(0)
        return self.route(request)

    def route(self, request: httpx.Request) -> httpx.Response:
        path = unquote(request.url.raw_path.decode().split("?")[0]).removeprefix("/v1.0")
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        parts = path.split("/")
        if parts[1] == "users":
            user = parts[2]
            if user not in USERS:
                return httpx.Response(404, json={"error": {"code": "Request_ResourceNotFound"}})
            if len(parts) == 3:
                return httpx.Response(200, json=USERS[user])
            if parts[3] == "manager":
                return httpx.Response(200, json=MANAGERS[user])
            return self._paged(DRIVES[user] * 30, query, path)
        if parts[1] == "sites":
            rows = LIST_ROWS
            flt = query.get("$filter")
            if flt:
                assert flt.startswith("fields/EmployeeEmail eq '")
                wanted = flt.split(" eq ", 1)[1][1:-1].replace("''", "'")
                rows = [r for r in rows if r["fields"]["EmployeeEmail"] == wanted]
            return self._paged(rows, query, path)
        if parts[1] == "drives":
            if parts[-1] == "children":
                return self._paged([{"id": "ITEM1234567", "name": "Policy.pdf"}], query, path)
            return httpx.Response(200, json={"id": parts[-1], "name": "Policy.pdf", "size": 5})
        return httpx.Response(404)

    @staticmethod
    def _paged(rows: list[Any], query: dict[str, str], path: str) -> httpx.Response:
        top = int(query.get("$top", "25"))
        skip = int(query.get("$skiptoken", "0"))
        body: dict[str, Any] = {"value": rows[skip : skip + top]}
        if skip + top < len(rows):
            body["@odata.nextLink"] = f"{GRAPH}{path}?$top={top}&$skiptoken={skip + top}"
        return httpx.Response(200, json=body)


def _bind(graph: Graph, principal=None, **overrides) -> TurnConnectors:
    turn = TurnConnectors(
        principal or _principal(),
        [ConnectorConfig.model_validate(_graph(**overrides))],
        transport=httpx.MockTransport(graph.handler),
        resolver=public_dns,
    )
    bind_turn(turn)
    return turn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_missing_setting_fails_config_loading():
    settings = {
        "site_id": SITE_ID,
        "owner_column": "EmployeeEmail",
        "drive_id": DRIVE_ID,
        "folder_id": FOLDER_ID,
    }
    with pytest.raises(ValidationError, match="needs settings.list_id"):
        ConnectorConfig.model_validate(_graph(settings=settings))


def test_field_name_setting_must_be_an_identifier():
    settings = {**_graph()["settings"], "owner_column": "EmployeeEmail ne null or Title"}
    with pytest.raises(ValidationError, match="plain identifier"):
        ConnectorConfig.model_validate(_graph(settings=settings))


def test_only_settings_for_enabled_operations_are_required():
    ConnectorConfig.model_validate(
        _graph(settings={}, operations=[{"name": "my_profile", "required_labels": ["r:x"]}])
    )


def test_graph_preset_is_client_credentials_only():
    with pytest.raises(ValidationError, match="supports auth methods"):
        ConnectorConfig.model_validate(
            _graph(auth={"method": "bearer", "token": "${G_CLIENT_SECRET}"})
        )


def test_settings_and_identity_are_hidden_from_the_model():
    _bind(Graph())
    for op in ("my_profile", "list_items", "my_list_items", "library_folder_items"):
        schema = json.dumps(tool_definition(f"m365__{op}").input_schema)
        for hidden in ("user", "site_id", "list_id", "owner_column", "drive_id", "folder_id"):
            assert hidden not in schema


# ---------------------------------------------------------------------------
# Entra ID client credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entra_client_credentials_token_is_requested_once_and_reused():
    graph = Graph()
    for _ in range(3):
        _bind(graph)
        assert "data" in await execute_tool("m365__my_profile", {})
    assert len(graph.tokens) == 1
    (form,) = graph.token_forms
    assert form == {
        "grant_type": ["client_credentials"],
        "scope": ["https://graph.microsoft.com/.default"],
        "client_id": ["app-client-id"],
        "client_secret": ["app-client-secret-0123456789"],
    }
    token_request = graph.requests[0]
    assert token_request.url.path == f"/{TENANT}/oauth2/v2.0/token"
    assert "authorization" not in token_request.headers


@pytest.mark.asyncio
async def test_expired_graph_token_is_refreshed_after_401():
    graph = Graph()
    _bind(graph)
    await execute_tool("m365__my_profile", {})
    graph.tokens.clear()  # Graph no longer accepts the cached token
    assert "data" in await execute_tool("m365__my_profile", {})
    assert len(graph.token_forms) == 2


# ---------------------------------------------------------------------------
# The signed-in user's own data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_profile_and_manager_are_the_callers_own():
    graph = Graph()
    _bind(graph)
    profile = await execute_tool("m365__my_profile", {"user": BOB})
    manager = await execute_tool("m365__my_manager", {})

    paths = [r.url.raw_path.decode() for r in graph.requests[1:]]
    assert paths[0].startswith(f"/v1.0/users/{ALICE}?")
    assert paths[1].startswith(f"/v1.0/users/{ALICE}/manager?")
    assert parse_qs(graph.requests[1].url.query.decode())["$select"][0].startswith("displayName")
    assert profile["data"]["displayName"] == "Alice A"
    assert "passwordProfile" not in json.dumps(profile)
    assert manager["data"] == {"displayName": "Mgr Alice", "mail": "mgr.a@contoso.test"}


@pytest.mark.asyncio
async def test_my_drive_items_follows_odata_next_link_on_graph_only():
    graph = Graph()
    _bind(graph)
    result = await execute_tool("m365__my_drive_items", {})
    assert [i["name"] for i in result["data"]["items"]] == ["alice-cv.docx"] * 30
    assert result["data"]["truncated"] is False
    second = graph.requests[-1]
    assert second.url.host == "graph.microsoft.com"
    assert parse_qs(second.url.query.decode()) == {"$top": ["25"], "$skiptoken": ["25"]}
    assert "bob-salary" not in json.dumps(result)


@pytest.mark.asyncio
async def test_next_link_to_another_host_is_refused():
    graph = Graph()
    graph.route = lambda r: httpx.Response(
        200,
        json={  # type: ignore[method-assign]
            "value": [{"id": "x"}],
            "@odata.nextLink": "https://graph.microsoft.com.evil.test/v1.0/users/x/drive?$skiptoken=1",
        },
    )
    _bind(graph)
    assert (await execute_tool("m365__my_drive_items", {}))["error_kind"] == "redirect_refused"


@pytest.mark.asyncio
async def test_unknown_user_is_not_found():
    _bind(Graph(), principal=_principal("nobody@contoso.test"))
    assert (await execute_tool("m365__my_profile", {}))["error_kind"] == "not_found"


@pytest.mark.asyncio
async def test_unverified_email_gets_no_graph_call():
    graph = Graph()
    _bind(graph, principal=_principal(verified=False))
    assert (await execute_tool("m365__my_profile", {}))["error_kind"] == "identity_unverified"
    assert graph.requests == []


# ---------------------------------------------------------------------------
# SharePoint list and library
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_items_use_configured_site_and_list_whatever_the_model_sends():
    graph = Graph()
    _bind(graph)
    result = await execute_tool(
        "m365__list_items", {"site_id": "evil.sharepoint.com,1,2", "list_id": "x"}
    )
    request = graph.requests[-1]
    assert request.url.raw_path.decode().startswith(f"/v1.0/sites/{SITE_ID}/lists/{LIST_ID}/items")
    q = parse_qs(request.url.query.decode())
    assert q["expand"] == ["fields"] and q["$top"] == ["25"]
    assert [i["id"] for i in result["data"]["items"]] == ["1", "2"]


@pytest.mark.asyncio
async def test_my_list_items_filters_on_the_owner_column_with_escaping():
    graph = Graph()
    _bind(graph)
    result = await execute_tool("m365__my_list_items", {})
    request = graph.requests[-1]
    assert parse_qs(request.url.query.decode())["$filter"] == [f"fields/EmployeeEmail eq '{ALICE}'"]
    assert request.headers["prefer"] == "HonorNonIndexedQueriesWarningMayFailRandomly"
    assert [i["fields"]["Title"] for i in result["data"]["items"]] == ["Laptop"]

    _bind(graph, principal=_principal("o'brien@contoso.test"))
    await execute_tool("m365__my_list_items", {})
    assert parse_qs(graph.requests[-1].url.query.decode())["$filter"] == [
        "fields/EmployeeEmail eq 'o''brien@contoso.test'"
    ]


@pytest.mark.asyncio
async def test_library_metadata_uses_configured_drive_and_validates_item_id():
    graph = Graph()
    _bind(graph)
    listing = await execute_tool("m365__library_folder_items", {})
    assert (
        graph.requests[-1]
        .url.raw_path.decode()
        .startswith(f"/v1.0/drives/{DRIVE_ID}/items/{FOLDER_ID}/children")
    )
    assert listing["data"]["items"] == [{"id": "ITEM1234567", "name": "Policy.pdf"}]

    item = await execute_tool("m365__get_library_item", {"item_id": "ITEM1234567"})
    assert item["data"]["name"] == "Policy.pdf"
    before = len(graph.requests)
    for bad in ("../../users/bob", "root:/HR/salaries.xlsx:", "ITEM?x=1"):
        assert (await execute_tool("m365__get_library_item", {"item_id": bad}))[
            "error_kind"
        ] == "invalid_arguments"
    assert len(graph.requests) == before


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "throttle",
    [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(503, headers={"Retry-After": "0"}),
        httpx.Response(429),  # no Retry-After: exponential backoff
    ],
)
async def test_graph_throttling_is_retried(throttle):
    graph = Graph()
    graph.throttle = [throttle]
    _bind(graph)
    assert (await execute_tool("m365__my_profile", {}))["data"]["displayName"] == "Alice A"


@pytest.mark.asyncio
async def test_sustained_throttling_returns_rate_limited():
    graph = Graph()
    graph.throttle = [httpx.Response(429) for _ in range(5)]
    _bind(graph)
    assert (await execute_tool("m365__my_profile", {}))["error_kind"] == "rate_limited"
    assert len(graph.throttle) == 2  # 1 try + 2 retries consumed


# ---------------------------------------------------------------------------
# Cross-user isolation and labels
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "alice_sees", "bob_sees"),
    [
        ("m365__my_profile", "Engineer", "Payroll Lead"),
        ("m365__my_manager", "Mgr Alice", "Mgr Bob"),
        ("m365__my_drive_items", "alice-cv.docx", "bob-salary.xlsx"),
        ("m365__my_list_items", "Laptop", "Phone"),
    ],
)
async def test_each_user_only_gets_their_own_graph_data(tool, alice_sees, bob_sees):
    recorder = TurnRecorder(
        operation="chat",
        tenant_id="t1",
        assistant_id="a",
        principal_ref="r",
        provider="p",
        model_name="m",
    )
    audit.bind(recorder)
    graph = Graph()
    _bind(graph, principal=_principal(ALICE))
    as_alice = json.dumps(await execute_tool(tool, {"user": BOB}))
    _bind(graph, principal=_principal(BOB))
    as_bob = json.dumps(await execute_tool(tool, {"user": ALICE}))

    assert alice_sees in as_alice and bob_sees not in as_alice
    assert bob_sees in as_bob and alice_sees not in as_bob
    dumped = json.dumps(recorder.as_row(), default=str)
    for value in (alice_sees, bob_sees, ALICE, BOB, "graph-token"):
        assert value not in dumped


@pytest.mark.asyncio
async def test_missing_label_hides_graph_tools_and_makes_no_call():
    graph = Graph()
    turn = _bind(graph, principal=_principal(labels=("role:contractor",)))
    assert turn.authorized_tool_names() == []
    assert (await execute_tool("m365__my_profile", {}))["error_kind"] == "denied"
    assert graph.requests == []
