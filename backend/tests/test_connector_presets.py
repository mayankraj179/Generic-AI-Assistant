"""Phase 2 presets (Jira Cloud, Jira Data Center, ServiceNow, Salesforce) and
the generic HRMS template, against mocks of each system's API.

Each mock filters by the identity in the query it receives, the way the real
system would, so the cross-user tests show that a user only ever gets
records scoped to their own verified identity, whatever the model sends.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
import yaml
from pydantic import ValidationError

from app.config.connectors import ConnectorConfig
from app.connectors.auth import ConnectorCallError, reset_auth_cache
from app.connectors.runtime import TurnConnectors, bind_turn
from app.connectors.spec import OperationSpec, TemplateSpec
from app.connectors.templates import literal, render
from app.core.principal import PrincipalContext
from app.observability import audit
from app.observability.audit import TurnRecorder
from app.tools.builtin import execute_tool, tool_definition

LABELS = ["role:employee"]
ALICE, BOB = "alice@example.test", "bob@example.test"
HRMS_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "configs/examples/connectors/hrms_generic.yaml"
)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    reset_auth_cache()
    for name, value in {
        "P_USER": "integration@example.test",
        "P_TOKEN": "api-token-0123456789",
        "P_PAT": "pat-0123456789abcdef",
        "P_CLIENT_ID": "client-abc",
        "P_CLIENT_SECRET": "client-secret-0123456789",
    }.items():
        monkeypatch.setenv(name, value)
    yield
    reset_auth_cache()


async def public_dns(host: str, port: int) -> list[str]:
    return ["93.184.216.34"]


def _principal(email: str = ALICE, verified: bool = True) -> PrincipalContext:
    return PrincipalContext(
        tenant_id="t1",
        principal_id=f"sub-{email}",
        labels=frozenset({"role:employee"}),
        clearance=1,
        claims={"email": email, "email_verified": "true" if verified else "false"},
    )


class Remote:
    def __init__(self, route) -> None:
        self.route = route
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.route(request)

    def query(self, index: int = -1) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(self.requests[index].url.query.decode()).items()}


def _bind(remote: Remote, connector: dict, principal: PrincipalContext | None = None):
    config = ConnectorConfig.model_validate(connector)
    turn = TurnConnectors(
        principal or _principal(),
        [config],
        transport=httpx.MockTransport(remote.handler),
        resolver=public_dns,
    )
    bind_turn(turn)
    return turn


def _ops(*names: str) -> list[dict]:
    return [{"name": n, "required_labels": LABELS} for n in names]


# ---------------------------------------------------------------------------
# Template escaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("language", "value", "expected"),
    [
        ("jql", 'a" OR project = HR OR x = "b', '"a\\" OR project = HR OR x = \\"b"'),
        ("jql", "back\\slash", '"back\\\\slash"'),
        ("soql", "x' OR Name != '", "'x\\' OR Name != \\''"),
        ("soql", 'q"uote\\', "'q\\\"uote\\\\'"),
        ("odata", "O'Brien", "'O''Brien'"),
        ("servicenow", "alice@example.test", "alice@example.test"),
        ("jql", 42, "42"),
        ("soql", True, "true"),
    ],
)
def test_literals_are_escaped(language, value, expected):
    assert literal(value, language) == expected


@pytest.mark.parametrize("language", ["jql", "soql", "odata", "servicenow"])
def test_control_characters_rejected(language):
    with pytest.raises(ConnectorCallError, match="control character"):
        literal("a\nORDER BY x", language)


def test_servicenow_caret_rejected():
    with pytest.raises(ConnectorCallError, match=r"\^"):
        literal("x^ORcaller_id.email=bob@example.test", "servicenow")


def test_optional_parts_are_dropped_when_unset():
    template = TemplateSpec.model_validate(
        {"language": "jql", "parts": ["a = {x}", "AND b = {y}", "ORDER BY c"]}
    )
    assert render(template, {"x": "1"}) == 'a = "1" ORDER BY c'
    assert render(template, {"x": "1", "y": 2}) == 'a = "1" AND b = 2 ORDER BY c'


def _template_op(**overrides) -> dict:
    op = {
        "name": "op",
        "description": "d",
        "path": "/q",
        "parameters": {"user": {"in": "template", "identity": True}},
        "templates": {"q": {"language": "soql", "parts": ["SELECT Id FROM X WHERE E = {user}"]}},
    }
    op.update(overrides)
    return op


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"templates": {"q": {"language": "soql", "parts": ["DELETE FROM X"]}}}, "single SELECT"),
        (
            {
                "templates": {
                    "q": {"language": "soql", "parts": ["SELECT Id FROM X; SELECT Id FROM Y"]}
                }
            },
            "single SELECT",
        ),
        ({"templates": {"q": {"language": "jql", "parts": ["a = {nope}"]}}}, "not 'in: template'"),
        (
            {
                "parameters": {
                    "user": {"in": "template", "identity": True},
                    "extra": {"in": "template"},
                }
            },
            "not used by any template",
        ),
        (
            {"templates": {"q": {"in": "body", "language": "jql", "parts": ["a = {user}"]}}},
            "needs a POST",
        ),
        (
            {
                "parameters": {
                    "user": {"in": "template", "identity": True},
                    "k": {"in": "template", "pattern": "[A-Z]+"},
                }
            },
            "anchored",
        ),
    ],
)
def test_template_load_rules(overrides, message):
    with pytest.raises(ValidationError, match=message):
        OperationSpec.model_validate(_template_op(**overrides))


def test_preset_auth_methods_are_enforced():
    with pytest.raises(ValidationError, match="supports auth methods"):
        ConnectorConfig.model_validate(_jira_cloud(auth={"method": "api_key", "key": "${P_TOKEN}"}))


# ---------------------------------------------------------------------------
# Jira
# ---------------------------------------------------------------------------

ISSUES = {
    ALICE: [
        {
            "key": "IT-1",
            "fields": {
                "summary": "Laptop",
                "status": {"name": "Open"},
                "assignee": {"displayName": "Alice"},
                "secret": "x",
            },
        }
    ],
    BOB: [
        {
            "key": "HR-9",
            "fields": {
                "summary": "Bob payroll",
                "status": {"name": "Open"},
                "assignee": {"displayName": "Bob"},
            },
        }
    ],
}
_JQL_USER = re.compile(r'assignee = "([^"\\]*)"')


def _jira_route(cursor: bool):
    def route(request: httpx.Request) -> httpx.Response:
        jql = parse_qs(request.url.query.decode())["jql"][0]
        user = _JQL_USER.search(jql).group(1)  # what the real system would filter on
        issues = ISSUES.get(user, [])
        key = re.search(r'key = "([^"]+)"', jql)
        if key:
            issues = [i for i in issues if i["key"] == key.group(1)]
        body: dict[str, Any] = {"issues": issues}
        if cursor:
            body["isLast"] = True
        else:
            body.update(startAt=0, maxResults=20, total=len(issues))
        return httpx.Response(200, json=body)

    return route


def _jira_cloud(**overrides) -> dict:
    data = {
        "name": "jira",
        "type": "jira_cloud",
        "display_name": "Jira",
        "base_url": "https://acme.atlassian.net",
        "auth": {"method": "basic", "username": "${P_USER}", "password": "${P_TOKEN}"},
        "operations": _ops("my_issues", "get_issue"),
    }
    data.update(overrides)
    return data


@pytest.mark.asyncio
async def test_jira_cloud_my_issues_is_scoped_to_the_caller():
    remote = Remote(_jira_route(cursor=True))
    _bind(remote, _jira_cloud())
    result = await execute_tool("jira__my_issues", {"status_category": "In Progress", "user": BOB})

    request = remote.requests[0]
    assert request.url.path == "/rest/api/3/search/jql"
    q = remote.query()
    assert q["jql"] == (
        f'(assignee = "{ALICE}" OR reporter = "{ALICE}") AND statusCategory = "In Progress" '
        "ORDER BY updated DESC"
    )
    assert q["maxResults"] == "20" and "summary" in q["fields"]
    assert request.headers["authorization"].startswith("Basic ")
    assert result["data"]["items"] == [
        {
            "key": "IT-1",
            "fields.summary": "Laptop",
            "fields.status.name": "Open",
            "fields.assignee.displayName": "Alice",
        }
    ]
    assert "user" not in json.dumps(tool_definition("jira__my_issues").input_schema)


@pytest.mark.asyncio
async def test_jira_text_injection_stays_inside_the_string_literal():
    remote = Remote(_jira_route(cursor=True))
    _bind(remote, _jira_cloud())
    await execute_tool("jira__my_issues", {"text": f'x" OR assignee = "{BOB}'})
    jql = remote.query()["jql"]
    assert jql.startswith(f'(assignee = "{ALICE}" OR reporter = "{ALICE}") AND text ~ "x\\"')
    assert _JQL_USER.findall(jql) == [ALICE]


@pytest.mark.asyncio
async def test_jira_get_issue_cannot_fetch_another_users_issue():
    remote = Remote(_jira_route(cursor=True))
    _bind(remote, _jira_cloud())
    result = await execute_tool("jira__get_issue", {"issue_key": "HR-9"})
    assert remote.query()["jql"] == (
        f'key = "HR-9" AND (assignee = "{ALICE}" OR reporter = "{ALICE}")'
    )
    assert result["data"]["items"] == []  # Bob's issue never matches Alice's scope

    bad = await execute_tool("jira__get_issue", {"issue_key": 'HR-9" OR key != "X'})
    assert bad["error_kind"] == "invalid_arguments"
    assert len(remote.requests) == 1


@pytest.mark.asyncio
async def test_jira_cloud_cursor_pagination():
    def route(request):
        token = parse_qs(request.url.query.decode()).get("nextPageToken", [None])[0]
        if token is None:
            return httpx.Response(200, json={"issues": ISSUES[ALICE], "nextPageToken": "p2"})
        return httpx.Response(200, json={"issues": [{"key": "IT-2", "fields": {}}], "isLast": True})

    remote = Remote(route)
    _bind(remote, _jira_cloud())
    result = await execute_tool("jira__my_issues", {})
    assert [i["key"] for i in result["data"]["items"]] == ["IT-1", "IT-2"]
    assert remote.query(1)["nextPageToken"] == "p2"
    assert remote.query(1)["jql"] == remote.query(0)["jql"]


@pytest.mark.asyncio
async def test_jira_rejected_query_is_not_unreachable():
    remote = Remote(lambda r: httpx.Response(400, json={"errorMessages": ["no such issue"]}))
    turn = _bind(remote, _jira_cloud())
    result = await execute_tool("jira__get_issue", {"issue_key": "IT-404"})
    assert result["error_kind"] == "rejected"
    assert "no such issue" not in result["error"]
    assert turn.unreachable_sources() == ()


@pytest.mark.asyncio
async def test_jira_data_center_offset_pagination_and_pat():
    pages = [ISSUES[ALICE] * 20, ISSUES[ALICE] * 3]

    def route(request):
        start = int(parse_qs(request.url.query.decode()).get("startAt", ["0"])[0])
        return httpx.Response(
            200, json={"issues": pages[0 if start == 0 else 1], "startAt": start, "total": 23}
        )

    remote = Remote(route)
    _bind(
        remote,
        _jira_cloud(
            type="jira_data_center",
            base_url="https://jira.acme.test/jira",
            auth={"method": "bearer", "token": "${P_PAT}"},
        ),
    )
    result = await execute_tool("jira__my_issues", {})
    assert remote.requests[0].url.path == "/jira/rest/api/2/search"
    assert remote.requests[0].headers["authorization"] == "Bearer pat-0123456789abcdef"
    assert remote.query(1)["startAt"] == "20"
    assert result["data"]["returned"] == 23 and result["data"]["truncated"] is False


# ---------------------------------------------------------------------------
# ServiceNow
# ---------------------------------------------------------------------------

INCIDENTS = {
    ALICE: [
        {
            "number": "INC0010001",
            "short_description": "VPN down",
            "state": "In Progress",
            "caller_id": "Alice",
            "sys_id": "abc",
        }
    ],
    BOB: [{"number": "INC0010002", "short_description": "Bob's HR complaint", "state": "New"}],
}


def _snow_route(request: httpx.Request) -> httpx.Response:
    query = parse_qs(request.url.query.decode())["sysparm_query"][0]
    clauses = query.split("^")
    email = next(
        c.split("=", 1)[1]
        for c in clauses
        if c.startswith(("caller_id.email=", "requested_for.email="))
    )
    rows = INCIDENTS.get(email, [])
    number = next((c.split("=", 1)[1] for c in clauses if c.startswith("number=")), None)
    if number:
        rows = [r for r in rows if r["number"] == number]
    return httpx.Response(200, json={"result": rows})


def _servicenow(**overrides) -> dict:
    data = {
        "name": "snow",
        "type": "servicenow",
        "display_name": "ServiceNow",
        "base_url": "https://acme.service-now.com",
        "auth": {"method": "basic", "username": "${P_USER}", "password": "${P_TOKEN}"},
        "operations": _ops("my_incidents", "get_incident", "my_requests", "get_request"),
    }
    data.update(overrides)
    return data


@pytest.mark.asyncio
async def test_servicenow_my_incidents_scoped_and_display_values():
    remote = Remote(_snow_route)
    _bind(remote, _servicenow())
    result = await execute_tool("snow__my_incidents", {"active": True, "user": BOB})
    q = remote.query()
    assert remote.requests[0].url.path == "/api/now/table/incident"
    assert q["sysparm_query"] == f"caller_id.email={ALICE}^active=true^ORDERBYDESCsys_updated_on"
    assert q["sysparm_display_value"] == "true" and q["sysparm_limit"] == "20"
    assert [r["number"] for r in result["data"]["items"]] == ["INC0010001"]
    assert "sys_id" not in json.dumps(result)


@pytest.mark.asyncio
async def test_servicenow_requests_use_requested_for():
    remote = Remote(_snow_route)
    _bind(remote, _servicenow())
    await execute_tool("snow__my_requests", {})
    assert remote.requests[0].url.path == "/api/now/table/sc_request"
    assert remote.query()["sysparm_query"] == (
        f"requested_for.email={ALICE}^ORDERBYDESCsys_updated_on"
    )


@pytest.mark.asyncio
async def test_servicenow_query_injection_never_reaches_the_system():
    remote = Remote(_snow_route)
    _bind(remote, _servicenow())
    attempt = await execute_tool(
        "snow__get_incident", {"number": f"INC0010002^ORcaller_id.email={BOB}"}
    )
    assert attempt["error_kind"] == "invalid_arguments"
    other_users = await execute_tool("snow__get_incident", {"number": "INC0010002"})
    assert other_users["data"]["items"] == []
    assert remote.query()["sysparm_query"] == f"number=INC0010002^caller_id.email={ALICE}"
    assert len(remote.requests) == 1


@pytest.mark.asyncio
async def test_servicenow_identity_with_caret_is_refused():
    remote = Remote(_snow_route)
    _bind(remote, _servicenow(), principal=_principal(f"x^ORcaller_id.email={BOB}"))
    assert (await execute_tool("snow__my_incidents", {}))["error_kind"] == "invalid_arguments"
    assert remote.requests == []


@pytest.mark.asyncio
async def test_servicenow_offset_pagination_and_oauth_post():
    rows = [{"number": f"INC00100{i:02d}"} for i in range(25)]
    issued: list[str] = []

    def route(request):
        if request.url.path == "/oauth_token.do":
            form = parse_qs(request.content.decode())
            assert form["client_secret"] == ["client-secret-0123456789"]
            issued.append("tok-0123456789")
            return httpx.Response(200, json={"access_token": "tok-0123456789", "expires_in": 1800})
        offset = int(parse_qs(request.url.query.decode()).get("sysparm_offset", ["0"])[0])
        return httpx.Response(200, json={"result": rows[offset : offset + 20]})

    remote = Remote(route)
    auth = {
        "method": "oauth_client_credentials",
        "token_url": "https://acme.service-now.com/oauth_token.do",
        "client_id": "${P_CLIENT_ID}",
        "client_secret": "${P_CLIENT_SECRET}",
        "client_auth": "post",
    }
    _bind(remote, _servicenow(auth=auth))
    result = await execute_tool("snow__my_incidents", {})
    assert result["data"]["returned"] == 25 and result["data"]["truncated"] is False
    assert remote.query(2)["sysparm_offset"] == "20"
    assert len(issued) == 1


# ---------------------------------------------------------------------------
# Salesforce
# ---------------------------------------------------------------------------

CASES = {
    ALICE: [
        {
            "attributes": {"type": "Case"},
            "CaseNumber": "00001026",
            "Subject": "Login",
            "Status": "New",
            "Priority": "High",
            "OwnerId": "005A",
        }
    ],
    BOB: [
        {
            "attributes": {"type": "Case"},
            "CaseNumber": "00001099",
            "Subject": "Bob's deal",
            "Status": "New",
            "Priority": "Low",
        }
    ],
}
_SOQL_EMAIL = re.compile(r"WHERE Email = '((?:[^'\\]|\\.)*)'")
SF_BASE = "https://acme.my.salesforce.com/services/data/v62.0"


def _sf_route(request: httpx.Request) -> httpx.Response:
    if "/query/" in request.url.path:  # a nextRecordsUrl page
        return httpx.Response(200, json={"done": True, "records": []})
    soql = parse_qs(request.url.query.decode())["q"][0]
    emails = _SOQL_EMAIL.findall(soql)
    rows = CASES.get(emails[0], []) if len(emails) == 1 else []
    number = re.search(r"CaseNumber = '([0-9]+)'", soql)
    if number:
        rows = [r for r in rows if r["CaseNumber"] == number.group(1)]
    return httpx.Response(200, json={"totalSize": len(rows), "done": True, "records": rows})


def _salesforce(**overrides) -> dict:
    data = {
        "name": "sfdc",
        "type": "salesforce",
        "display_name": "Salesforce",
        "base_url": SF_BASE,
        "auth": {
            "method": "oauth_client_credentials",
            "token_url": "https://acme.my.salesforce.com/services/oauth2/token",
            "client_id": "${P_CLIENT_ID}",
            "client_secret": "${P_CLIENT_SECRET}",
            "client_auth": "post",
        },
        "operations": _ops("my_open_cases", "get_case", "my_open_opportunities"),
    }
    data.update(overrides)
    return data


def _with_token(route):
    def wrapped(request):
        if request.url.path.endswith("/oauth2/token"):
            return httpx.Response(
                200,
                json={"access_token": "sf-token-0123456789", "instance_url": "https://evil.test"},
            )
        assert request.headers["authorization"] == "Bearer sf-token-0123456789"
        return route(request)

    return wrapped


@pytest.mark.asyncio
async def test_salesforce_soql_is_the_reviewed_template_scoped_to_the_caller():
    remote = Remote(_with_token(_sf_route))
    _bind(remote, _salesforce())
    result = await execute_tool("sfdc__my_open_cases", {"priority": "High", "user": BOB})
    request = remote.requests[-1]
    assert request.url.host == "acme.my.salesforce.com"  # never the token's instance_url
    assert request.url.path == "/services/data/v62.0/query"
    assert remote.query()["q"] == (
        "SELECT CaseNumber, Subject, Status, Priority, Origin, LastModifiedDate FROM Case "
        f"WHERE IsClosed = false AND OwnerId IN (SELECT Id FROM User WHERE Email = '{ALICE}') "
        "AND Priority = 'High' ORDER BY LastModifiedDate DESC LIMIT 50"
    )
    assert result["data"]["items"] == [
        {"CaseNumber": "00001026", "Subject": "Login", "Status": "New", "Priority": "High"}
    ]


@pytest.mark.asyncio
async def test_salesforce_soql_injection_is_escaped():
    remote = Remote(_with_token(_sf_route))
    _bind(remote, _salesforce())
    await execute_tool(
        "sfdc__my_open_opportunities",
        {"stage": f"x' OR OwnerId IN (SELECT Id FROM User WHERE Email = '{BOB}"},
    )
    soql = remote.query()["q"]
    assert "StageName = 'x\\' OR OwnerId IN (SELECT Id FROM User WHERE Email = \\'" in soql
    assert _SOQL_EMAIL.findall(soql)[0] == ALICE


@pytest.mark.asyncio
async def test_salesforce_get_case_scoped_and_pattern_checked():
    remote = Remote(_with_token(_sf_route))
    _bind(remote, _salesforce())
    other = await execute_tool("sfdc__get_case", {"case_number": "00001099"})
    assert other["data"]["items"] == []
    bad = await execute_tool("sfdc__get_case", {"case_number": "1' OR Id != '"})
    assert bad["error_kind"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_salesforce_next_records_url_pagination_and_foreign_link_refused():
    def route(request):
        if "/query/" in request.url.path:
            return httpx.Response(200, json={"done": True, "records": CASES[ALICE]})
        return httpx.Response(
            200,
            json={
                "done": False,
                "records": CASES[ALICE],
                "nextRecordsUrl": "/services/data/v62.0/query/01gD0000002HU6KIAW-2000",
            },
        )

    remote = Remote(_with_token(route))
    _bind(remote, _salesforce())
    result = await execute_tool("sfdc__my_open_cases", {})
    assert result["data"]["returned"] == 2
    assert remote.requests[-1].url.path == "/services/data/v62.0/query/01gD0000002HU6KIAW-2000"

    remote2 = Remote(
        _with_token(
            lambda r: httpx.Response(
                200,
                json={
                    "done": False,
                    "records": [],
                    "nextRecordsUrl": "https://evil.test/services/data/x",
                },
            )
        )
    )
    _bind(remote2, _salesforce())
    assert (await execute_tool("sfdc__my_open_cases", {}))["error_kind"] == "redirect_refused"


# ---------------------------------------------------------------------------
# Generic HRMS template
# ---------------------------------------------------------------------------

PROFILES = {
    ALICE: {
        "display_name": "Alice",
        "job_title": "Engineer",
        "department": "Eng",
        "manager": {"display_name": "Mgr", "email": "mgr@example.test"},
        "salary": 123456,
        "national_id": "ID-1",
    },
    BOB: {"display_name": "Bob", "job_title": "Analyst", "salary": 99999},
}


def _hrms_route(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/oauth/token":
        return httpx.Response(200, json={"access_token": "hr-token-0123456789", "expires_in": 300})
    parts = request.url.path.split("/")  # /api/v1/employees/<id>[/leave-balance]
    employee = parts[4]
    if len(parts) == 6:
        return httpx.Response(
            200,
            json={
                "as_of": "2026-10-12",
                "employee_id": employee,
                "balances": [{"leave_type": "annual", "available_days": 7}],
            },
        )
    profile = PROFILES.get(employee)
    return httpx.Response(200, json=profile) if profile else httpx.Response(404)


def _hrms() -> dict:
    data = yaml.safe_load(HRMS_TEMPLATE.read_text(encoding="utf-8"))
    data["base_url"] = "https://hrms.example.test/api/v1"
    data["auth"]["token_url"] = "https://hrms.example.test/oauth/token"
    data["auth"]["client_id"] = "${P_CLIENT_ID}"
    data["auth"]["client_secret"] = "${P_CLIENT_SECRET}"
    return data


def test_hrms_template_loads_as_is():
    config = ConnectorConfig.model_validate(
        yaml.safe_load(HRMS_TEMPLATE.read_text(encoding="utf-8"))
    )
    assert config.type == "rest" and config.identity.claim == "email"
    assert [op.name for op in config.operations] == ["get_leave_balance", "get_employee_profile"]


@pytest.mark.asyncio
async def test_hrms_profile_keeps_only_work_fields_for_the_caller():
    remote = Remote(_hrms_route)
    _bind(remote, _hrms())
    result = await execute_tool("hrms__get_employee_profile", {"employee_id": BOB})
    assert remote.requests[-1].url.raw_path.decode() == "/api/v1/employees/alice%40example.test"
    data = result["data"]
    assert data["display_name"] == "Alice" and data["manager.display_name"] == "Mgr"
    assert "salary" not in json.dumps(data) and "national_id" not in json.dumps(data)


# ---------------------------------------------------------------------------
# Cross-user isolation across every preset, plus email_verified
# ---------------------------------------------------------------------------

CASES_PER_SYSTEM = [
    ("jira__my_issues", _jira_cloud, _jira_route(cursor=True), "Laptop", "Bob payroll"),
    ("snow__my_incidents", _servicenow, _snow_route, "VPN down", "Bob's HR complaint"),
    ("sfdc__my_open_cases", _salesforce, _with_token(_sf_route), "Login", "Bob's deal"),
]


@pytest.mark.parametrize(("tool", "connector", "route", "alice_sees", "bob_sees"), CASES_PER_SYSTEM)
@pytest.mark.asyncio
async def test_each_user_only_gets_their_own_records(tool, connector, route, alice_sees, bob_sees):
    recorder = TurnRecorder(
        operation="chat",
        tenant_id="t1",
        assistant_id="a",
        principal_ref="r",
        provider="p",
        model_name="m",
    )
    audit.bind(recorder)
    remote = Remote(route)

    _bind(remote, connector(), principal=_principal(ALICE))
    as_alice = json.dumps(
        await execute_tool(tool.split("__")[0] + "__" + tool.split("__")[1], {"user": BOB})
    )
    _bind(remote, connector(), principal=_principal(BOB))
    as_bob = json.dumps(await execute_tool(tool, {}))

    assert alice_sees in as_alice and bob_sees not in as_alice
    assert bob_sees in as_bob and alice_sees not in as_bob
    dumped = json.dumps(recorder.as_row(), default=str)
    for value in (alice_sees, bob_sees, ALICE, BOB):
        assert value not in dumped


@pytest.mark.parametrize(("tool", "connector", "route", "_a", "_b"), CASES_PER_SYSTEM)
@pytest.mark.asyncio
async def test_unverified_email_gets_nothing(tool, connector, route, _a, _b):
    remote = Remote(route)
    _bind(remote, connector(), principal=_principal(ALICE, verified=False))
    assert (await execute_tool(tool, {}))["error_kind"] == "identity_unverified"
    assert remote.requests == []
