"""Phase 4: SAP presets (S/4HANA OData v2, SuccessFactors OData v2) against
mocks, plus the on-prem pieces: allow_private_network, sap-client and a
corporate CA bundle.

The mocks filter by the user named in $filter, as SAP would, so the
cross-user tests show each caller only ever gets their own documents.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from pydantic import ValidationError

import app.connectors.runtime as runtime
from app.config.connectors import ConnectorConfig
from app.connectors.auth import ConnectorCallError, reset_auth_cache
from app.connectors.runtime import TurnConnectors, bind_turn, tls_context
from app.core.principal import PrincipalContext
from app.observability import audit
from app.observability.audit import TurnRecorder
from app.tools.builtin import execute_tool, tool_definition

S4_BASE = "http://s4h.corp.internal:8000/sap/opu/odata/sap"
SF_BASE = "https://api4.successfactors.com/odata/v2"
LABELS = ["role:employee"]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    reset_auth_cache()
    monkeypatch.setenv("SAP_USER", "RFC_READER")
    monkeypatch.setenv("SAP_PASSWORD", "sap-password-0123456789")
    monkeypatch.setenv("SF_USER", "apiuser@ACMECORP")
    monkeypatch.setenv("SF_PASSWORD", "sf-password-0123456789")
    yield
    reset_auth_cache()


async def private_dns(host: str, port: int) -> list[str]:
    return ["10.20.30.40"]


async def public_dns(host: str, port: int) -> list[str]:
    return ["155.56.1.1"]


def _principal(
    sap_user: str = "JDOE",
    email: str = "jdoe@acme.test",
    verified: bool = True,
    labels=("role:employee",),
) -> PrincipalContext:
    return PrincipalContext(
        tenant_id="t1",
        principal_id=f"sub-{sap_user}",
        labels=frozenset(labels),
        clearance=1,
        claims={
            "sap_user": sap_user,
            "email": email,
            "email_verified": "true" if verified else "false",
        },
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


def _bind(remote: Remote, connector: dict, principal=None, resolver=private_dns):
    turn = TurnConnectors(
        principal or _principal(),
        [ConnectorConfig.model_validate(connector)],
        transport=httpx.MockTransport(remote.handler),
        resolver=resolver,
    )
    bind_turn(turn)
    return turn


# ---------------------------------------------------------------------------
# S/4HANA
# ---------------------------------------------------------------------------

PURCHASE_ORDERS = {
    "JDOE": [
        {
            "__metadata": {"uri": "x"},
            "PurchaseOrder": f"45000001{i:02d}",
            "Supplier": "17300001",
            "CompanyCode": "1010",
            "CreationDate": "/Date(1790000000000)/",
            "DocumentCurrency": "EUR",
            "PurchasingProcessingStatus": "05",
            "CreatedByUser": "JDOE",
        }
        for i in range(30)
    ],
    "MSMITH": [
        {
            "PurchaseOrder": "4500009999",
            "Supplier": "17300099",
            "CompanyCode": "1010",
            "CreatedByUser": "MSMITH",
            "DocumentCurrency": "USD",
        }
    ],
}
_S4_USER = re.compile(r"CreatedByUser eq '((?:[^']|'')*)'")


def _s4_route(request: httpx.Request) -> httpx.Response:
    q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
    assert q["sap-client"] == "100" and q["$format"] == "json"
    if request.url.path.endswith("/A_BusinessPartner"):
        return httpx.Response(
            200,
            json={
                "d": {
                    "results": [
                        {
                            "BusinessPartner": "17300001",
                            "BusinessPartnerFullName": "Acme Supplies",
                            "BusinessPartnerCategory": "2",
                            "IBAN": "never-returned",
                        }
                    ]
                }
            },
        )
    user = _S4_USER.search(q["$filter"]).group(1).replace("''", "'")
    rows = PURCHASE_ORDERS.get(user, [])
    number = re.search(r"PurchaseOrder eq '([0-9]+)'", q["$filter"])
    if number:
        rows = [r for r in rows if r["PurchaseOrder"] == number.group(1)]
    skip, top = int(q.get("$skip", "0")), int(q.get("$top", "1000"))
    return httpx.Response(200, json={"d": {"results": rows[skip : skip + top]}})


def _s4(**overrides) -> dict:
    data = {
        "name": "s4",
        "type": "sap_s4hana",
        "display_name": "SAP",
        "base_url": S4_BASE,
        "allow_private_network": True,
        "auth": {"method": "basic", "username": "${SAP_USER}", "password": "${SAP_PASSWORD}"},
        "identity": {"claim": "sap_user"},
        "settings": {"sap_client": "100"},
        "operations": [
            {"name": n, "required_labels": LABELS}
            for n in (
                "my_purchase_orders",
                "get_purchase_order",
                "my_purchase_requisitions",
                "get_business_partner",
            )
        ],
    }
    data.update(overrides)
    return data


def test_s4_requires_an_explicit_identity_claim():
    data = _s4()
    del data["identity"]
    with pytest.raises(ValidationError, match="set identity.claim"):
        ConnectorConfig.model_validate(data)


def test_s4_requires_sap_client_setting():
    with pytest.raises(ValidationError, match="needs settings.sap_client"):
        ConnectorConfig.model_validate(_s4(settings={}))


def test_on_prem_http_needs_allow_private_network():
    with pytest.raises(ValidationError, match="allow_private_network"):
        ConnectorConfig.model_validate(_s4(allow_private_network=False))


def test_s4_hidden_parameters_never_reach_the_model():
    _bind(Remote(_s4_route), _s4())
    schema = json.dumps(tool_definition("s4__my_purchase_orders").input_schema)
    assert "sap-client" not in schema and "user" not in schema
    assert "supplier" in schema


@pytest.mark.asyncio
async def test_s4_my_purchase_orders_on_prem_scoped_and_paged():
    remote = Remote(_s4_route)
    _bind(remote, _s4())
    result = await execute_tool("s4__my_purchase_orders", {"user": "MSMITH"})

    first = remote.requests[0]
    assert str(first.url).startswith(f"{S4_BASE}/API_PURCHASEORDER_PROCESS_SRV/A_PurchaseOrder?")
    assert first.headers["authorization"].startswith("Basic ")
    q = remote.query(0)
    assert q["$filter"] == "CreatedByUser eq 'JDOE'"
    assert q["$top"] == "25" and q["$orderby"] == "CreationDate desc"
    assert remote.query(1)["$skip"] == "25"
    items = result["data"]["items"]
    assert len(items) == 30 and result["data"]["truncated"] is False
    assert set(items[0]) == {
        "PurchaseOrder",
        "Supplier",
        "CompanyCode",
        "CreationDate",
        "DocumentCurrency",
        "PurchasingProcessingStatus",
    }
    assert "4500009999" not in json.dumps(result)


@pytest.mark.asyncio
async def test_s4_get_purchase_order_is_scoped_and_pattern_checked():
    remote = Remote(_s4_route)
    _bind(remote, _s4())
    other = await execute_tool("s4__get_purchase_order", {"purchase_order": "4500009999"})
    assert other["data"]["items"] == []
    assert remote.query()["$filter"] == "PurchaseOrder eq '4500009999' and CreatedByUser eq 'JDOE'"
    for bad in ("4500009999' or CreatedByUser ne 'X", "45000099"):
        result = await execute_tool("s4__get_purchase_order", {"purchase_order": bad})
        assert result["error_kind"] == "invalid_arguments"
    assert len(remote.requests) == 1


@pytest.mark.asyncio
async def test_s4_odata_literal_escaping_for_identity():
    remote = Remote(_s4_route)
    _bind(remote, _s4(), principal=_principal(sap_user="O'NEIL"))
    await execute_tool("s4__my_purchase_orders", {})
    assert remote.query()["$filter"] == "CreatedByUser eq 'O''NEIL'"


@pytest.mark.asyncio
async def test_s4_business_partner_is_projected():
    remote = Remote(_s4_route)
    _bind(remote, _s4())
    result = await execute_tool("s4__get_business_partner", {"business_partner": "17300001"})
    assert result["data"]["items"][0]["BusinessPartnerFullName"] == "Acme Supplies"
    assert "IBAN" not in json.dumps(result)


@pytest.mark.asyncio
async def test_sap_user_claim_does_not_need_email_verified():
    remote = Remote(_s4_route)
    _bind(remote, _s4(), principal=_principal(verified=False))
    assert "data" in await execute_tool("s4__my_purchase_orders", {})


@pytest.mark.asyncio
async def test_s4_cross_user_isolation():
    recorder = TurnRecorder(
        operation="chat",
        tenant_id="t1",
        assistant_id="a",
        principal_ref="r",
        provider="p",
        model_name="m",
    )
    audit.bind(recorder)
    remote = Remote(_s4_route)
    _bind(remote, _s4(), principal=_principal("JDOE"))
    as_jdoe = json.dumps(await execute_tool("s4__my_purchase_orders", {"user": "MSMITH"}))
    _bind(remote, _s4(), principal=_principal("MSMITH", "msmith@acme.test"))
    as_msmith = json.dumps(await execute_tool("s4__my_purchase_orders", {"user": "JDOE"}))
    assert "4500000100" in as_jdoe and "4500009999" not in as_jdoe
    assert "4500009999" in as_msmith and "4500000100" not in as_msmith
    dumped = json.dumps(recorder.as_row(), default=str)
    for value in ("4500000100", "4500009999", "JDOE", "MSMITH", "sap-password"):
        assert value not in dumped


# ---------------------------------------------------------------------------
# On-prem TLS: a corporate CA bundle, never verify=False
# ---------------------------------------------------------------------------


def _ca_pem(tmp_path: Path) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Acme Corp Root CA")])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    path = tmp_path / "corp-ca.pem"
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return path


def test_tls_context_trusts_the_bundle_and_still_verifies(tmp_path):
    context = tls_context(str(_ca_pem(tmp_path)))
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
    subjects = [dict(x[0] for x in c["subject"]) for c in context.get_ca_certs()]
    assert {"commonName": "Acme Corp Root CA"} in subjects


def test_tls_context_missing_or_invalid_bundle_is_not_configured(tmp_path):
    with pytest.raises(ConnectorCallError) as missing:
        tls_context(str(tmp_path / "nope.pem"))
    assert missing.value.kind == "not_configured"
    bad = tmp_path / "bad.pem"
    bad.write_text("not a certificate")
    with pytest.raises(ConnectorCallError) as invalid:
        tls_context(str(bad))
    assert invalid.value.kind == "not_configured"


@pytest.mark.asyncio
async def test_connector_with_ca_bundle_gets_its_own_verifying_client(tmp_path, monkeypatch):
    created: list[dict[str, Any]] = []

    class FakeClient:
        is_closed = False

        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr(runtime.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(runtime, "_shared_clients", type(runtime._shared_clients)())
    bundle = str(_ca_pem(tmp_path))
    default = runtime._shared_client(None)
    corp = runtime._shared_client(bundle)
    assert runtime._shared_client(bundle) is corp and default is not corp
    assert created[0]["verify"] is True
    assert isinstance(created[1]["verify"], ssl.SSLContext)
    assert all(kw["follow_redirects"] is False for kw in created)


def test_ca_bundle_is_a_config_field():
    config = ConnectorConfig.model_validate(_s4(tls_ca_bundle="certs/corp-ca.pem"))
    assert config.tls_ca_bundle == "certs/corp-ca.pem"


# ---------------------------------------------------------------------------
# SuccessFactors
# ---------------------------------------------------------------------------

SF_USERS = {
    "jdoe@acme.test": {
        "userId": "100001",
        "defaultFullName": "Jane Doe",
        "title": "Engineer",
        "department": "IT",
        "email": "jdoe@acme.test",
        "salary": "never",
        "manager": {"defaultFullName": "Max Boss", "email": "max@acme.test", "title": "Head of IT"},
    },
    "msmith@acme.test": {
        "userId": "100002",
        "defaultFullName": "Mark Smith",
        "title": "Payroll",
        "department": "Finance",
        "email": "msmith@acme.test",
        "manager": {"defaultFullName": "Fin Boss", "email": "fin@acme.test"},
    },
}
TIME_OFF = {
    "jdoe@acme.test": [
        {
            "externalCode": "T1",
            "timeType": "VACATION",
            "quantityInDays": "3",
            "approvalStatus": "APPROVED",
        }
    ],
    "msmith@acme.test": [
        {
            "externalCode": "T9",
            "timeType": "SICK",
            "quantityInDays": "1",
            "approvalStatus": "PENDING",
        }
    ],
}
_SF_EMAIL = re.compile(r"(email|manager/email|userIdNav/email) eq '((?:[^']|'')*)'")


def _sf_route(request: httpx.Request) -> httpx.Response:
    q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
    field, email = _SF_EMAIL.search(q["$filter"]).groups()
    email = email.replace("''", "'")
    if request.url.path.endswith("/User"):
        if field == "manager/email":
            rows = [u for u in SF_USERS.values() if u["manager"]["email"] == email]
        else:
            rows = [SF_USERS[email]] if email in SF_USERS else []
    else:
        rows = TIME_OFF.get(email, [])
        status = re.search(r"approvalStatus eq '([A-Z]+)'", q["$filter"])
        if status:
            rows = [r for r in rows if r["approvalStatus"] == status.group(1)]
    return httpx.Response(200, json={"d": {"results": rows}})


def _sf(**overrides) -> dict:
    data = {
        "name": "sf",
        "type": "sap_successfactors",
        "display_name": "SuccessFactors",
        "base_url": SF_BASE,
        "auth": {"method": "basic", "username": "${SF_USER}", "password": "${SF_PASSWORD}"},
        "operations": [
            {"name": n, "required_labels": LABELS}
            for n in ("my_profile", "my_direct_reports", "my_time_off_requests")
        ],
    }
    data.update(overrides)
    return data


@pytest.mark.asyncio
async def test_sf_profile_by_verified_email_with_manager():
    remote = Remote(_sf_route)
    _bind(remote, _sf(), resolver=public_dns)
    result = await execute_tool("sf__my_profile", {"user": "msmith@acme.test"})
    q = remote.query()
    assert remote.requests[0].url.path == "/odata/v2/User"
    assert q["$filter"] == "email eq 'jdoe@acme.test'" and q["$expand"] == "manager"
    (profile,) = result["data"]["items"]
    assert (
        profile["defaultFullName"] == "Jane Doe"
        and profile["manager.defaultFullName"] == "Max Boss"
    )
    assert "salary" not in json.dumps(result)


@pytest.mark.asyncio
async def test_sf_time_off_filters_by_user_navigation_and_status():
    remote = Remote(_sf_route)
    _bind(remote, _sf(), resolver=public_dns)
    result = await execute_tool("sf__my_time_off_requests", {"approval_status": "APPROVED"})
    assert remote.query()["$filter"] == (
        "userIdNav/email eq 'jdoe@acme.test' and approvalStatus eq 'APPROVED'"
    )
    assert result["data"]["items"][0]["timeType"] == "VACATION"
    bad = await execute_tool("sf__my_time_off_requests", {"approval_status": "' or 1 eq 1"})
    assert bad["error_kind"] == "invalid_arguments"


@pytest.mark.asyncio
async def test_sf_direct_reports_of_the_caller_only():
    remote = Remote(_sf_route)
    _bind(remote, _sf(), principal=_principal(email="max@acme.test"), resolver=public_dns)
    result = await execute_tool("sf__my_direct_reports", {})
    assert remote.query()["$filter"] == "manager/email eq 'max@acme.test'"
    assert [r["defaultFullName"] for r in result["data"]["items"]] == ["Jane Doe"]


@pytest.mark.asyncio
async def test_sf_unverified_email_gets_nothing():
    remote = Remote(_sf_route)
    _bind(remote, _sf(), principal=_principal(verified=False), resolver=public_dns)
    assert (await execute_tool("sf__my_profile", {}))["error_kind"] == "identity_unverified"
    assert remote.requests == []


def test_sf_is_public_cloud_https_only():
    with pytest.raises(ValidationError, match="must use https"):
        ConnectorConfig.model_validate(_sf(base_url="http://api4.successfactors.com/odata/v2"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "mine", "theirs"),
    [
        ("sf__my_profile", "Jane Doe", "Mark Smith"),
        ("sf__my_time_off_requests", "VACATION", "SICK"),
    ],
)
async def test_sf_cross_user_isolation(tool, mine, theirs):
    remote = Remote(_sf_route)
    _bind(remote, _sf(), principal=_principal(email="jdoe@acme.test"), resolver=public_dns)
    as_jdoe = json.dumps(await execute_tool(tool, {"user": "msmith@acme.test"}))
    _bind(remote, _sf(), principal=_principal("MSMITH", "msmith@acme.test"), resolver=public_dns)
    as_msmith = json.dumps(await execute_tool(tool, {}))
    assert mine in as_jdoe and theirs not in as_jdoe
    assert theirs in as_msmith and mine not in as_msmith
