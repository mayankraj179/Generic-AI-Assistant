"""Phase 5: the README's worked example (configs/examples/connectors/
room_bookings.yaml), a system added as YAML only, tested the way any new
connector should be: against a mock of its API, including cross-user
isolation; plus the `python -m app.connectors.check` validator.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import yaml

from app.config.connectors import ConnectorConfig
from app.connectors import check
from app.connectors.auth import reset_auth_cache
from app.connectors.runtime import TurnConnectors, bind_turn
from app.core.principal import PrincipalContext
from app.tools.builtin import execute_tool

EXAMPLE = Path(__file__).resolve().parent.parent / "configs/examples/connectors/room_bookings.yaml"
ALICE, BOB = "alice@example.test", "bob@example.test"
BOOKINGS = [
    {
        "id": "bk_alice01",
        "organizer": ALICE,
        "room": {"name": "Everest", "floor": 3},
        "start": "2026-10-13T10:00",
        "end": "2026-10-13T11:00",
        "status": "confirmed",
        "notes": "Q4 planning",
        "attendees_count": 6,
        "billing_code": "never-returned",
    },
    {
        "id": "bk_bob0001",
        "organizer": BOB,
        "room": {"name": "Kailash", "floor": 2},
        "start": "2026-10-13T12:00",
        "end": "2026-10-13T13:00",
        "status": "confirmed",
        "notes": "Bob: compensation review",
        "attendees_count": 2,
    },
]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch):
    reset_auth_cache()
    monkeypatch.setenv("ROOMS_API_KEY", "rooms-key-0123456789")
    yield
    reset_auth_cache()


async def public_dns(host: str, port: int) -> list[str]:
    return ["93.184.216.34"]


def _principal(email: str) -> PrincipalContext:
    return PrincipalContext(
        tenant_id="t1",
        principal_id=f"sub-{email}",
        labels=frozenset({"role:employee"}),
        clearance=1,
        claims={"email": email, "email_verified": "true"},
    )


class Rooms:
    """Mock of the booking API: the service key sees everyone's bookings;
    filters are applied the way the real API would apply them."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["x-api-key"] == "rooms-key-0123456789"
        if request.url.path == "/api/v1/rooms/availability/search":
            body = json.loads(request.content)
            rooms = [
                {
                    "name": "Everest",
                    "floor": 3,
                    "capacity": 8,
                    "equipment": ["screen"],
                    "cost_centre": "never-returned",
                }
            ]
            return httpx.Response(
                200,
                json={"rooms": [r for r in rooms if r["capacity"] >= body.get("min_capacity", 0)]},
            )
        q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        rows = [b for b in BOOKINGS if b["organizer"] == q.get("organizer")]
        if "id" in q:
            rows = [b for b in rows if b["id"] == q["id"]]
        return httpx.Response(200, json={"data": rows, "meta": {"next_cursor": None}})


def _bind(rooms: Rooms, email: str) -> None:
    config = ConnectorConfig.model_validate(yaml.safe_load(EXAMPLE.read_text(encoding="utf-8")))
    bind_turn(
        TurnConnectors(
            _principal(email),
            [config],
            transport=httpx.MockTransport(rooms.handler),
            resolver=public_dns,
        )
    )


@pytest.mark.asyncio
async def test_my_bookings_is_scoped_and_minimised():
    rooms = Rooms()
    _bind(rooms, ALICE)
    result = await execute_tool("rooms__my_bookings", {"status": "confirmed", "organizer": BOB})
    q = parse_qs(rooms.requests[0].url.query.decode())
    assert q["organizer"] == [ALICE] and q["status"] == ["confirmed"]
    assert q["limit"] == ["20"] and q["sort"] == ["start"]
    assert result["data"]["items"] == [
        {
            "id": "bk_alice01",
            "room.name": "Everest",
            "room.floor": 3,
            "start": "2026-10-13T10:00",
            "end": "2026-10-13T11:00",
            "status": "confirmed",
        }
    ]


@pytest.mark.asyncio
async def test_get_booking_by_someone_elses_id_finds_nothing():
    rooms = Rooms()
    _bind(rooms, ALICE)
    theirs = await execute_tool("rooms__get_booking", {"id": "bk_bob0001"})
    assert theirs["data"]["items"] == []
    assert "compensation" not in json.dumps(theirs)
    mine = await execute_tool("rooms__get_booking", {"id": "bk_alice01"})
    assert mine["data"]["items"][0]["notes"] == "Q4 planning"
    for request in rooms.requests:
        assert parse_qs(request.url.query.decode())["organizer"] == [ALICE]
    assert (await execute_tool("rooms__get_booking", {"id": "../bookings"}))[
        "error_kind"
    ] == "invalid_arguments"


@pytest.mark.asyncio
async def test_find_free_rooms_is_a_read_only_post_search():
    rooms = Rooms()
    _bind(rooms, ALICE)
    result = await execute_tool(
        "rooms__find_free_rooms",
        {"start": "2026-10-14T09:00", "end": "2026-10-14T10:00", "min_capacity": 6},
    )
    request = rooms.requests[0]
    assert request.method == "POST"
    assert json.loads(request.content) == {
        "include_unbookable": False,
        "start": "2026-10-14T09:00",
        "end": "2026-10-14T10:00",
        "min_capacity": 6,
    }
    assert result["data"]["items"] == [
        {"name": "Everest", "floor": 3, "capacity": 8, "equipment": ["screen"]}
    ]


@pytest.mark.asyncio
async def test_cross_user_isolation_for_the_worked_example():
    rooms = Rooms()
    _bind(rooms, ALICE)
    as_alice = json.dumps(await execute_tool("rooms__my_bookings", {"organizer": BOB}))
    _bind(rooms, BOB)
    as_bob = json.dumps(await execute_tool("rooms__my_bookings", {"organizer": ALICE}))
    assert "Everest" in as_alice and "Kailash" not in as_alice
    assert "Kailash" in as_bob and "Everest" not in as_bob


def test_check_cli_describes_what_the_model_sees(capsys, monkeypatch):
    monkeypatch.delenv("ROOMS_API_KEY")
    monkeypatch.setattr(check, "_dotenv", lambda: {})
    assert check.main([str(EXAMPLE)]) == 0
    out = capsys.readouterr().out
    assert "tool rooms__my_bookings" in out
    assert "filled server-side: ['organizer <- claim email']" in out
    assert "${ROOMS_API_KEY}: MISSING" in out
    assert "POST (read_only_post)" in out
    assert "rooms-key" not in out


def _write(tmp_path, data) -> str:
    path = tmp_path / "connector.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_check_cli_rejects_a_write_method(tmp_path, capsys):
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["operations"][0]["method"] = "DELETE"
    assert check.main([_write(tmp_path, data)]) == 1
    err = capsys.readouterr().err
    assert "method 'DELETE' is not allowed: connectors are read-only" in err


def test_check_cli_rejects_a_literal_secret_without_echoing_it(tmp_path, capsys):
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["auth"]["key"] = "literal-secret-0123456789"
    assert check.main([_write(tmp_path, data)]) == 1
    err = capsys.readouterr().err
    assert "auth.api_key.key" in err and "literal secrets are not allowed" in err
    assert "literal-secret-0123456789" not in err


def test_check_cli_reads_an_assistant_config():
    hr = Path(__file__).resolve().parent.parent / "configs/hr_live_assistant.yaml"
    (connector,) = check.check(hr)
    assert connector.name == "hr_system"
