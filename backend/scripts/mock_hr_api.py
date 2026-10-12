"""A small mock HR system for the hr_api connector (configs/hr_live_assistant.yaml).

Dev only. Run from backend/:

    python scripts/mock_hr_api.py            # http://127.0.0.1:8099

Read-only, like the connector that calls it:
  POST /oauth/token        client-credentials grant (HTTP Basic or form
                           client_id/client_secret). Issues a short-lived
                           bearer token.
  GET  /api/v1/employees/{employee_id}/leave-balance[?leave_type=annual]
                           Requires that bearer token. 404 for an unknown
                           employee.

The service credential is HR_API_CLIENT_ID / HR_API_CLIENT_SECRET, read from
the environment or backend/.env: the same values the backend sends.

Dev-only helpers, for exercising the failure paths and checking whose data
was requested (no auth, bound to 127.0.0.1 only):
  POST /__mock/faults      {"delay_seconds": 10} or {"status": 503}; {} clears
  GET  /__mock/requests    employee ids requested so far (never tokens)
  DELETE /__mock/requests  clears that log
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import secrets
import time
from datetime import date
from pathlib import Path

from dotenv import dotenv_values
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

_ENV = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
TOKEN_TTL_SECONDS = int(os.environ.get("MOCK_HR_TOKEN_TTL_SECONDS", "300"))

# Keyed by the id the backend maps from the JWT claim (preferred_username in
# the example config).
EMPLOYEES: dict[str, dict] = {
    "dev_user": {
        "annual": {"available_days": 14.5, "used_days": 6.5, "accrued_days": 21.0},
        "sick": {"available_days": 9.0, "used_days": 3.0, "accrued_days": 12.0},
        "casual": {"available_days": 4.0, "used_days": 2.0, "accrued_days": 6.0},
    },
    "dev_user_b": {
        "annual": {"available_days": 3.0, "used_days": 18.0, "accrued_days": 21.0},
        "sick": {"available_days": 12.0, "used_days": 0.0, "accrued_days": 12.0},
        "casual": {"available_days": 1.0, "used_days": 5.0, "accrued_days": 6.0},
    },
}

app = FastAPI(title="Mock HR API", docs_url=None, redoc_url=None)
_tokens: dict[str, float] = {}
_faults: dict[str, float | int] = {}
_requested: list[str] = []


def _credential(name: str) -> str:
    value = os.environ.get(name) or _ENV.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set (environment or backend/.env)")
    return value


def _client_from_request(authorization: str | None, form: dict) -> tuple[str, str]:
    if authorization and authorization.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(authorization[6:]).decode()
            client_id, _, client_secret = decoded.partition(":")
            return client_id, client_secret
        except (ValueError, UnicodeDecodeError):
            return "", ""
    return str(form.get("client_id", "")), str(form.get("client_secret", ""))


@app.post("/oauth/token")
async def issue_token(request: Request, authorization: str | None = Header(default=None)):
    form = dict(await request.form())
    if form.get("grant_type") != "client_credentials":
        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)
    client_id, client_secret = _client_from_request(authorization, form)
    expected_id = _credential("HR_API_CLIENT_ID")
    expected_secret = _credential("HR_API_CLIENT_SECRET")
    if not (
        secrets.compare_digest(client_id, expected_id)
        and secrets.compare_digest(client_secret, expected_secret)
    ):
        return JSONResponse({"error": "invalid_client"}, status_code=401)
    token = secrets.token_urlsafe(32)
    _tokens[token] = time.time() + TOKEN_TTL_SECONDS
    return {"access_token": token, "token_type": "Bearer", "expires_in": TOKEN_TTL_SECONDS}


def _require_token(authorization: str | None) -> None:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    expires = _tokens.get(authorization[7:].strip())
    if expires is None or expires < time.time():
        raise HTTPException(status_code=401, detail="invalid or expired token")


@app.get("/api/v1/employees/{employee_id}/leave-balance")
async def leave_balance(
    employee_id: str,
    leave_type: str | None = None,
    authorization: str | None = Header(default=None),
):
    _require_token(authorization)
    _requested.append(employee_id)
    print(f"mock-hr: leave-balance requested for employee_id={employee_id!r}", flush=True)
    if _faults.get("delay_seconds"):
        await asyncio.sleep(float(_faults["delay_seconds"]))
    if _faults.get("status"):
        raise HTTPException(status_code=int(_faults["status"]), detail="injected fault")
    record = EMPLOYEES.get(employee_id)
    if record is None:
        raise HTTPException(status_code=404, detail="employee not found")
    types = [leave_type] if leave_type else list(record)
    if leave_type and leave_type not in record:
        raise HTTPException(status_code=400, detail="unknown leave_type")
    return {
        "employee_id": employee_id,
        "as_of": date.today().isoformat(),
        "balances": [{"leave_type": t, **record[t]} for t in types],
    }


@app.post("/__mock/faults")
async def set_faults(request: Request):
    body = await request.json() if await request.body() else {}
    _faults.clear()
    _faults.update({k: v for k, v in body.items() if k in ("delay_seconds", "status")})
    return {"faults": _faults}


@app.get("/__mock/requests")
async def list_requests():
    return {"employee_ids": list(_requested)}


@app.delete("/__mock/requests")
async def clear_requests():
    _requested.clear()
    return {"employee_ids": []}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    _credential("HR_API_CLIENT_ID"), _credential("HR_API_CLIENT_SECRET")  # fail fast
    # access_log off: uvicorn's access log is harmless here (no tokens in
    # URLs), but the print above is the record this mock means to keep.
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
