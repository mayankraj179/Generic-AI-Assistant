# Generic AI Assistant Framework

Day 1 scaffold. Backend: Python 3.12 + FastAPI + Pydantic v2. Frontend: React + Vite
Shadow DOM embeddable widget. DB: Postgres + pgvector, via Alembic migrations.

## Backend

```
cd backend
python -m pytest tests/ -v
python -m uvicorn app.main:app --reload
```

Requires: `fastapi`, `pydantic`, `pyyaml`, `pytest`, and the JWT/OIDC validation stack.

### Known environment quirk: port 8000 refusing to bind on some Windows setups

On at least one Windows dev machine, `uvicorn ... --port 8000` failed immediately with
`WinError 10013` ("An attempt was made to access a socket in a way forbidden by its access
permissions") — not a normal "port already in use" error. Investigated and ruled out:

- A stale process holding the port (`netstat -ano` showed nothing on 8000).
- Windows' reserved/excluded port range (`netsh interface ipv4 show excludedportrange
  protocol=tcp` didn't include 8000).
- A named Windows Firewall rule targeting port 8000 specifically.
- A Docker container (this project's or otherwise) publishing 8000.

None of the above reproduced the failure in a separate shell session on the same machine —
`uvicorn --port 8000` bound successfully there. That points at something scoped to the
specific interactive session/process that hit it (a security product doing per-process
socket filtering, a VPN client, or similar), not a persistent, generally-reproducible cause —
not something this repo can detect or fix. **If you hit this**: run the backend on a
different port instead, e.g. `python -m uvicorn app.main:app --reload --port 8001`, and
point the frontend dev harness's "API base URL" field (`frontend/chat.html`) at
`http://localhost:8001` to match. If the wrong port ends up in that field, `postChat`/
`postChatStream` now report it by name (`Could not reach http://localhost:8000 — check the
API base URL and confirm the backend is running there.`) instead of a generic "is it
running?" message, so a base-URL mismatch is visible directly in the error text.

### Authentication architecture

```
User / Browser
   ↓
Keycloak (local development IdP)
   ↓
JWT access token
   ↓
FastAPI resource server
   ↓
AuthProvider / JWT validation
   ↓
AuthenticatedIdentity
   ↓
PrincipalResolver
   ↓
PrincipalContext
   ↓
DevelopmentAuthorizationPolicy
   ↓
Business logic / RAG / tools
```

The application never stores passwords or issues JWTs itself. Keycloak is the local identity provider and the FastAPI app validates externally issued bearer tokens only.

### Local Keycloak development setup

From the repository root:

```
cd infra
docker compose up -d
```

This starts:
- PostgreSQL + pgvector
- Redis
- Keycloak on `http://localhost:8080`

The Keycloak import file lives in `infra/keycloak/realm-export.json` and creates a development realm named `generic-ai-dev` with a public client called `generic-ai-api` and a development user `dev_user` / `dev_password123`.

The app configuration is environment-driven via `backend/.env` (see `.env.example`):

```
AUTH_ENABLED=true
AUTH_ISSUER=http://localhost:8080/realms/generic-ai-dev
AUTH_AUDIENCE=generic-ai-api
AUTH_JWKS_URL=http://localhost:8080/realms/generic-ai-dev/protocol/openid-connect/certs
```

To obtain a token locally:

```
curl -X POST 'http://localhost:8080/realms/generic-ai-dev/protocol/openid-connect/token' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=dev_user&password=dev_password123&grant_type=password&client_id=generic-ai-api'
```

Then call the API with the returned access token:

```
curl http://localhost:8000/assistants \
  -H 'Authorization: Bearer <ACCESS_TOKEN>'
```

### Internal flow and future provider swap

The framework-internal path stays stable:

```
JWT
  ↓
AuthProvider
  ↓
AuthenticatedIdentity
  ↓
PrincipalResolver
  ↓
PrincipalContext
  ↓
AuthorizationPolicy
```

Later, the same flow can be used with Microsoft Entra ID or another OIDC provider simply by changing the issuer, audience, and JWKS configuration. Business logic does not depend on provider-specific claims or Keycloak APIs.

### What's implemented and tested
- `app/core/principal.py` — `PrincipalContext`: namespaced labels, clearance, frozen,
  zero-label detection, validated label format (`namespace:value`).
- `app/auth/provider.py` — JWT/OIDC validation with bearer-token extraction, issuer/audience/expiry checks, and JWKS-backed verification.
- `app/auth/models.py` — normalized `AuthenticatedIdentity` model detached from provider-specific claims.
- `app/auth/policy.py` — `DevelopmentAuthorizationPolicy` assigns the same common permission set to every valid authenticated principal.
- `app/auth/dependencies.py` — dependency injection for `PrincipalContext` with protected-route permission checks.
- `app/main.py` — `/health` is public; `/assistants` and `/chat` require a valid authenticated principal with development permissions.
- `infra/docker-compose.yml` — local Keycloak service plus PostgreSQL/Redis.
- `infra/keycloak/realm-export.json` — reproducible development realm/client/user configuration.

### Not yet implemented
- Employee-specific authorization, manager/HR/admin policies, or claim-to-role mapping beyond the development default.
- Frontend login UI, password-based auth, or app-owned identity flows.
- Database-backed app users; identity remains external to the framework.
- Additional model/tool execution logic beyond the request boundary.

## Database

```
cd infra
docker-compose up -d          # postgres (pgvector) + redis
cd ../backend
alembic upgrade head          # creates documents + chunks tables
```

**Not run yet in this session** — Docker was not available/started here, so the
migration has not been applied against a real database. The migration file itself
(`alembic/versions/0001_initial_documents_and_chunks.py`) has not been executed,
only authored.

## Frontend

```
cd frontend
npm install
npm run dev
```

**Not run yet in this session** — `npm install` was not attempted. The widget
(`src/widget.tsx`, `src/AssistantWidgetApp.tsx`) is a mount/unmount shell only,
no chat logic.

## Known environment issue

`pip install` fails in this sandboxed shell — TLS handshake to pypi.org fails at
the OS (schannel) revocation-check step, before Python/pip cert config even
applies. This blocks creating a clean venv here. Tests were run instead against
this machine's existing system Python, which already had the required packages
installed from prior work. Run `pip install -e ".[dev]"` from a normal terminal
(outside this tool) to get a proper isolated venv.
