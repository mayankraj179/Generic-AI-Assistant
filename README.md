# Generic AI Assistant Framework

Backend: Python 3.12 + FastAPI + Pydantic v2. Application UI: `ui/` (React + Vite,
Keycloak sign-in via Authorization Code + PKCE). Embeddable widget: `frontend/` (React +
Vite Shadow DOM custom element). DB: Postgres + pgvector, via Alembic migrations.

## Running the whole stack (local development)

Three terminals, from the repository root:

```
# 1. Postgres, Redis, Keycloak, Jaeger, MinIO
cd infra
docker compose up -d

# 2. Backend API on http://localhost:8000
cd backend
.\.venv\Scripts\Activate.ps1
alembic upgrade head
python -m uvicorn app.main:app --reload

# 3. Application UI on http://localhost:5174
cd ui
npm install
npm run dev
```

Open **http://localhost:5174**, click **Sign in with Keycloak**, and log in on Keycloak's
own page (dev user `dev_user` / `dev_password123`). You land on a demo company intranet
page with the Bitwise Assist **AI button** in the bottom-right corner: click it for a
compact chat window, and **maximize** that for the full Bitwise Assist UI (one chat, one
session, three views). The full-page app on its own is at http://localhost:5174/app.html.
See `ui/README.md` for UI configuration.
The UI never sees the password; it receives tokens through the PKCE code exchange and sends
`Authorization: Bearer <access token>` to the backend.

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
set `VITE_API_BASE=http://localhost:8001` in `ui/.env` (see `ui/.env.example`) and
restart `npm run dev` to match. If the wrong port ends up in that field, `postChat`/
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

The application UI (`ui/`) signs in with Authorization Code + PKCE (the client requires
S256 PKCE). For command-line testing only (curl, ingestion), the dev client still allows the
password grant — no UI uses it:

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
- Password-based auth in any UI, or app-owned identity flows (`ui/` uses Keycloak's
  hosted login via Authorization Code + PKCE).
- Database-backed app users; identity remains external to the framework.
- Additional model/tool execution logic beyond the request boundary.

## Knowledge sources

An assistant can declare where its knowledge comes from in a `knowledge_sources` block,
so onboarding a data source is YAML only. All three types are read-only:

| type | reads | document per | source_uri |
|---|---|---|---|
| `filesystem` | supported files under a directory (or one file) | file | `fs://<name>/<relative path>` |
| `object_storage` | supported objects under an S3-compatible bucket/prefix | object | `s3://<name>/<key>` |
| `database` | reviewed, parameterised SELECTs (PostgreSQL, MySQL, SQL Server, Oracle) | row (`key_column`) | `db://<name>/<query>/<key>` |

Each source needs a unique `name` and explicit `access_labels`. Credentials are only
`${ENV_VAR}` references; a literal secret, a password inside a URL, a missing or unknown
`type`, or a non-SELECT query fails config loading. Database queries run read-only (what
that means per engine is under [Database sources](#database-sources)) with a statement
timeout, and more rows (or objects) than the cap fails the sync rather than truncating it.
See `backend/configs/kb_demo_assistant.yaml` for one of each.

Sync an assistant's sources (`admin:ingest` permission; `source` is optional):

```
POST /admin/ingest/sync
{"assistant_id": "kb_demo_assistant", "tenant_id": "local-development", "source": "office_db"}
```

`tenant_id` is the tenant the content is for, as with `/admin/ingest`: retrieval scopes by
the signed-in principal's tenant, and the dev Keycloak tokens carry no tenant claim, so
local users are `local-development`. After discovery, everything goes through the existing
pipeline (parse, chunk, embed, content-hash idempotency, atomic generation replace), so
re-syncing an unchanged source embeds nothing. Content that has disappeared from a source is
retired (`is_current = false`, nothing deleted); an item that fails to parse is skipped and
kept; a source that can't be listed or queried is reported as failed and nothing of it is
retired.

Local demo data (MinIO bucket + read-only user, Postgres table + read-only role):

```
cd infra
docker compose up -d postgres minio
cd ../backend
python scripts/seed_kb_demo.py
# live adapter tests (offline hash embeddings, throwaway tenant):
$env:RUN_LIVE_SOURCE_TESTS = "1"; python -m pytest tests/test_knowledge_sources_live.py
```

### Database sources

A `database` source reads rows from an external database at sync time. Their embeddings
are stored in the framework's own PostgreSQL/pgvector database, as for every other source.
The two databases are separate: an assistant can read from MySQL and still store in
pgvector.

**Engines.** `dialect` picks the engine. It defaults to `postgresql`, so configs written
before dialects existed still work. Every engine goes through SQLAlchemy's own dialects and
drivers; there is no custom protocol code. Only the drivers below are accepted. Any other
driver in the URL is refused at sync time, even if SQLAlchemy could load it.

| `dialect` | driver (URL scheme) | install | verified live against | read-only enforcement | statement timeout |
|---|---|---|---|---|---|
| `postgresql` | asyncpg 0.31 (`postgresql+asyncpg://`) | core dependency | PostgreSQL 16 | `SET TRANSACTION READ ONLY` | server `statement_timeout` (transaction-local) |
| `mysql` | PyMySQL 1.2 (`mysql+pymysql://`) | `pip install -e ".[mysql]"` | MySQL 8.4 | `START TRANSACTION READ ONLY` | server `max_execution_time` (SELECT only) + client `read_timeout` backstop |
| `mssql` | pymssql 2.4 (`mssql+pymssql://`) | `pip install -e ".[mssql]"` | SQL Server 2022 | **none in the engine**: the transaction is always rolled back, never committed | pymssql query timeout, rounded **up to whole seconds**, plus `SET LOCK_TIMEOUT` |
| `oracle` | python-oracledb 26, thin mode (`oracle+oracledb://`) | `pip install -e ".[oracle]"` | Oracle Database 23ai Free | `SET TRANSACTION READ ONLY` | driver `call_timeout` (per round trip) |

`pip install -e ".[databases]"` installs all three optional drivers.
- A bare scheme such as `mysql://...` means the driver in the table.
- A URL whose scheme doesn't match `dialect` fails the sync for that source. Examples are
  `mariadb://`, or a `postgresql://` URL on a `mysql` source.
- MariaDB, aiomysql/asyncmy, pyodbc/aioodbc and cx_Oracle are **not** supported or tested.
- Adding an engine means one class in `backend/app/ingestion/sources/dialects.py`; that
  module's docstring lists the steps.

**YAML.** `backend/configs/examples/database_sources.yaml` has one source per engine.

```yaml
knowledge_sources:
  - name: employee_directory
    type: database
    dialect: mysql                      # postgresql (default) | mysql | mssql | oracle
    url: ${HR_DATABASE_URL}             # an env var reference only, never the URL itself
    password: ${HR_DATABASE_PASSWORD}   # optional, for a secret store that hands it out separately
    access_labels: ["role:authenticated"]
    max_rows: 1000                      # per query; more fails the sync (never truncates)
    statement_timeout_ms: 5000
    fetch_batch_size: 500               # rows per round trip (server-side cursor where available)
    queries:
      - name: employee_records
        key_column: employee_id         # unique, non-null: the row's stable identity
        sql: |
          SELECT employee_id, department, job_title
          FROM employee_directory
          WHERE active = :active
        params:
          active: true
```

Connection URLs go in the environment or `backend/.env`:

```
HR_DATABASE_URL=postgresql+asyncpg://kb_reader:<password>@db-host:5432/hr
HR_DATABASE_URL=mysql+pymysql://kb_reader:<password>@db-host:3306/hr
HR_DATABASE_URL=mssql+pymssql://kb_reader:<password>@db-host:1433/hr
HR_DATABASE_URL=oracle+oracledb://kb_reader:<password>@db-host:1521/?service_name=HRPDB
```

**Checked at config load.** None of these checks touch the network.
- `url` and `password` must be `${NAME}` references. A literal value fails loading, and the
  error does not echo it.
- Each query must start with SELECT or WITH and contain no `;`.
- Each query then passes two dialect-aware checks in `backend/app/config/sql_safety.py`:
  1. **Keyword deny-list.** Any unquoted INSERT, UPDATE, DELETE, MERGE, INTO, EXEC, CALL,
     SET, BEGIN, COMMIT, LOCK, WAITFOR, OPENROWSET, and similar, fails the query. This is
     needed because T-SQL needs no `;` between statements, and a parser reads
     `SELECT id FROM t DELETE FROM t` as one SELECT. Quote a column that really has one of
     these names.
  2. **Parse with sqlglot**, in the engine's own grammar. The query must be exactly one read
     query, with no write, `SELECT INTO` or `FOR UPDATE` anywhere in the tree, including in
     CTEs. SQL that sqlglot can't parse is rejected (fails closed).
- The query's `:name` placeholders and its `params` must match exactly. Values are always
  bound by the driver, never formatted into the SQL.

**At sync time.**
- A fresh, pool-less connection is opened. The URL's engine and driver are checked against
  `dialect` first.
- The session is made read-only where the engine supports it, and gets the statement
  timeout.
- Each query reads at most `max_rows + 1` rows, in batches of `fetch_batch_size`. The
  transaction is then rolled back. Nothing is ever committed.
- An overall deadline backs up the driver timeouts: connect timeout + each query's timeout
  + 10 s.
- Every driver or SQLAlchemy error becomes a per-source failure. It is classified as one of:
  could not connect, authentication failed, timed out, tried to write and was refused, or
  failed.
- Before an error is logged or returned, the password, the URL, and the user and host names
  are removed. The original exception is not chained, so its traceback can't leak them
  either.
- A failed source retires nothing. A query that succeeds with zero rows means the source is
  now empty, so its rows are retired.

**Database permissions.** Use a dedicated login that can only read the tables the queries
use. The session's read-only mode is the second layer of protection, not the first.
- PostgreSQL: `CREATE ROLE kb_reader LOGIN PASSWORD '...'`, then `GRANT CONNECT` on the
  database, `GRANT USAGE` on the schema, and `GRANT SELECT` on the tables. Also run
  `ALTER ROLE kb_reader SET default_transaction_read_only = on`.
- MySQL: `CREATE USER 'kb_reader'@'%' IDENTIFIED BY '...'`, then
  `GRANT SELECT ON hr.<table> TO 'kb_reader'@'%'`, and nothing else.
- SQL Server: create a login and a database user, then add the user to `db_datareader`, or
  grant the narrower `GRANT SELECT ON dbo.<table>`. SQL Server has no read-only transaction,
  so **this grant is the main write protection**. Never use a login that owns the schema or
  is in `db_owner`.
- Oracle: `CREATE USER kb_reader IDENTIFIED BY "..."`, then `GRANT CREATE SESSION` and
  `GRANT SELECT ON owner.<table>`. Never grant `SELECT ANY TABLE`.

**Known engine limitations.**
- **SQL Server**:
  - The engine does not enforce a read-only session. A write that somehow got past
    validation would run, then be rolled back at the end of the sync. The live test shows
    this with an admin login.
  - The timeout has 1-second granularity.
- **MySQL**:
  - `READ ONLY` still allows writes to `TEMPORARY` tables.
  - `max_execution_time` covers SELECT statements only.
  - When the row cap is exceeded, PyMySQL reads the rest of the streamed result before
    closing. `max_execution_time` still bounds that time.
  - MariaDB is not supported, because it names the timeout variable differently.
- **Oracle**:
  - `call_timeout` limits each round trip, not the whole statement.
  - When the timeout fires, the driver discards the connection.
  - Bind numbers, not Python `True`, for `NUMBER(1)` flag columns; on 23ai `True` binds as a
    `BOOLEAN`.
  - A table recreated or altered moments before a sync fails that sync with `ORA-01466`,
    because a READ ONLY transaction can't read DDL newer than its snapshot. The source is
    reported as failed, nothing is retired, and the next sync works.
- **All engines**:
  - Validation can't tell whether a function a query calls has side effects. Only grants and
    the read-only transaction cover that.
  - Each sync reads one full snapshot. There is no incremental or keyset pagination, because
    retiring deleted rows needs the complete key set.
  - For the thread-based drivers (all except asyncpg), when the overall deadline fires, the
    sync reports the source as failed straight away. The worker thread finishes at the
    driver's own timeout.

**Sync and inspect.**

```
POST /admin/ingest/sync           (admin:ingest permission)
{"assistant_id": "<assistant>", "tenant_id": "local-development", "source": "employee_directory"}
```

The response has one entry per source:
- `status` (`ok` or `failed`) and `error`, the redacted reason.
- Counts: `discovered`, `created`, `updated`, `unchanged`, `retired` and `chunks_embedded`.
- The URIs in `failed_items` and `retired_items`.

The logs carry the same redacted error. After you re-sync an unchanged table, every row
should be `unchanged`, with `chunks_embedded: 0`. To see what is stored, query the
`documents` rows whose `source_uri` starts with `db://<source name>/` and that have
`is_current` set.

**Tests.**
- `tests/test_database_sources.py` needs no network, plus the Postgres test DB for the sync
  cases. It covers:
  - config validation per dialect, and the URL and driver checks;
  - the read-only session setup each dialect issues, batching and the row cap;
  - error classification using each driver's real exception types, and redaction;
  - sync behaviour: change detection, retiring, labels, tenant and assistant isolation, and
    one failing source among working ones.
- `tests/test_database_engines_live.py` runs against real engines in throwaway containers;
  the start commands are in its docstring. For each engine it covers:
  - rows and params, and update and retire on re-sync;
  - writes blocked for both the SELECT-only login and an admin login;
  - timeout, row cap and an empty result;
  - wrong password and an unreachable server.

  ```
  $env:RUN_LIVE_DB_ENGINE_TESTS = "1"; python -m pytest tests/test_database_engines_live.py
  ```

**Why not MCP.** Ingestion stays an explicit, admin-triggered sync. A database source only
fetches its reviewed queries during that sync and exposes nothing to the LLM. Supporting
more engines needs drivers, not a tool protocol.

MCP fits a different, future feature: an assistant reading live data, or taking approved
actions, while it answers. That feature would need its own design:
- **Per-user authorization.** Each tool call runs with the signed-in principal's rights,
  not the ingestion reader's. The tool layer checks tenant, labels and permissions before
  every call. The MCP server gets a user-scoped token through token exchange or
  on-behalf-of, not a shared service account.
- **Allow-listed tools.** Tools are read-only by default and parameterised, with no free-form
  SQL. Writes need explicit approval.
- **Credentials.** They stay in the MCP server's own secret store, never in the model,
  the prompt or this framework's config.
- **Auditing.** Every call is audited, as `turn_audit` does for chat turns.

## Connectors

A connector lets an assistant fetch live data from an external system at question time
(for example the asking employee's current leave balance), alongside its ingested
documents. Connectors are **read-only**: an operation can only be a GET, and there is no
create, update or delete path, so there is no confirmation gate.

**Shape.** A connector *type* (`backend/app/connectors/catalog.py`) declares its operations:
input schema, timeout, response-size cap, baseline required labels, which fields are
server-supplied identity, and which response fields reach the model. An assistant enables
operations with a `connectors:` block. Each enabled operation becomes an ordinary tool
(`<connector>__<operation>`) on the existing tool path, for every provider, and counts
toward `max_tool_calls`. Example, from `backend/configs/hr_live_assistant.yaml`:

```yaml
connectors:
  - name: hr_system
    type: hr_api                  # must be a known type; unknown -> config load fails
    display_name: HR system       # shown as "Source: HR system, live"
    base_url: http://127.0.0.1:8099
    auth:
      method: oauth_client_credentials    # or: api_key (header + key)
      token_url: http://127.0.0.1:8099/oauth/token
      client_id: ${HR_API_CLIENT_ID}       # env var references only
      client_secret: ${HR_API_CLIENT_SECRET}
    identity:
      claim: preferred_username   # validated JWT claim = the user's id in that system
    operations:
      - name: get_leave_balance   # must be declared by the type; unknown -> load fails
        required_labels: ["role:authenticated", "scope:development"]
        timeout_seconds: 3        # may only tighten the type's default
```

**Rules enforced in code:**
- **Secrets.** Credential fields accept only `${ENV_VAR}`. A literal value, an unknown key
  or credentials in a URL fail config loading, and the rejected value is never echoed in
  the error.
- **Identity.** The user's id comes from the configured claim on the validated Principal
  and is injected server-side. Identity parameters are removed from the schema the model
  sees. If the model sends one anyway it is ignored, and the audit row records a
  `connector_identity_argument_ignored` guardrail action (field names only). A missing
  claim means no call.
- **Authorization.** An operation is offered and callable only if the caller holds **all**
  of its required labels; there is no empty "everyone" default. A denied call makes no
  request.
- **Failures.** Timeouts, unreachable hosts, 4xx/5xx, unreadable or oversized responses
  become a structured error for the model ("couldn't be reached, don't guess"). On a turn
  with no document context, a reply not backed by a successful tool result is replaced
  with a fixed "I couldn't reach the live data source (HR system)" reply. A reply with
  document context gets a fixed notice appended.
- **Provenance.** Connector answers are not citations. `/chat` returns `live_sources`
  (connector, display name, operations, `fetched_at`); `/chat/stream` returns them on
  the `done` event. The UI shows "Source: HR system, live (10:42)". `grounded` keeps
  meaning "backed by retrieved documents" only.
- **Observability.** `turn_audit.tool_calls` records connector, operation, outcome and
  latency for each call, never arguments, values or tokens. Each call has a
  `connector.call` span with the same fields. Log output is redacted: known secrets and
  issued tokens, bearer/basic headers, JWTs and `client_secret=`-style pairs.

**Service credential.** The backend authenticates as itself with an API key, or with
OAuth client credentials (token cached until shortly before expiry, refreshed, retried
once after a 401). On-behalf-of / token exchange is not built. The extension point is
`ConnectorAuth` in `backend/app/connectors/auth.py`, which already receives the Principal.
It would also need the inbound access token, which `PrincipalContext` doesn't carry today.
The `on_behalf_of` method name is reserved and rejected with a clear error.

**Local demo** (mock HR system plus `hr_live_assistant`):

```bash
cd backend
python scripts/mock_hr_api.py      # http://127.0.0.1:8099; reads HR_API_CLIENT_ID/SECRET from .env
# then, as an admin, ingest the leave policy document:
# POST /admin/ingest/sync {"assistant_id": "hr_live_assistant", "tenant_id": "local-development"}
```

The mock knows `dev_user` and `dev_user_b` (by `preferred_username`).
`POST /__mock/faults {"delay_seconds": 10}` or `{"status": 503}` simulates failures, and
`GET /__mock/requests` lists which employee ids were requested.

**Caveats.**
- Every authenticated dev user gets the same default labels, so the demo's
  `required_labels` don't separate users. Use a real role, such as `role:employee`, in
  production.
- On Gemini, google-adk records tool arguments and results on its own spans by default.
  Set `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false` to keep live data out of traces.
- `live_sources` isn't persisted, so a reloaded conversation shows document sources only.

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

## Application UI

`ui/` is the application to run — see "Running the whole stack" above and `ui/README.md`.

## Embeddable widget (`frontend/`)

`frontend/` builds the `<assistant-widget>` custom element (`src/widget.tsx`,
`src/AssistantWidgetApp.tsx`, `src/api.ts`) as a library (`npm run build` → `dist/`) for
embedding in other host pages. It has no sign-in of its own: the host page supplies an
access token through the `auth-token` attribute. It is not needed to run the application.

For widget development only, `npm run dev` in `frontend/` serves a harness on
http://localhost:5173 (`chat.html`: paste a dev token obtained with the curl command above).

## Known environment issue

`pip install` fails in this sandboxed shell — TLS handshake to pypi.org fails at
the OS (schannel) revocation-check step, before Python/pip cert config even
applies. This blocks creating a clean venv here. Tests were run instead against
this machine's existing system Python, which already had the required packages
installed from prior work. Run `pip install -e ".[dev]"` from a normal terminal
(outside this tool) to get a proper isolated venv.
