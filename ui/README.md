# Assistant Chat UI

Full-page React chatbot for the Generic AI Assistant Framework backend.
Sign in with Keycloak, pick an assistant, chat with streamed replies,
formatted answers (lists, tables, code), source chips, charts and a chat
history sidebar. Light and dark mode; works on phones.

It is a standalone Vite + React 18 + TypeScript app — it only talks to the
backend over HTTP, and needs no code from `backend/` or `frontend/`.

## Prerequisites

- Node.js 18+ and npm
- The backend running (`uvicorn app.main:app`, default `http://localhost:8000`)
- Keycloak running with the dev realm (`infra/docker-compose.yml` →
  `docker compose up -d keycloak`, default `http://localhost:8080`)

## Run it

```bash
cd ui
npm install
npm run dev
```

Open **http://localhost:5174**, click **Sign in with Keycloak** and log in
(dev realm user: `dev_user` / `dev_password123`).

## Configuration

Defaults match the backend's local-dev setup. If your backend or Keycloak
run elsewhere, copy `.env.example` to `.env` and edit:

| Variable | Default |
|---|---|
| `VITE_API_BASE` | `http://localhost:8000` |
| `VITE_KEYCLOAK_ISSUER` | `http://localhost:8080/realms/generic-ai-dev` |
| `VITE_KEYCLOAK_CLIENT_ID` | `generic-ai-api` |

Restart `npm run dev` after changing `.env`.

If you serve the UI from anything other than `http://localhost:*`, add that
origin to the Keycloak client's **Valid redirect URIs** and **Web origins**.

## How sign-in works

OIDC Authorization Code + PKCE against Keycloak's public client — the app
never sees a password. Tokens live in `sessionStorage` (per tab) and are
refreshed automatically before they expire. Every API call sends
`Authorization: Bearer <access token>`, exactly as the backend expects.

## Backend endpoints used

| Endpoint | Used for |
|---|---|
| `GET /assistants` | Assistant picker |
| `POST /chat/stream` | Sending a message (SSE: `delta`, `chart`, `done`, `error`) |
| `GET /sessions/{id}/messages` | Re-opening a past chat |

The backend has no "list my conversations" endpoint, so the sidebar's chat
list is kept in the browser's `localStorage` (titles and ids only — message
content is always re-fetched from the backend, which enforces ownership).

## Troubleshooting

- **Every answer is "I don't have enough information…"** — no documents are
  ingested for that assistant *under the signed-in user's tenant*. The dev
  Keycloak token has no tenant claim, so the backend uses tenant
  `local-development`; ingest with that `tenant_id` and an access label the
  user has (e.g. `role:authenticated`):

  ```bash
  curl -X POST http://localhost:8000/admin/ingest \
    -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
    -d '{"source_path":"<absolute path to doc>","assistant_id":"hr_assistant",
         "tenant_id":"local-development","access_labels":["role:authenticated"]}'
  ```

- **"Could not reach http://localhost:8000"** — backend not running, or on a
  different port; set `VITE_API_BASE`.
- **Keycloak "Invalid parameter: redirect_uri"** — the UI's origin isn't in
  the client's redirect URIs (see Configuration).

## Project layout

```
ui/
├── index.html
├── vite.config.ts        # dev server on :5174
├── .env.example
└── src/
    ├── main.tsx          # completes the sign-in redirect, mounts the app
    ├── ChatApp.tsx       # sign-in screen, layout, empty state
    ├── useChatSession.ts # messages + streaming for the open conversation
    ├── Sidebar.tsx       # new chat, assistant picker, chat history
    ├── Message.tsx       # message rendering, sources, copy
    ├── Composer.tsx      # auto-growing input
    ├── Markdown.tsx      # safe Markdown subset (no innerHTML)
    ├── ChartView.tsx     # bar / line / pie charts (recharts)
    ├── auth.ts           # Keycloak PKCE sign-in, refresh, sign-out
    ├── api.ts            # backend client (snake_case wire format)
    ├── history.ts        # sidebar chat list (localStorage)
    ├── config.ts         # endpoints + starter prompts
    └── app.css           # design tokens and styles
```
