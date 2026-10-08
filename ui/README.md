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
(dev realm user: `dev_user` / `dev_password123`). You land on a demo company
intranet page with only the **AI button** (bottom-right):

```
signed in → AI button → click → compact chat window → maximize → full UI
```

The full-page app on its own is at **http://localhost:5174/app.html**.

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
If a refresh fails or the backend answers 401, the local tokens are
dropped and the sign-in screen is shown. **Sign out** clears them and ends
the Keycloak session.

The Keycloak client (`generic-ai-api` in `infra/keycloak/realm-export.json`)
requires S256 PKCE. Keycloak only imports that file when the realm doesn't
exist yet, so an existing `keycloak_data` volume keeps whatever client
settings it was created with.

## Backend endpoints used

| Endpoint | Used for |
|---|---|
| `GET /assistants` | Assistant picker |
| `POST /chat/stream` | Sending a message (SSE: `delta`, `chart`, `done`, `error`) |
| `GET /sessions/{id}/messages` | Re-opening a past chat |

The backend has no "list my conversations" endpoint, so the sidebar's chat
list is kept in the browser's `localStorage` (titles and ids only — message
content is always re-fetched from the backend, which enforces ownership).

## Embedding on another page (chat launcher)

The same app also ships as a floating launcher for an intranet or portal
page: a bubble in the corner opens a compact chat popup, and **Maximize**
expands the same live conversation into the full layout above (sidebar,
history, assistant picker, sign out). Maximize/restore never restarts the
chat or interrupts a streaming reply.

**Try it:** with `npm run dev` running, open **http://localhost:5174/**
(`index.html`). It's a fake "Bitwise Intranet" page whose nav links are
real page loads, so you can check the chat survives navigation. It sets
`data-sign-in="page"`, so you sign in first and then land on the closed
launcher.

**Build it:** `npm run build:embed` writes one self-contained file,
`dist-embed/bitwise-assist.js` (React and CSS included, ~607 kB, ~180 kB
gzipped; most of that is recharts). Add it to any page:

```html
<script src="https://assist.example.com/bitwise-assist.js"
  data-api-base="https://assist-api.example.com"
  data-keycloak-url="https://sso.example.com" data-realm="generic-ai-dev"
  data-client-id="generic-ai-api"
  data-default-assistant="hr_assistant"
  data-position="right"
  data-z-index="2147483000"
  data-sign-in="popup"></script>
```

Every attribute is optional; defaults are those in `src/config.ts`.
`data-default-assistant` falls back to the first assistant `GET /assistants`
returns if it isn't offered. `data-position` is `right` (default) or `left`.

How it behaves:

- **Isolation.** The widget renders in a Shadow DOM root, so the host page's
  CSS can't restyle it and its CSS can't leak out. Sizes are in px, so the
  host's root font size doesn't distort it. The only thing it adds to the
  host document is the Google Fonts `<link>` in `<head>`, because
  `@font-face` doesn't register from inside a shadow root.
- **Sign-in.** Sign-in is the same full-page Authorization Code + PKCE
  redirect as the app. Keycloak returns to the exact page you were on (path,
  query and hash) and the `code`/`state` parameters are removed. The widget
  never sees a password. `data-sign-in` picks where signed-out users sign in:
  - `popup` (default, for embedding on any site): the bubble shows, and the
    popup holds a "Sign in with Keycloak" card; the popup reopens after
    sign-in.
  - `page` (what this app's `index.html` uses): a full-page sign-in covers
    the page first; after signing in you land on the closed launcher (just
    the AI button). Sign-out returns to the sign-in page.
- **No iframe.** Keycloak's login page refuses to be framed
  (`X-Frame-Options: SAMEORIGIN`, `frame-ancestors 'self'`, verified against
  the dev Keycloak), so the widget lives in the host page itself.
- **One login = one new chat.** The open/closed/maximized state and the
  active conversation id are kept in `sessionStorage["gaaf.widget"]`, so
  moving between portal pages keeps the chat open and intact. A fresh
  sign-in (a completed code exchange) and sign-out clear the conversation;
  a normal page load, a token refresh or maximize/restore don't. Past chats
  stay in the maximized sidebar history.
- **Keyboard.** Escape steps down one level: maximized → popup → closed.
  Opening the popup focuses the message box.
- On screens 480px wide or less, the popup is a full-screen sheet.

### Deploying on a different origin

The demo works with no configuration changes because it runs on
`http://localhost:5174`. A real portal at, say, `https://intranet.example.com`
needs:

1. **Backend CORS:** add the portal origin to the allow-list in
   `backend/app/main.py` (local dev currently allows `*`).
2. **Keycloak client** `generic-ai-api`: add a **Valid redirect URI** for the
   portal pages (e.g. `https://intranet.example.com/*`; the query string is
   part of the redirect URI, so a wildcard is needed for pages with query
   parameters) and the portal origin under **Web origins**.
3. Serve `bitwise-assist.js` from anywhere the portal can load scripts from,
   and allow that source, the API, Keycloak and Google Fonts in the portal's
   Content-Security-Policy if it has one.

If the portal already signs users in through the same Keycloak realm,
"Sign in" is instant: Keycloak's SSO cookie completes the redirect without
showing a login form.

### Known limitation (security)

Embedded in-page, the tokens live in the **host page's** origin storage
(`sessionStorage`), so any script running on that portal page can read
them, the same as any other code in that page. Use it only on pages whose
scripts you trust. The hardening path is to serve the chat from its own
origin in an iframe and have that frame do the sign-in redirect in a popup
window or top-level navigation, which keeps tokens out of the host
origin. That is not implemented.

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
├── index.html            # "/": demo intranet page + the launcher (entry page)
├── app.html              # "/app.html": the full-page app on its own
├── vite.config.ts        # dev server on :5174
├── vite.embed.config.ts  # npm run build:embed → dist-embed/bitwise-assist.js
├── .env.example
└── src/
    ├── main.tsx          # completes the sign-in redirect, mounts the app
    ├── ChatApp.tsx       # sign-in screen, FullLayout, empty state
    ├── useChatController.ts # assistants, history, the one chat session
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
    ├── app.css           # design tokens and styles
    ├── embed.tsx         # launcher entry: script data-*, Shadow DOM mount
    └── widget/
        ├── Launcher.tsx  # bubble, compact popup, maximized overlay
        ├── widgetState.ts # view + active conversation (sessionStorage)
        └── widget.css    # launcher styles (on top of app.css)
```
