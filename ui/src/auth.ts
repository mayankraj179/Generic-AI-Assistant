import { KEYCLOAK_CLIENT_ID, KEYCLOAK_ISSUER } from "./config";

/** OIDC Authorization Code + PKCE against Keycloak. The app never sees a
 * password: sign-in is a redirect to Keycloak's own login page, and the
 * backend only ever receives the resulting bearer token (see README.md →
 * "Authentication architecture"). Tokens live in sessionStorage, so they
 * are scoped to this tab and gone when it closes. */

const TOKENS_KEY = "gaaf.tokens";
const PKCE_KEY = "gaaf.pkce";
const REFRESH_MARGIN_MS = 30_000;

interface StoredTokens {
  accessToken: string;
  refreshToken: string;
  idToken?: string;
  expiresAt: number;
}

interface TokenResponse {
  access_token: string;
  refresh_token: string;
  id_token?: string;
  expires_in: number;
}

export interface SignedInUser {
  subject: string;
  name: string;
}

const endpoint = (path: string) => `${KEYCLOAK_ISSUER}/protocol/openid-connect/${path}`;
// Parameters Keycloak adds to the callback; everything else in the query
// belongs to the host page and is kept, so a portal page like
// /home?page=leave reloads as itself after sign-in, not as bare /home.
const CALLBACK_PARAMS = ["code", "state", "session_state", "iss", "error", "error_description", "error_uri"];

/** This page's URL minus the hash and any callback parameters. Identical
 * at sign-in time and at callback time, as the code exchange requires. */
const redirectUri = () => {
  const url = new URL(window.location.href);
  url.hash = "";
  CALLBACK_PARAMS.forEach((p) => url.searchParams.delete(p));
  return url.href;
};

interface PendingSignIn {
  verifier: string;
  state: string;
  /** Full URL (query + hash included) the user signed in from. */
  returnTo?: string;
}

function takePendingSignIn(): PendingSignIn | null {
  const raw = sessionStorage.getItem(PKCE_KEY);
  sessionStorage.removeItem(PKCE_KEY);
  try {
    return raw ? (JSON.parse(raw) as PendingSignIn) : null;
  } catch {
    return null;
  }
}

/** Where to put the address bar after the callback: the page the user
 * signed in from (any path, query or hash on this origin), else the
 * callback URL without its code/state. */
function returnUrl(pending: PendingSignIn | null): string {
  try {
    const target = new URL(pending?.returnTo ?? "", window.location.href);
    if (pending?.returnTo && target.origin === window.location.origin) return target.href;
  } catch {
    // Unparseable — fall through.
  }
  return redirectUri();
}

function base64Url(bytes: Uint8Array): string {
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomString(byteLength: number): string {
  return base64Url(crypto.getRandomValues(new Uint8Array(byteLength)));
}

function readTokens(): StoredTokens | null {
  try {
    const raw = sessionStorage.getItem(TOKENS_KEY);
    return raw ? (JSON.parse(raw) as StoredTokens) : null;
  } catch {
    return null;
  }
}

function saveTokens(response: TokenResponse): StoredTokens {
  const tokens: StoredTokens = {
    accessToken: response.access_token,
    refreshToken: response.refresh_token,
    idToken: response.id_token,
    expiresAt: Date.now() + response.expires_in * 1000,
  };
  sessionStorage.setItem(TOKENS_KEY, JSON.stringify(tokens));
  return tokens;
}

async function requestTokens(form: Record<string, string>): Promise<TokenResponse> {
  const response = await fetch(endpoint("token"), {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ client_id: KEYCLOAK_CLIENT_ID, ...form }),
  });
  if (!response.ok) {
    throw new Error(`Keycloak token request failed (${response.status})`);
  }
  return (await response.json()) as TokenResponse;
}

export async function signIn(): Promise<void> {
  const verifier = randomString(48);
  const state = randomString(16);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  const pending: PendingSignIn = { verifier, state, returnTo: window.location.href };
  sessionStorage.setItem(PKCE_KEY, JSON.stringify(pending));

  const params = new URLSearchParams({
    client_id: KEYCLOAK_CLIENT_ID,
    response_type: "code",
    scope: "openid profile",
    redirect_uri: redirectUri(),
    state,
    code_challenge: base64Url(new Uint8Array(digest)),
    code_challenge_method: "S256",
  });
  window.location.assign(`${endpoint("auth")}?${params}`);
}

/** Completes a sign-in redirect if the current URL carries one. Returns
 * true when tokens were obtained — i.e. a fresh sign-in, as opposed to a
 * normal page load with an existing session. Throws on a tampered/stale
 * callback. Either way the address bar is put back to the page the user
 * started from, minus the code/state. */
export async function completeSignInIfRedirected(): Promise<boolean> {
  const url = new URL(window.location.href);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const error = url.searchParams.get("error");
  if (error && state) {
    // Keycloak redirected back without a code (user cancelled, login
    // disabled, ...). Clear the callback from the URL and surface it.
    window.history.replaceState(null, "", returnUrl(takePendingSignIn()));
    throw new Error(`Sign-in was not completed (${url.searchParams.get("error_description") ?? error}).`);
  }
  if (!code || !state) {
    return false;
  }
  // Must equal the redirect_uri sent to /auth (see redirectUri()).
  const callbackRedirectUri = redirectUri();
  const pending = takePendingSignIn();
  window.history.replaceState(null, "", returnUrl(pending));
  if (!pending?.verifier || state !== pending.state) {
    throw new Error("Sign-in response did not match this browser session. Please sign in again.");
  }

  saveTokens(
    await requestTokens({
      grant_type: "authorization_code",
      code,
      redirect_uri: callbackRedirectUri,
      code_verifier: pending.verifier,
    }),
  );
  return true;
}

// Shared by concurrent callers so a burst of requests near expiry spends the
// refresh token once instead of racing several refreshes against Keycloak.
let pendingRefresh: Promise<string | null> | null = null;

/** A valid access token, refreshed when close to expiry; null when the
 * session is gone and the user must sign in again. */
export async function getAccessToken(): Promise<string | null> {
  const tokens = readTokens();
  if (!tokens) {
    return null;
  }
  if (tokens.expiresAt - REFRESH_MARGIN_MS > Date.now()) {
    return tokens.accessToken;
  }
  pendingRefresh ??= requestTokens({ grant_type: "refresh_token", refresh_token: tokens.refreshToken })
    .then((response) => saveTokens(response).accessToken)
    .catch(() => {
      clearSession();
      return null;
    })
    .finally(() => {
      pendingRefresh = null;
    });
  return pendingRefresh;
}

/** Drops the local token set without a Keycloak round trip — used when the
 * backend rejects the token (401), so it is never sent again. */
export function clearSession(): void {
  sessionStorage.removeItem(TOKENS_KEY);
}

export function currentUser(): SignedInUser | null {
  const tokens = readTokens();
  if (!tokens) {
    return null;
  }
  try {
    const payload = tokens.accessToken.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const claims = JSON.parse(atob(payload)) as Record<string, unknown>;
    const name = claims.name ?? claims.preferred_username ?? "Signed in";
    return { subject: String(claims.sub ?? ""), name: String(name) };
  } catch {
    return null;
  }
}

export function signOut(): void {
  const idToken = readTokens()?.idToken;
  clearSession();
  const params = new URLSearchParams({
    client_id: KEYCLOAK_CLIENT_ID,
    post_logout_redirect_uri: redirectUri(),
  });
  if (idToken) {
    params.set("id_token_hint", idToken);
  }
  window.location.assign(`${endpoint("logout")}?${params}`);
}
