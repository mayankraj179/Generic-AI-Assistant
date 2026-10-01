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
const redirectUri = () => `${window.location.origin}${window.location.pathname}`;

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
  sessionStorage.setItem(PKCE_KEY, JSON.stringify({ verifier, state }));

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
 * true when tokens were obtained. Throws on a tampered/stale callback. */
export async function completeSignInIfRedirected(): Promise<boolean> {
  const url = new URL(window.location.href);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  if (!code || !state) {
    return false;
  }
  window.history.replaceState(null, "", redirectUri());

  const pending = sessionStorage.getItem(PKCE_KEY);
  sessionStorage.removeItem(PKCE_KEY);
  const { verifier, state: expectedState } = pending
    ? (JSON.parse(pending) as { verifier: string; state: string })
    : { verifier: "", state: "" };
  if (!verifier || state !== expectedState) {
    throw new Error("Sign-in response did not match this browser session. Please sign in again.");
  }

  saveTokens(
    await requestTokens({
      grant_type: "authorization_code",
      code,
      redirect_uri: redirectUri(),
      code_verifier: verifier,
    }),
  );
  return true;
}

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
  try {
    const refreshed = saveTokens(
      await requestTokens({ grant_type: "refresh_token", refresh_token: tokens.refreshToken }),
    );
    return refreshed.accessToken;
  } catch {
    sessionStorage.removeItem(TOKENS_KEY);
    return null;
  }
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
  sessionStorage.removeItem(TOKENS_KEY);
  const params = new URLSearchParams({
    client_id: KEYCLOAK_CLIENT_ID,
    post_logout_redirect_uri: redirectUri(),
  });
  if (idToken) {
    params.set("id_token_hint", idToken);
  }
  window.location.assign(`${endpoint("logout")}?${params}`);
}
