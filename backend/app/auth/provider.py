from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jwt
from jwt import PyJWKClient

from app.auth.models import AuthenticatedIdentity
from app.auth.policy import DevelopmentAuthorizationPolicy
from app.config.settings import Settings
from app.core.principal import PrincipalContext


class AuthenticationError(Exception):
    """Raised when JWT validation fails. The dependency layer converts this into an HTTP 401."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        self.status_code = 401


class JwtAuthProvider:
    """OIDC/JWT provider adapter that validates bearer tokens and resolves PrincipalContext."""

    def __init__(
        self,
        settings: Settings,
        signing_key: str | bytes | None = None,
        private_key: Any | None = None,
        jwks_client: PyJWKClient | None = None,
    ) -> None:
        self.settings = settings
        self._signing_key = signing_key
        self._private_key = private_key
        self._jwks_client = jwks_client
        self._policy = DevelopmentAuthorizationPolicy()

    def require_principal(self, authorization_header: str | None) -> PrincipalContext:
        if not self.settings.auth_enabled:
            return PrincipalContext(
                tenant_id="local-development",
                principal_id="anonymous",
                labels=frozenset(),
                clearance=0,
            )

        token = self._extract_bearer_token(authorization_header)
        return self.resolve_principal(token)

    def resolve_principal(self, token: str) -> PrincipalContext:
        identity = self.validate_token(token)
        labels = set(self._policy.default_labels)

        if identity.tenant_id:
            labels.add(f"tenant:{identity.tenant_id}")

        for role in identity.roles:
            labels.add(f"role:{role}")

        for group in identity.groups:
            labels.add(f"group:{group}")

        return PrincipalContext(
            tenant_id=identity.tenant_id or "local-development",
            principal_id=identity.subject,
            labels=frozenset(labels),
            clearance=1,
        )

    def validate_token(self, token: str) -> AuthenticatedIdentity:
        if not token or not token.strip():
            raise AuthenticationError("missing token")

        try:
            headers = jwt.get_unverified_header(token)
            algorithm = headers.get("alg")
            allowed_algorithms = {"RS256", "RS384", "RS512", "ES256", "ES384", "ES512"}
            if algorithm not in allowed_algorithms:
                raise AuthenticationError("unsupported signing algorithm")

            key = self._resolve_signing_key(token)
            claims = jwt.decode(
                token,
                key=key,
                algorithms=[algorithm],
                audience=self.settings.auth_audience,
                issuer=self.settings.auth_issuer,
                options={"require": ["sub", "iss", "aud", "exp"]},
            )

            if self.settings.auth_audience and not _audience_matches(
                claims.get("aud"),
                self.settings.auth_audience,
            ):
                raise AuthenticationError("invalid audience")

            token_type = claims.get("typ") or claims.get("token_type")
            if token_type is not None and token_type not in {"Bearer", "access"}:
                raise AuthenticationError("invalid token type")

            return AuthenticatedIdentity.from_claims(claims)
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token has expired") from exc
        except jwt.InvalidAudienceError as exc:
            raise AuthenticationError("invalid audience") from exc
        except jwt.InvalidIssuerError as exc:
            raise AuthenticationError("invalid issuer") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError("malformed or invalid token") from exc
        except AuthenticationError:
            raise
        # RecursionError: a deeply nested JSON payload can escape PyJWT's own
        # error handling (GHSA-42vr-xj54-vc7v, PyJWT < 2.15); it must be a
        # 401 like any other malformed token, never a 500.
        except (TypeError, ValueError, RecursionError) as exc:
            raise AuthenticationError("malformed or invalid token") from exc

    def _extract_bearer_token(self, authorization_header: str | None) -> str:
        if authorization_header is None:
            raise AuthenticationError("missing Authorization header")

        parts = authorization_header.split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError("invalid bearer token format")

        token = parts[1].strip()
        if not token:
            raise AuthenticationError("missing bearer token")
        return token

    def _resolve_signing_key(self, token: str) -> str | bytes:
        if self._signing_key is not None:
            return self._signing_key
        if self.settings.auth_jwks_url:
            if self._jwks_client is None:
                self._jwks_client = PyJWKClient(self.settings.auth_jwks_url)
            return self._jwks_client.get_signing_key_from_jwt(token).key
        raise AuthenticationError("JWT verification is not configured")


def _audience_matches(jwt_audience: Any, expected_audience: str) -> bool:
    if isinstance(jwt_audience, str):
        return jwt_audience == expected_audience
    if isinstance(jwt_audience, Sequence):
        return expected_audience in [str(item) for item in jwt_audience]
    return False
