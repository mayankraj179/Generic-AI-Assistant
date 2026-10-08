from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from app.auth.models import AuthenticatedIdentity
from app.auth.policy import DevelopmentAuthorizationPolicy
from app.auth.provider import AuthenticationError, JwtAuthProvider
from app.config.settings import Settings
from app.core.principal import PrincipalContext
from app.main import app


@pytest.fixture
def auth_settings() -> Settings:
    return Settings(
        auth_enabled=True,
        auth_issuer="http://localhost:8080/realms/generic-ai-dev",
        auth_audience="generic-ai-api",
        auth_jwks_url="http://localhost:8080/realms/generic-ai-dev/protocol/openid-connect/certs",
    )


@pytest.fixture
def signing_keys() -> tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture
def provider(
    auth_settings: Settings,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> JwtAuthProvider:
    private_key, public_key = signing_keys
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return JwtAuthProvider(
        settings=auth_settings,
        signing_key=public_bytes,
        private_key=private_key,
    )


def _make_token(
    *,
    private_key: rsa.RSAPrivateKey,
    issuer: str,
    audience: str,
    subject: str = "dev-user-123",
    now: datetime | None = None,
    expires_in: int = 3600,
    additional_claims: dict[str, object] | None = None,
) -> str:
    now = now or datetime.now(UTC)
    claims = {
        "iss": issuer,
        "aud": audience,
        "sub": subject,
        "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
        "iat": int(now.timestamp()),
        "preferred_username": "dev_user",
        "email": "dev_user@example.com",
        "tenant_id": "dev-tenant",
        "roles": ["developer"],
        "groups": ["team:ai"],
    }
    if additional_claims:
        claims.update(additional_claims)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key"})


def test_missing_authorization_header_returns_401(
    auth_settings: Settings,
    provider: JwtAuthProvider,
) -> None:
    app.state.auth_provider = provider
    with pytest.raises(AuthenticationError):
        provider.require_principal("")


def test_invalid_bearer_format_returns_401(provider: JwtAuthProvider) -> None:
    with pytest.raises(AuthenticationError):
        provider.require_principal("Token abc")


def test_malformed_jwt_returns_401(provider: JwtAuthProvider) -> None:
    with pytest.raises(AuthenticationError):
        provider.validate_token("not-a-jwt")


def test_invalid_signature_returns_401(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    wrong_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _make_token(
        private_key=wrong_private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )
    with pytest.raises(AuthenticationError):
        provider.validate_token(token)


def test_expired_jwt_returns_401(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
        now=datetime.now(UTC) - timedelta(hours=2),
        expires_in=10,
    )
    with pytest.raises(AuthenticationError):
        provider.validate_token(token)


def test_wrong_issuer_returns_401(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/other-realm",
        audience="generic-ai-api",
    )
    with pytest.raises(AuthenticationError):
        provider.validate_token(token)


def test_wrong_audience_returns_401(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="wrong-api",
    )
    with pytest.raises(AuthenticationError):
        provider.validate_token(token)


def test_valid_jwt_becomes_authenticated_identity(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )
    identity = provider.validate_token(token)
    assert isinstance(identity, AuthenticatedIdentity)
    assert identity.subject == "dev-user-123"
    assert identity.tenant_id == "dev-tenant"
    assert identity.username == "dev_user"


def test_valid_jwt_becomes_principal_context(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )
    principal = provider.resolve_principal(token)
    assert isinstance(principal, PrincipalContext)
    assert principal.principal_id == "dev-user-123"
    assert principal.tenant_id == "dev-tenant"
    assert "role:authenticated" in principal.labels


def test_development_permission_set_is_assigned(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )
    principal = provider.resolve_principal(token)
    policy = DevelopmentAuthorizationPolicy()
    assert policy.is_allowed(principal, "assistant:use")
    assert policy.is_allowed(principal, "conversation:read")
    assert not policy.is_allowed(principal, "employee:approve")


def test_authenticated_principal_can_access_protected_route(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )
    app.state.auth_provider = provider
    client = TestClient(app)
    response = client.get("/assistants", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_unauthenticated_principal_cannot_access_protected_route(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    app.state.auth_provider = provider
    client = TestClient(app)
    response = client.get("/assistants")
    assert response.status_code == 401


def test_client_cannot_override_principal_identity(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
) -> None:
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )
    principal = provider.resolve_principal(token)
    assert principal.principal_id == "dev-user-123"
    assert principal.tenant_id == "dev-tenant"
    assert "role:admin" not in principal.labels


def _deeply_nested_token(depth: int = 6000) -> str:
    """A forged token whose payload is JSON nested `depth` levels deep
    (PyJWT GHSA-42vr-xj54-vc7v). At depth 6000 it is ~16 KB, which still fits
    in uvicorn's 16 KB request-header limit, so a real client can send it."""

    def b64url(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = b64url(json.dumps({"alg": "RS256", "typ": "JWT", "kid": "test-key"}).encode())
    payload = b64url(b"[" * depth + b"]" * depth)
    return f"{header}.{payload}.{b64url(b'forged-signature')}"


def test_deeply_nested_token_returns_401_not_500(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same path as production: keys come from JWKS (no static signing key).
    # The payload is parsed before any key is fetched, so the unreachable
    # JWKS URL is never contacted.
    settings = Settings(
        auth_enabled=True,
        auth_issuer="http://localhost:8080/realms/generic-ai-dev",
        auth_audience="generic-ai-api",
        auth_jwks_url="http://127.0.0.1:9/unreachable/certs",
    )
    monkeypatch.setattr(app.state, "auth_provider", JwtAuthProvider(settings=settings))
    # raise_server_exceptions=False: an unhandled error must surface as the
    # 500 a real client would get, not as an exception inside the test.
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/assistants", headers={"Authorization": f"Bearer {_deeply_nested_token()}"}
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "malformed or invalid token"}


def test_recursion_error_during_validation_is_rejected_as_invalid_token(
    provider: JwtAuthProvider,
    signing_keys: tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pins our own guard independently of the PyJWT version: even if a
    # library call raises RecursionError, validation fails closed with a 401.
    private_key, _ = signing_keys
    token = _make_token(
        private_key=private_key,
        issuer="http://localhost:8080/realms/generic-ai-dev",
        audience="generic-ai-api",
    )

    def raise_recursion_error(*args: object, **kwargs: object) -> None:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(jwt, "decode", raise_recursion_error)

    with pytest.raises(AuthenticationError) as excinfo:
        provider.validate_token(token)
    assert excinfo.value.detail == "malformed or invalid token"
