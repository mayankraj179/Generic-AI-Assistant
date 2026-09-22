from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class AuthenticatedIdentity(BaseModel):
    """Normalized identity extracted from a validated JWT."""

    subject: str
    issuer: str
    audience: str | list[str]
    tenant_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    roles: frozenset[str] = Field(default_factory=frozenset)
    groups: frozenset[str] = Field(default_factory=frozenset)
    raw_claims: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_claims(cls, claims: Mapping[str, Any]) -> AuthenticatedIdentity:
        subject = str(
            claims.get("sub") or claims.get("principal_id") or claims.get("user_id") or ""
        )
        if not subject:
            raise ValueError("JWT is missing the required 'sub' claim")

        issuer = str(claims.get("iss") or "")
        if not issuer:
            raise ValueError("JWT is missing the required 'iss' claim")

        audience = claims.get("aud")
        if audience is None:
            raise ValueError("JWT is missing the required 'aud' claim")

        roles = _normalize_set(claims.get("realm_access", {}).get("roles"))
        roles |= _normalize_set(claims.get("roles"))

        groups = _normalize_set(claims.get("groups"))
        username = claims.get("preferred_username") or claims.get("email") or claims.get("username")
        display_name = claims.get("name") or claims.get("given_name")
        tenant_id = (
            claims.get("tenant_id")
            or claims.get("tenant")
            or claims.get("realm")
            or "local-development"
        )

        return cls(
            subject=subject,
            issuer=issuer,
            audience=audience if isinstance(audience, str) else list(audience),
            tenant_id=str(tenant_id),
            username=str(username) if username is not None else None,
            display_name=str(display_name) if display_name is not None else None,
            roles=roles,
            groups=groups,
            raw_claims={
                "tenant_id": tenant_id,
                "preferred_username": claims.get("preferred_username"),
                "email": claims.get("email"),
                "name": claims.get("name"),
                "groups": list(groups),
                "roles": list(roles),
            },
        )


def _normalize_set(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value})
    if isinstance(value, (list, tuple, set)):
        return frozenset(str(item) for item in value)
    return frozenset({str(value)})
