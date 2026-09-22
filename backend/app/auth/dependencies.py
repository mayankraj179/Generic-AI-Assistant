from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from app.auth.policy import AuthorizationService
from app.auth.provider import AuthenticationError, JwtAuthProvider
from app.config.settings import settings
from app.core.principal import PrincipalContext


def get_auth_provider() -> JwtAuthProvider:
    return JwtAuthProvider(settings=settings)


async def get_current_principal(request: Request) -> PrincipalContext:
    provider = getattr(request.app.state, "auth_provider", None)
    if provider is None:
        provider = get_auth_provider()
        request.app.state.auth_provider = provider

    try:
        return provider.require_principal(request.headers.get("Authorization"))
    except AuthenticationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def require_permission(permission: str):
    async def dependency(
        principal: Annotated[PrincipalContext, Depends(get_current_principal)],
    ) -> PrincipalContext:
        service = AuthorizationService()
        if not service.is_allowed(principal, permission):
            raise HTTPException(status_code=403, detail=f"permission '{permission}' required")
        return principal

    return dependency
