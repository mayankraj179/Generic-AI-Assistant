from __future__ import annotations

from app.core.principal import PrincipalContext


class DevelopmentAuthorizationPolicy:
    """Every valid authenticated principal gets a common development permission set."""

    default_permissions: frozenset[str] = frozenset(
        {
            "assistant:use",
            "conversation:read",
            "conversation:write",
            "retrieval:use",
            "tool:invoke",
            # Distinct from "assistant:use" on purpose so /admin/ingest is
            # never reachable via the same permission an ordinary chat user
            # already has — kept in this same shared dev-permission set only
            # because this policy grants everyone the same permissions today
            # (no real admin/RBAC distinction exists yet, same documented
            # limitation as employee/manager-specific authorization). When
            # real role-based policy is added, only this grant needs to
            # change — the route already checks a separate permission
            # string, not assistant:use.
            "admin:ingest",
        }
    )
    default_labels: frozenset[str] = frozenset({"role:authenticated", "scope:development"})

    def permissions_for(self, principal: PrincipalContext) -> frozenset[str]:
        if principal.is_zero_label:
            return frozenset()
        return self.default_permissions

    def is_allowed(self, principal: PrincipalContext, permission: str) -> bool:
        return permission in self.permissions_for(principal)


class AuthorizationService:
    """Thin authorization boundary to keep business logic independent of IdP-specific claims."""

    def __init__(self, policy: DevelopmentAuthorizationPolicy | None = None) -> None:
        self._policy = policy or DevelopmentAuthorizationPolicy()

    def is_allowed(self, principal: PrincipalContext, permission: str) -> bool:
        return self._policy.is_allowed(principal, permission)
