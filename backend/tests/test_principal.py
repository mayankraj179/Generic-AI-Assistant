import pytest
from pydantic import ValidationError

from app.core.principal import PrincipalContext


def test_creates_principal_with_valid_labels():
    principal = PrincipalContext(
        tenant_id="bitwise-global",
        principal_id="e4471",
        labels=["role:hr_admin", "dept:finance"],
        clearance=2,
    )
    assert principal.has_label("role:hr_admin")
    assert not principal.has_label("role:sales")
    assert principal.labels_in_namespace("dept") == frozenset({"dept:finance"})


def test_rejects_unnamespaced_label():
    with pytest.raises(ValidationError):
        PrincipalContext(
            tenant_id="bitwise-global",
            principal_id="e4471",
            labels=["hr_admin"],
        )


def test_zero_label_principal_is_flagged():
    principal = PrincipalContext(tenant_id="bitwise-global", principal_id="e4471")
    assert principal.is_zero_label


def test_principal_is_frozen():
    principal = PrincipalContext(tenant_id="bitwise-global", principal_id="e4471")
    with pytest.raises(ValidationError):
        principal.principal_id = "someone-else"
