from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PrincipalContext(BaseModel):
    """The authenticated caller's identity and access scope for one request.

    Labels are namespaced strings like "role:hr_admin", "user:e4471",
    "group:sales-emea", "dept:finance". Access predicates are compiled
    directly into SQL against these labels — never post-filtered.
    A principal with zero labels must be rejected by the caller (fail-closed),
    not defaulted to public/unrestricted access.
    """

    model_config = {"frozen": True}

    tenant_id: str
    principal_id: str
    labels: frozenset[str] = Field(default_factory=frozenset)
    clearance: int = 0

    @field_validator("labels", mode="before")
    @classmethod
    def _coerce_labels(cls, v: object) -> frozenset[str]:
        if isinstance(v, (list, set, tuple)):
            return frozenset(v)
        if isinstance(v, frozenset):
            return v
        raise TypeError("labels must be a list, set, tuple, or frozenset of strings")

    @field_validator("labels")
    @classmethod
    def _validate_label_format(cls, v: frozenset[str]) -> frozenset[str]:
        for label in v:
            if ":" not in label:
                raise ValueError(
                    f"label '{label}' is not namespaced — expected 'namespace:value'"
                )
        return v

    def has_label(self, label: str) -> bool:
        return label in self.labels

    def labels_in_namespace(self, namespace: str) -> frozenset[str]:
        prefix = f"{namespace}:"
        return frozenset(label for label in self.labels if label.startswith(prefix))

    @property
    def is_zero_label(self) -> bool:
        return len(self.labels) == 0
