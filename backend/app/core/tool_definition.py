from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class IdempotencyGuarantee(StrEnum):
    """How safe it is to retry this tool after an ambiguous failure.

    - supported: the downstream system accepts a caller-supplied idempotency
      key and will dedupe on it.
    - adapter_guaranteed: the tool adapter itself makes the call safe to
      retry (e.g. it does a read-check-write, or the underlying op is
      naturally idempotent like a PUT-by-id).
    - unsafe: retrying can duplicate the side effect. Must not be
      auto-retried; the confirmation-gate state machine should surface
      this as FAILED and let the caller decide.
    """

    SUPPORTED = "supported"
    ADAPTER_GUARANTEED = "adapter_guaranteed"
    UNSAFE = "unsafe"


class ToolKind(StrEnum):
    QUERY = "query"  # read-only, no confirmation gate
    ACTION = "action"  # side-effecting, goes through the confirmation-gate state machine


class ToolDefinition(BaseModel):
    """Describes one callable tool/capability an assistant config can enable.

    This is the contract between the generic core and a per-client "shell":
    a shell enables tools by name in its AssistantConfig YAML, and the core
    resolves them against the registry built from these definitions.
    """

    model_config = {"frozen": True}

    name: str
    description: str
    kind: ToolKind
    input_schema: dict = Field(default_factory=dict)  # JSON Schema for tool input
    idempotency: IdempotencyGuarantee = IdempotencyGuarantee.UNSAFE
    requires_labels: frozenset[str] = Field(default_factory=frozenset)

    def is_side_effecting(self) -> bool:
        return self.kind == ToolKind.ACTION
