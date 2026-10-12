"""The connector interface: a connector type declares read-only operations,
and an assistant config enables some of them (app/config/connectors.py).

Each enabled operation becomes an ordinary tool (ToolDefinition) on the
existing tool-calling path; app/connectors/runtime.py executes it. There is
no write path: the executor only ever issues GET requests, and an operation
has no field that could name another method.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# OpenAI-style function names allow [A-Za-z0-9_-]{1,64}; Gemini is stricter
# only about the first character, which a connector name already satisfies.
TOOL_NAME_SEPARATOR = "__"
MAX_TOOL_NAME_LENGTH = 64


@dataclass(frozen=True)
class ConnectorOperation:
    """One read-only operation of a connector type.

    ``parameters`` is a JSON Schema ``properties`` map covering every input,
    including the identity ones. ``identity_params`` are filled server-side
    from the validated Principal (never by the model): they are removed from
    the schema the model sees, and ignored if the model sends them anyway.
    ``path`` placeholders and ``query_params`` name entries of ``parameters``.
    ``result_fields`` limits which top-level response fields reach the model
    (empty means all of them). ``required_labels`` is a code-level baseline;
    the assistant config adds its own and the caller must hold all of them.
    """

    name: str
    description: str
    path: str
    parameters: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    identity_params: frozenset[str] = frozenset()
    query_params: tuple[str, ...] = ()
    result_fields: tuple[str, ...] = ()
    timeout_seconds: float = 5.0
    max_response_bytes: int = 64_000
    required_labels: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        unknown_identity = self.identity_params - self.parameters.keys()
        if unknown_identity:
            raise ValueError(f"identity params not in parameters: {sorted(unknown_identity)}")
        overlap = self.identity_params & set(self.query_params)
        if overlap:
            raise ValueError(f"identity params cannot also be query params: {sorted(overlap)}")

    def model_input_schema(self) -> dict[str, Any]:
        """The JSON Schema the model sees: identity params stripped."""
        return {
            "type": "object",
            "properties": {
                name: dict(schema)
                for name, schema in self.parameters.items()
                if name not in self.identity_params
            },
            "required": [name for name in self.required if name not in self.identity_params],
            "additionalProperties": False,
        }


@dataclass(frozen=True)
class ConnectorType:
    """A kind of external system the framework knows how to call. Adding
    one is a code change (an entry in app/connectors/catalog.py); enabling
    one for an assistant is config only."""

    type: str
    description: str
    operations: Mapping[str, ConnectorOperation]


def tool_name(connector_name: str, operation_name: str) -> str:
    return f"{connector_name}{TOOL_NAME_SEPARATOR}{operation_name}"
