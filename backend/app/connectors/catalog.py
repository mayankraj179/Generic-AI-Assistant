"""Connector types this framework can call. A config's ``connectors[].type``
must name one of these, and its operations must be declared here; anything
else fails config loading."""

from __future__ import annotations

from app.connectors.base import ConnectorOperation, ConnectorType

HR_API = ConnectorType(
    type="hr_api",
    description=(
        "Read-only HR system REST API (leave balances). The mock in "
        "scripts/mock_hr_api.py implements it for local development."
    ),
    operations={
        "get_leave_balance": ConnectorOperation(
            name="get_leave_balance",
            description=(
                "Gets the signed-in employee's current leave balance (days "
                "available, used and accrued, per leave type) live from the HR "
                "system. It always returns the balance of the user who is "
                "asking and cannot look up any other employee."
            ),
            path="/api/v1/employees/{employee_id}/leave-balance",
            parameters={
                "employee_id": {"type": "string", "description": "HR employee id."},
                "leave_type": {
                    "type": "string",
                    "enum": ["annual", "sick", "casual"],
                    "description": "Optional: one leave type. Omit to get every type.",
                },
            },
            identity_params=frozenset({"employee_id"}),
            query_params=("leave_type",),
            result_fields=("as_of", "balances"),
            timeout_seconds=5.0,
            max_response_bytes=16_000,
        ),
    },
)

CONNECTOR_TYPES: dict[str, ConnectorType] = {HR_API.type: HR_API}
