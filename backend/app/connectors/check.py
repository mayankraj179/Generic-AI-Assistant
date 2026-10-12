"""Checks a connector YAML before it is enabled, and shows what each
operation exposes.

    python -m app.connectors.check configs/examples/connectors/room_bookings.yaml
    python -m app.connectors.check configs/hr_live_assistant.yaml

The file is either one connector mapping or an assistant config with a
``connectors:`` list. For every operation it prints the tool name, the
request it builds, the parameters the model can set (its JSON Schema),
the parameters filled server-side (identity claim, settings), the labels a
caller needs, and whether each referenced env var is set (names only, never
values). It makes no network calls. Exits 1 if validation fails.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.config.connectors import ConnectorConfig
from app.config.knowledge_sources import ENV_REF_RE
from app.connectors.auth import _dotenv
from app.connectors.spec import tool_name


def _env_refs(value: Any) -> list[str]:
    if isinstance(value, str):
        match = ENV_REF_RE.fullmatch(value)
        return [match.group(1)] if match else []
    if isinstance(value, dict):
        return [name for item in value.values() for name in _env_refs(item)]
    return []


def describe(connector: ConnectorConfig) -> str:
    lines = [
        f"connector {connector.name!r} (type {connector.type}, shown as "
        f"'Source: {connector.display_name}, live')",
        f"  base_url: {connector.base_url}",
        f"  network: allow_private_network={connector.allow_private_network}"
        + (f", tls_ca_bundle={connector.tls_ca_bundle}" if connector.tls_ca_bundle else ""),
        f"  auth: {connector.auth.method}",
    ]
    for name in _env_refs(connector.auth.model_dump()):
        state = "set" if os.environ.get(name) or _dotenv().get(name) else "MISSING"
        lines.append(f"    ${{{name}}}: {state}")
    lines.append(f"  identity claim: {connector.identity.claim}")
    for op in connector.resolved_operations():
        hidden = []
        for pname, param in op.parameters.items():
            if param.identity:
                hidden.append(f"{pname} <- claim {param.claim or connector.identity.claim}")
            elif param.setting is not None:
                hidden.append(f"{pname} <- settings.{param.setting}")
        lines += [
            "",
            f"  tool {tool_name(connector.name, op.name)}",
            f"    request: {op.method}{' (read_only_post)' if op.read_only_post else ''} "
            f"{connector.base_url}{op.path}",
            f"    required labels (all): {sorted(op.required_labels)}",
            f"    model can set: {json.dumps(op.model_input_schema()['properties'])}",
            f"    filled server-side: {hidden or 'nothing'}",
        ]
        for tname, template in op.templates.items():
            lines.append(
                f"    template {tname} ({template.language}): "
                f"{template.separator.join(template.parts)}"
            )
        lines += [
            f"    response: root={op.response.root!r} fields={op.response.fields or 'ALL'} "
            f"max_items={op.response.max_items}",
            f"    pagination: {op.pagination.strategy}, max_pages={op.pagination.max_pages}; "
            f"timeout {op.timeout_seconds}s, cap {op.max_response_bytes} bytes",
        ]
    return "\n".join(lines)


def check(path: Path) -> list[ConnectorConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = (
        raw.get("connectors", []) if isinstance(raw, dict) and "assistant_id" in raw else [raw]
    )
    return [ConnectorConfig.model_validate(entry) for entry in entries]


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: python -m app.connectors.check <file.yaml>", file=sys.stderr)
        return 2
    try:
        connectors = check(Path(args[0]))
    except ValidationError as exc:
        # Location and message only: errors() also carries the rejected
        # input, which can be a literal secret.
        print("INVALID:", file=sys.stderr)
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(file)"
            print(f"  {location}: {error['msg']}", file=sys.stderr)
        return 1
    if not connectors:
        print("no connectors found in this file", file=sys.stderr)
        return 1
    print("\n\n".join(describe(c) for c in connectors))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
