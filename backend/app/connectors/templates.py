"""Renders reviewed query templates (spec.TemplateSpec) with escaped values.

The model never writes query text. A template's text is fixed in config; a
``{param}`` placeholder becomes a literal of the template's language:

- jql:        "value" with \\ and " backslash-escaped (Jira)
- soql:       'value' with \\ ' " and control characters backslash-escaped
              (Salesforce SOQL string literal rules)
- odata:      'value' with ' doubled (OData v2/v4 $filter string literal)
- servicenow: the bare value (encoded queries have no quoting). ``^``
              separates clauses and has no escape, so a value containing it
              is rejected rather than "cleaned".

Numbers and booleans are written as literals. In every language, a value with
a line break or other control character is rejected: no legitimate field
lookup needs one.
"""

from __future__ import annotations

import re
from typing import Any

from app.connectors.auth import ConnectorCallError
from app.connectors.spec import IDENTIFIER_RE, TemplateLanguage, TemplateSpec

_PLACEHOLDER_RE = re.compile(r"\{([^{}]*)\}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def literal(value: Any, language: TemplateLanguage) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value) if isinstance(value, float) else str(value)
    text = str(value)
    if _CONTROL_RE.search(text):
        raise ConnectorCallError("invalid_arguments", "a value contains a control character")
    if language == "jql":
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if language == "soql":
        escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        return "'" + escaped + "'"
    if language == "odata":
        return "'" + text.replace("'", "''") + "'"
    # servicenow
    if "^" in text:
        raise ConnectorCallError("invalid_arguments", "a value contains '^'")
    return text


def _value(
    name: str, values: dict[str, Any], raw: set[str] | frozenset[str], template: TemplateSpec
) -> str:
    if name in raw:
        text = str(values[name])
        if not IDENTIFIER_RE.fullmatch(text):
            raise ConnectorCallError(
                "invalid_arguments", "a field-name setting is not an identifier"
            )
        return text
    return literal(values[name], template.language)


def render(
    template: TemplateSpec, values: dict[str, Any], raw: set[str] | frozenset[str] = frozenset()
) -> str:
    """Joins the parts whose placeholders all have values, each placeholder
    replaced by its escaped literal. ``raw`` names admin-set field names
    (identifiers, checked at config load and again here) inserted unquoted."""
    kept = []
    for part in template.parts:
        names = _PLACEHOLDER_RE.findall(part)
        if any(values.get(name) is None for name in names):
            continue
        kept.append(_PLACEHOLDER_RE.sub(lambda m: _value(m.group(1), values, raw, template), part))
    return template.separator.join(kept)
