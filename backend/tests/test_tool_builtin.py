from __future__ import annotations

import pytest

from app.tools.builtin import (
    TOOL_REGISTRY,
    ToolExecutionError,
    calculate,
    current_datetime,
    execute_tool,
)

# ---------------------------------------------------------------------------
# current_datetime
# ---------------------------------------------------------------------------


def test_current_datetime_defaults_to_utc():
    result = current_datetime()
    assert result["timezone"] == "UTC"
    assert result["iso"].endswith("+00:00")
    assert result["day_of_week"] in (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    )
    assert "error" not in result


def test_current_datetime_with_explicit_timezone():
    result = current_datetime("America/New_York")
    assert result["timezone"] == "America/New_York"
    assert "-04:" in result["iso"] or "-05:" in result["iso"]  # EDT/EST offset
    assert "error" not in result


def test_current_datetime_unknown_timezone_returns_structured_error():
    result = current_datetime("Not/AZone")
    assert "error" in result
    assert "Not/AZone" in result["error"]


def test_current_datetime_with_valid_format_hint():
    result = current_datetime("UTC", format="%Y-%m-%d")
    assert "formatted" in result
    assert len(result["formatted"]) == len("YYYY-MM-DD")


def test_current_datetime_with_invalid_format_hint_returns_structured_error():
    result = current_datetime("UTC", format="%Q")
    assert "error" in result


# ---------------------------------------------------------------------------
# calculate — valid expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4.0),
        ("2 - 5", -3.0),
        ("3 * 4", 12.0),
        ("10 / 4", 2.5),
        ("(2 + 3) * 4", 20.0),
        ("-5 + 3", -2.0),
        ("2 * (3 + (4 - 1))", 12.0),
    ],
)
def test_calculate_valid_arithmetic(expression, expected):
    result = calculate(expression)
    assert "error" not in result
    assert result["kind"] == "arithmetic"
    assert result["result"] == pytest.approx(expected)


def test_calculate_supports_unary_operators():
    # "2 ++ 3" is 2 + (+3) = 5 under standard unary-operator precedence,
    # same as any ordinary calculator grammar — not malformed input.
    assert calculate("2 ++ 3")["result"] == pytest.approx(5.0)
    assert calculate("2 - -3")["result"] == pytest.approx(5.0)


def test_calculate_matches_real_finance_question():
    result = calculate("(96.5 - 79.1) / 79.1 * 100")
    assert result["result"] == pytest.approx(21.997, abs=1e-3)


def test_calculate_date_difference():
    result = calculate("days between 2026-01-01 and 2026-09-23")
    assert result["kind"] == "date_difference"
    assert result["result"] == 265


def test_calculate_date_difference_is_case_insensitive_and_tolerates_whitespace():
    result = calculate("  Days Between 2026-01-01 and 2026-01-11  ")
    assert result["result"] == 10


# ---------------------------------------------------------------------------
# calculate — invalid/malformed expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "   ",
        "2 +",
        "(2 + 3",
        "2 + 3)",
        "2 3",
        "days between 2026-13-40 and 2026-01-01",  # invalid calendar date
    ],
)
def test_calculate_malformed_expressions_return_structured_error(expression):
    result = calculate(expression)
    assert "error" in result
    assert "result" not in result


def test_calculate_division_by_zero_returns_structured_error():
    result = calculate("1 / 0")
    assert "error" in result
    assert "zero" in result["error"]


# ---------------------------------------------------------------------------
# calculate — the tool must genuinely reject eval-injection-shaped input,
# not merely have never been tried against one. Each of these contains at
# least one character (letters, underscores, quotes, brackets) the
# tokenizer never recognizes, so they fail before any evaluation is even
# attempted — there is no code path here that could reach a name lookup,
# attribute access, or function call.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "__import__('os').system(\"rm -rf /\")",
        "exec('print(1)')",
        "eval('1+1')",
        "().__class__.__mro__[1].__subclasses__()",
        "open('/etc/passwd').read()",
        "2 + __builtins__",
        "[x for x in range(10)]",
        "os.system('ls')",
    ],
)
def test_calculate_rejects_eval_injection_shaped_input(expression):
    result = calculate(expression)
    assert "error" in result
    assert "result" not in result


def test_calculate_non_string_input_returns_structured_error():
    result = calculate(None)  # type: ignore[arg-type]
    assert "error" in result


# ---------------------------------------------------------------------------
# execute_tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_dispatches_calculate():
    result = await execute_tool("calculate", {"expression": "2 + 2"})
    assert result["result"] == 4.0


@pytest.mark.asyncio
async def test_execute_tool_dispatches_current_datetime():
    result = await execute_tool("current_datetime", {})
    assert result["timezone"] == "UTC"


@pytest.mark.asyncio
async def test_execute_tool_unknown_name_raises_tool_execution_error():
    with pytest.raises(ToolExecutionError):
        await execute_tool("not_a_real_tool", {})


@pytest.mark.asyncio
async def test_execute_tool_wrong_argument_name_returns_structured_error_not_crash():
    result = await execute_tool("calculate", {"expr": "2 + 2"})  # wrong kwarg name
    assert "error" in result


# ---------------------------------------------------------------------------
# Registry shape sanity
# ---------------------------------------------------------------------------


def test_registry_contains_exactly_the_two_built_in_tools():
    assert set(TOOL_REGISTRY) == {"current_datetime", "calculate"}


def test_both_tool_definitions_are_query_kind_with_strict_schema():
    for definition in TOOL_REGISTRY.values():
        assert definition.kind.value == "query"
        assert definition.input_schema["additionalProperties"] is False
        assert definition.requires_labels == frozenset()
