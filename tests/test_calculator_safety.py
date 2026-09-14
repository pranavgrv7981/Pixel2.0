"""Tests verifying the AST-based safe calculator and malicious injection immunity."""

from __future__ import annotations

import pytest
from app.actions.calculator import safe_calculate, handle_calculate, CalculatorError
from app.actions.types import ActionRequest


@pytest.mark.unit
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", 4),
        ("25 * 8", 200),
        ("(10 + 5) * 2", 30),
        ("144 / 12", 12),
        ("10 / 4", 2.5),
        ("17 % 5", 2),
        ("2 ** 8", 256),
        ("10.5 + 2.5", 13),
        ("10.25 - 0.5", 9.75),
        ("-5 + 10", 5),
        ("+42", 42),
        ("((2 + 3) * (4 - 1)) / 3", 5),
    ],
)
def test_safe_calculate_valid_math(expression, expected):
    """Safe arithmetic evaluator handles standard mathematical expressions."""
    result = safe_calculate(expression)
    assert result == expected


@pytest.mark.unit
def test_safe_calculate_zero_division():
    """Division by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError, match="Division by zero is undefined"):
        safe_calculate("100 / 0")

    with pytest.raises(ZeroDivisionError, match="Division by zero is undefined"):
        safe_calculate("50 % 0")


@pytest.mark.unit
@pytest.mark.parametrize(
    "malicious_expr",
    [
        "__import__('os').system('calc')",
        "__import__('sys').exit(0)",
        "open('/etc/passwd')",
        "open('C:/Windows/win.ini')",
        "exec('x = 1')",
        "eval('2 + 2')",
        "(lambda x: x + 1)(2)",
        "[c for c in ().__class__.__base__.__subclasses__()]",
        "print('hello')",
        "os.system('whoami')",
        "import os",
        "x = 5",
        "'hello' + 'world'",
        "True + 1",
        "None",
        "{'a': 1}",
        "[1, 2, 3]",
        "(1, 2)",
        "2 & 3",
        "2 | 3",
        "2 ^ 3",
        "~5",
        "2 << 3",
        "2 >> 1",
    ],
)
def test_safe_calculate_rejects_malicious_code(malicious_expr):
    """AST evaluator strictly rejects non-whitelisted code, identifiers, and attacks."""
    with pytest.raises(CalculatorError):
        safe_calculate(malicious_expr)


@pytest.mark.unit
def test_safe_calculate_dos_exponent_rejection():
    """Expressions with huge exponents that cause denial-of-service are rejected."""
    with pytest.raises(CalculatorError, match="Power bounds exceeded"):
        safe_calculate("9 ** 999999999")

    with pytest.raises(CalculatorError, match="Power bounds exceeded"):
        safe_calculate("2 ** 50000")

    with pytest.raises(CalculatorError, match="Power bounds exceeded"):
        safe_calculate("99999 ** 2")


@pytest.mark.unit
def test_safe_calculate_max_length():
    """Expressions exceeding length limit are rejected."""
    long_expr = "1 + " * 100 + "1"
    with pytest.raises(CalculatorError, match="exceeds maximum length"):
        safe_calculate(long_expr)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_calculate_action():
    """Action handler executes valid expressions and reports verification details."""
    req = ActionRequest(action="calculate", arguments={"expression": "50 * 4"})
    result = await handle_calculate(req)
    assert result.success is True
    assert result.output == "200"
    assert result.verification is not None
    assert result.verification.verified is True
    assert result.verification.method == "ast_eval"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_calculate_action_error():
    """Action handler safely catches calculation errors without throwing exceptions."""
    req = ActionRequest(action="calculate", arguments={"expression": "100 / 0"})
    result = await handle_calculate(req)
    assert result.success is False
    assert "division by zero" in result.error.lower()
