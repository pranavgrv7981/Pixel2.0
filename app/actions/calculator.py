"""Safe arithmetic evaluator for Pixel v2.

Evaluates bounded mathematical expressions using Python's AST module
with strict node whitelisting. Eliminates any use of eval() or exec().
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Union

from .types import ActionRequest, ActionResult, ActionVerification


# Maximum length of arithmetic expression string to evaluate
MAX_EXPRESSION_LENGTH = 200

# Exponent and magnitude bounds to prevent DoS attacks (e.g. 9**9999999)
MAX_EXPONENT = 1000
MAX_BASE_FOR_POW = 10000

# Supported binary operators
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorError(ValueError):
    """Raised when an expression is malformed, unsafe, or exceeds safety bounds."""
    pass


def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST node within strict safety boundaries."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        # Strict numeric types only: int or float (exclude bool, str, None, etc.)
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise CalculatorError(f"Non-numeric literal not permitted: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorError(f"Unsupported binary operator: {op_type.__name__}")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT or abs(left) > MAX_BASE_FOR_POW:
                raise CalculatorError(f"Power bounds exceeded: exponent must be <= {MAX_EXPONENT}")

        if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Division by zero is undefined.")

        op_func = _BIN_OPS[op_type]
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _eval_node(node.operand)
        return _UNARY_OPS[op_type](operand)

    # Any other AST node (Call, Attribute, Name, Subscript, Lambda, etc.) is rejected
    raise CalculatorError(f"Disallowed expression element: {type(node).__name__}")


def safe_calculate(expression: str) -> Union[int, float]:
    """Parse and evaluate a bounded arithmetic expression safely.

    Args:
        expression: A string containing a valid arithmetic expression.

    Returns:
        int or float result of the calculation.

    Raises:
        CalculatorError: If the expression contains disallowed tokens, syntax errors, or exceeds bounds.
        ZeroDivisionError: If division by zero is attempted.
    """
    if not expression or not isinstance(expression, str):
        raise CalculatorError("Expression must be a non-empty string.")

    cleaned = expression.strip()
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression exceeds maximum length of {MAX_EXPRESSION_LENGTH} characters.")

    try:
        parsed = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Syntax error in arithmetic expression: {exc.msg}") from exc

    result = _eval_node(parsed)

    # Format cleanly: if result is float with no fractional part, convert to int representation
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


async def handle_calculate(request: ActionRequest) -> ActionResult:
    """Action handler for arithmetic calculations."""
    expr = request.arguments.get("expression", "")
    if not expr:
        return ActionResult(
            success=False,
            error="Missing 'expression' argument for calculator action",
        )

    try:
        val = safe_calculate(str(expr))
        return ActionResult(
            success=True,
            output=str(val),
            verification=ActionVerification(
                verified=True,
                method="ast_eval",
                details=f"Expression '{expr}' evaluated safely to {val}",
            ),
        )
    except ZeroDivisionError as zde:
        return ActionResult(
            success=False,
            error=str(zde),
            verification=ActionVerification(
                verified=True,
                method="ast_eval",
                details="Zero division caught safely",
            ),
        )
    except CalculatorError as ce:
        return ActionResult(
            success=False,
            error=str(ce),
            verification=ActionVerification(
                verified=True,
                method="ast_eval",
                details="Unsafe or malformed expression rejected",
            ),
        )
    except Exception as exc:
        return ActionResult(
            success=False,
            error=f"Calculation error: {str(exc)}",
        )
