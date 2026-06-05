"""A safe arithmetic calculator tool.

Evaluates expressions by walking a parsed AST and permitting only numeric literals
and a fixed set of arithmetic operators — never Python ``eval`` — so model-supplied
input cannot execute arbitrary code.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

from ..exceptions import ToolError

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        if isinstance(node.op, ast.Pow):
            exponent = _eval(node.right)
            if abs(exponent) > 100:
                raise ToolError("exponent too large")
            return _BIN_OPS[ast.Pow](_eval(node.left), exponent)
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ToolError("unsupported or unsafe expression")


def safe_eval(expression: str) -> float:
    """Safely evaluate an arithmetic expression. Raises :class:`ToolError` on bad input."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc}") from exc
    return _eval(tree.body)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression such as '2 + 3 * 4' or '(10 - 4) / 2'.

    Supports + - * / // % ** and parentheses over numbers. Use this for any arithmetic
    rather than computing it yourself.
    """
    return str(safe_eval(expression))


__all__ = ["calculator", "safe_eval"]
