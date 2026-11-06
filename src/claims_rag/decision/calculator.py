"""A calculator the LLM can call, so payable-amount arithmetic is done by Python and not by the model."""
import ast
import operator

from langchain_core.tools import tool

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
FUNCTIONS = {"min": min, "max": max, "round": round}


def evaluate(node: ast.AST) -> float:
    """Evaluate a parsed expression. Only numbers, + - * /, unary minus, parentheses and min/max/round are allowed."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        return BINARY_OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -evaluate(node.operand)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        return FUNCTIONS[node.func.id](*[evaluate(arg) for arg in node.args])
    raise ValueError("only numbers, + - * /, parentheses and min/max/round are allowed")


def calculate_expression(expression: str) -> str:
    """Compute an arithmetic expression. Returns the result, or an error message the LLM can read and fix."""
    try:
        result = evaluate(ast.parse(expression.strip(), mode="eval").body)
    except ZeroDivisionError:
        return "error: division by zero"
    except (SyntaxError, ValueError, TypeError) as error:
        return f"error: {error}"
    return str(round(result, 2))


# @tool turns the function into something the LLM can call: LangChain sends its name, docstring and arguments
# to the model. (Plain Python: write the JSON tool description by hand and call the function yourself.)
@tool
def calculate(expression: str) -> str:
    """Do exact arithmetic. Use it for every step of a payable amount, e.g. "(12000 - 500) * 0.8" or
    "min(7500, 9000)". Supports + - * /, parentheses and min, max, round. Returns the result rounded to 2 decimals."""
    return calculate_expression(expression)
