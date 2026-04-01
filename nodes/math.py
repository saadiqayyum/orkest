"""
Simple math node implementations for meta-workflow testing.
"""
from typing import Any, Dict


async def add(ctx, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Add two numbers."""
    a = input_data.get("a", 0)
    b = input_data.get("b", 0)
    result = a + b

    ctx.report_input({"a": a, "b": b})
    ctx.report_output({"sum": result})

    return {"sum": result, "status": "completed"}


async def multiply(ctx, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Multiply two numbers."""
    x = input_data.get("x", 0)
    y = input_data.get("y", 0)
    result = x * y

    ctx.report_input({"x": x, "y": y})
    ctx.report_output({"product": result})

    return {"product": result, "status": "completed"}


async def add_results(ctx, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add the results from add_step and multiply_step.
    The state will contain 'sum' from add_step and 'product' from multiply_step.
    """
    sum_value = input_data.get("sum", 0)
    product_value = input_data.get("product", 0)
    final_result = sum_value + product_value

    ctx.report_input({"sum": sum_value, "product": product_value})
    ctx.report_output({"final_result": final_result})

    return {"final_result": final_result, "status": "completed"}
