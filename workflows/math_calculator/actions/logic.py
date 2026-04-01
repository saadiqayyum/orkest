"""
Math workflow logic functions.

These functions follow the dg-team pattern:
1. Receive ctx and validated Pydantic input
2. Report meaningful input/output using ctx.report_input/report_output
3. Return Pydantic model with status for routing
"""
from .schemas import (
    ValidateInput,
    ValidateOutput,
    CalculateInput,
    CalculateOutput,
    FormatInput,
    FormatOutput,
)


VALID_OPERATIONS = {"add", "subtract", "multiply", "divide"}


async def validate_numbers(
    ctx,
    params: ValidateInput
) -> ValidateOutput:
    """Validate that numbers and operation are valid."""

    # Report what we received
    ctx.report_input({
        "a": params.a,
        "b": params.b,
        "operation": params.operation,
    })

    # Validate operation
    operation = params.operation.lower().strip()
    if operation not in VALID_OPERATIONS:
        ctx.report_output({
            "status": "invalid",
            "error": f"Invalid operation '{operation}'. Must be one of: {VALID_OPERATIONS}"
        })

        return ValidateOutput(
            status="invalid",
            error=f"Invalid operation '{operation}'. Must be one of: {VALID_OPERATIONS}"
        )

    # Validate division by zero
    if operation == "divide" and params.b == 0:
        ctx.report_output({
            "status": "invalid",
            "error": "Cannot divide by zero"
        })

        return ValidateOutput(
            status="invalid",
            error="Cannot divide by zero"
        )

    # Validation passed
    ctx.report_output({
        "validated_a": params.a,
        "validated_b": params.b,
        "validated_operation": operation,
    })

    return ValidateOutput(
        status="success",
        validated_a=params.a,
        validated_b=params.b,
        validated_operation=operation,
    )


async def calculate(
    ctx,
    params: CalculateInput
) -> CalculateOutput:
    """Perform the actual calculation."""

    ctx.report_input({
        "expression": f"{params.validated_a} {params.validated_operation} {params.validated_b}"
    })

    a = params.validated_a
    b = params.validated_b
    op = params.validated_operation

    # Perform calculation
    if op == "add":
        result = a + b
    elif op == "subtract":
        result = a - b
    elif op == "multiply":
        result = a * b
    elif op == "divide":
        result = a / b
    else:
        ctx.report_output({
            "status": "error",
            "error": f"Unknown operation: {op}"
        })
        return CalculateOutput(
            status="error",
            error=f"Unknown operation: {op}"
        )

    ctx.report_output({
        "result": result,
        "operation": op,
    })

    return CalculateOutput(
        status="success",
        result=result,
        operation_performed=f"{a} {op} {b} = {result}",
    )


async def format_result(
    ctx,
    params: FormatInput
) -> FormatOutput:
    """Format the calculation result for display."""

    ctx.report_input({
        "result": params.result,
        "operation": params.validated_operation,
    })

    # Create human-readable format
    op_symbols = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
    }
    symbol = op_symbols.get(params.validated_operation, "?")

    formatted = f"{params.validated_a} {symbol} {params.validated_b} = {params.result}"

    ctx.report_output({
        "formatted_result": formatted,
    })

    return FormatOutput(
        status="success",
        formatted_result=formatted,
        calculation_details={
            "operand_a": params.validated_a,
            "operand_b": params.validated_b,
            "operation": params.validated_operation,
            "result": params.result,
        }
    )
