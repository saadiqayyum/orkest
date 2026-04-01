"""
Pydantic schemas for Math workflow nodes.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


# --- Validate Node Schemas ---

class ValidateInput(BaseModel):
    """Input for validation node."""
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


class ValidateOutput(BaseModel):
    """Output from validation node."""
    status: Literal["success", "invalid"] = Field(..., description="Routing status")
    validated_a: Optional[float] = None
    validated_b: Optional[float] = None
    validated_operation: Optional[str] = None
    error: Optional[str] = None


# --- Calculate Node Schemas ---

class CalculateInput(BaseModel):
    """Input for calculation node."""
    validated_a: float
    validated_b: float
    validated_operation: str


class CalculateOutput(BaseModel):
    """Output from calculation node."""
    status: Literal["success", "error"] = Field(..., description="Routing status")
    result: Optional[float] = None
    operation_performed: Optional[str] = None
    error: Optional[str] = None


# --- Format Node Schemas ---

class FormatInput(BaseModel):
    """Input for format node."""
    validated_a: float
    validated_b: float
    validated_operation: str
    result: float


class FormatOutput(BaseModel):
    """Output from format node."""
    status: Literal["success"] = "success"
    formatted_result: str = Field(..., description="Human-readable result string")
    calculation_details: dict = Field(default_factory=dict)
