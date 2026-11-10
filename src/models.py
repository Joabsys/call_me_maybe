
"""Pydantic models for function definitions, prompts, and function calls."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ParameterType = Literal["string", "number", "integer"]
ReturnType = Literal["string", "number", "integer", "boolean"]
EnumValue = str | int | float


class ParameterDefinition(BaseModel):
    """Describe a function parameter and its validation constraints."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: ParameterType
    description: str | None = None
    enum: list[EnumValue] | None = None
    required: bool = True


class ReturnDefinition(BaseModel):
    """Describe the return type of a function."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: ReturnType


class FunctionDefinition(BaseModel):
    """Represent a function that the language model can call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition | None = None


class PromptInput(BaseModel):
    """Represent a natural-language prompt received by the application."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str


class FunctionCall(BaseModel):
    """Represent the structured function call produced by the application."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str = Field(min_length=1)
    parameters: dict[str, Any]
