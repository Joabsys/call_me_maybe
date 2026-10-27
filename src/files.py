
"""Read, validate, and write JSON files."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.errors import (
    DataValidationError,
    FileReadError,
    FileWriteError,
    InvalidJSONError,
)
from src.models import FunctionCall, FunctionDefinition, PromptInput


def read_json_file(file_path: Path) -> Any:
    """Read a JSON file and return its decoded content."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileReadError(
            f"Cannot read file '{file_path}': {error}"
        ) from error

    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise InvalidJSONError(
            f"Invalid JSON in '{file_path}': {error}"
        ) from error


def load_function_definitions(
    file_path: Path,
) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    data = read_json_file(file_path)

    if not isinstance(data, list):
        raise DataValidationError(
            f"Expected a JSON list in '{file_path}'."
        )

    try:
        return [
            FunctionDefinition.model_validate(item)
            for item in data
        ]
    except ValidationError as error:
        raise DataValidationError(
            f"Invalid function definition in '{file_path}': {error}"
        ) from error


def load_prompts(file_path: Path) -> list[PromptInput]:
    """Load and validate prompts from a JSON file."""
    data = read_json_file(file_path)

    if not isinstance(data, list):
        raise DataValidationError(
            f"Expected a JSON list in '{file_path}'."
        )

    try:
        return [
            PromptInput.model_validate(item)
            for item in data
        ]
    except ValidationError as error:
        raise DataValidationError(
            f"Invalid prompt data in '{file_path}': {error}"
        ) from error


def write_function_calls(
    file_path: Path,
    function_calls: list[FunctionCall],
) -> None:
    """Write function calls as a JSON list."""
    data = [
        function_call.model_dump(mode="json")
        for function_call in function_calls
    ]

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise FileWriteError(
            f"Cannot write file '{file_path}': {error}"
        ) from error
