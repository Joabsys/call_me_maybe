
"""Build prompts for structured function-call generation."""

import json

from src.models import FunctionDefinition


def build_prompt(
    prompt: str,
    definitions: list[FunctionDefinition],
) -> str:
    """Build a prompt from the user request and available functions."""
    function_data = [
        definition.model_dump(mode="json", exclude_none=True)
        for definition in definitions
    ]

    functions_json = json.dumps(
        function_data,
        ensure_ascii=False,
        indent=2,
    )

    return (
        "You are a function-calling assistant.\n"
        "Choose the function that best matches the user's request.\n"
        "Extract the arguments using the provided function definitions.\n"
        "When a request contains a label or instruction before the "
        "requested string value, exclude that label or instruction.\n"
        "Preserve the requested string content exactly, including "
        "punctuation, quotes, and braces.\n"
        "Choose the function that best matches the user's request.\n"
        "Extract the arguments using the provided function definitions.\n"
        "When a request contains a label or instruction before the "
        "requested string value, exclude that label or instruction.\n"
        "Preserve the requested string content exactly, including "
        "punctuation, quotes, and braces.\n"
        "Use the exact function name and parameter names.\n"
        "Use the exact function name and parameter names.\n"
        "Respect the declared parameter types and enum values.\n"
        "Do not calculate or execute the requested operation.\n"
        "Complete the JSON object with exactly these keys:\n"
        '"name", "parameters".\n'
        "Do not include Markdown, explanations, or extra keys.\n\n"
        f"Available functions:\n{functions_json}\n\n"
        f"User request:\n{json.dumps(prompt, ensure_ascii=False)}\n\n"
        '{"name": "'
    )
