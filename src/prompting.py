"""Build prompts for structured function-call generation."""

import json

from src.models import FunctionDefinition

INSTRUCTIONS = (
    "You are a function-calling assistant.\n"
    "Choose the function that best matches the user's request.\n"
    "Use the exact function name and parameter names.\n"
    "Respect the declared parameter types and enum values.\n"
    "Copy string values exactly, including punctuation, quotes and braces.\n"
    "A string value holds only the content the user wants processed. "
    "Never include the words that introduce it, such as "
    "'Translate this:' or 'Convert:'.\n"
    "Do not calculate or execute the requested operation.\n"
    'Complete the JSON object with exactly the keys "name" and '
    '"parameters". Do not add Markdown, explanations or extra keys.\n'
)


EXAMPLES = (
    "Examples (illustrative functions, not in the list below):\n"
    'User request:\n"Translate this: Good morning"\n'
    '{"name": "fn_translate", "parameters": {"text": "Good morning"}}\n\n'
    'User request:\n"Shout this: He said \\"stop\\" loudly"\n'
    '{"name": "fn_shout", "parameters": '
    '{"text": "He said \\"stop\\" loudly"}}\n'
)


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
        separators=(",", ":"),
    )

    return (
        f"{INSTRUCTIONS}\n"
        f"{EXAMPLES}\n"
        f"Available functions:\n{functions_json}\n\n"
        f"User request:\n{json.dumps(prompt, ensure_ascii=False)}\n"
        '{"name": "'
    )
