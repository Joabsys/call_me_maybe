
"""Command-line interface for the application."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from src.constraints import FunctionCallConstraint
from src.decoder import generate_tokens
from src.errors import (
    DataValidationError,
    FileReadError,
    FileWriteError,
    InvalidJSONError,
    VocabularyError,
)
from src.files import (
    load_function_definitions,
    load_prompts,
    write_function_calls,
)
from src.llm import LanguageModel
from src.models import FunctionCall
from src.prompting import build_prompt
from src.vocabulary import Vocabulary


DEFAULT_FUNCTIONS_FILE = Path(
    "data/input/functions_definition.json"
)
DEFAULT_INPUT_FILE = Path(
    "data/input/function_calling_tests.json"
)
DEFAULT_OUTPUT_FILE = Path(
    "data/output/function_calls.json"
)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate structured function calls from prompts."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_FILE,
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path where the function calls JSON will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate function calls from the input prompts."""
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        definitions = load_function_definitions(
            args.functions_definition
        )
        prompts = load_prompts(args.input)

        model = LanguageModel()
        vocabulary = Vocabulary(
            model.get_vocab_file_path(),
            model.get_tokenizer_file_path(),
        )

        constraint = FunctionCallConstraint(
            definitions,
            model,
            vocabulary,
        )

        function_calls: list[FunctionCall] = []

        for prompt in prompts:
            prompt_text = build_prompt(
                prompt.prompt,
                definitions,
            )

            generated_token_ids = generate_tokens(
                prompt_token_ids=model.encode(prompt_text),
                get_logits=model.get_next_token_logits,
                get_allowed_tokens=constraint.get_allowed_tokens,
                stop_token_ids=constraint.end_token_ids,
                get_forced_tokens=constraint.get_forced_tokens,
            )

            generated_text = (
                '{"name": "' + model.decode(generated_token_ids)
            )

            call_data = json.loads(generated_text)

            if not isinstance(call_data, dict):
                raise ValueError(
                    "The generated function call must be a JSON object."
                )

            function_call = FunctionCall.model_validate(
                {
                    "prompt": prompt.prompt,
                    **call_data,
                }
            )

            function_calls.append(function_call)

        write_function_calls(
            args.output,
            function_calls,
        )

    except (
        DataValidationError,
        FileReadError,
        FileWriteError,
        InvalidJSONError,
        VocabularyError,
        ValueError,
        ValidationError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Function-call generation successful.")
    print(f"Function definitions loaded: {len(definitions)}")
    print(f"Prompts processed: {len(function_calls)}")
    print(f"Output written to: {args.output}")

    return 0
