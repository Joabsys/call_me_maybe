
"""Command-line interface for the application."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

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
from src.pipeline import FunctionCallPipeline
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

        pipeline = FunctionCallPipeline(
            definitions,
            model,
            vocabulary,
        )

        function_calls = [
            pipeline.generate(prompt.prompt)
            for prompt in prompts
        ]

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
    except Exception as error:  # noqa: BLE001
        print(f"Unexpected error: {error}", file=sys.stderr)
        return 1

    print("Function-call generation successful.")
    print(f"Function definitions loaded: {len(definitions)}")
    print(f"Prompts processed: {len(function_calls)}")
    print(f"Output written to: {args.output}")

    return 0
