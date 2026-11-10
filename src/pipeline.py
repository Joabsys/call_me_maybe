
"""Orchestrate constrained function-call generation."""

import json

from src.constraints import FunctionCallConstraint
from src.decoder import generate_tokens
from src.llm import LanguageModel
from src.models import FunctionCall, FunctionDefinition
from src.prompting import build_prompt
from src.vocabulary import Vocabulary


class FunctionCallPipeline:
    """Generate validated function calls from natural-language prompts."""

    def __init__(
        self,
        definitions: list[FunctionDefinition],
        model: LanguageModel,
        vocabulary: Vocabulary,
    ) -> None:
        """Prepare the dependencies required for function-call generation."""
        self.definitions = definitions
        self.definitions_by_name = {
            definition.name: definition for definition in definitions
        }
        self.model = model
        self.constraint = FunctionCallConstraint(
            definitions,
            model,
            vocabulary,
        )

    def generate(self, prompt: str) -> FunctionCall:
        """Generate and validate a function call for a user prompt."""
        header_token_ids = generate_tokens(
            prompt_token_ids=self.model.encode(
                build_prompt(prompt, self.definitions)
            ),
            get_logits=self.model.get_next_token_logits,
            get_allowed_tokens=self.constraint.get_allowed_tokens,
            stop_token_ids=self.constraint.end_token_ids,
            get_forced_tokens=self.constraint.get_forced_tokens,
            should_stop=lambda token_ids: (
                self.constraint.identify_function(token_ids) is not None
            ),
        )
        function_name = self.constraint.identify_function(header_token_ids)
        if function_name is None:
            raise ValueError("The function name could not be generated.")

        generated_token_ids = generate_tokens(
            prompt_token_ids=self.model.encode(
                build_prompt(prompt, [self.definitions_by_name[function_name]])
            ),
            get_logits=self.model.get_next_token_logits,
            get_allowed_tokens=self.constraint.get_allowed_tokens,
            stop_token_ids=self.constraint.end_token_ids,
            get_forced_tokens=self.constraint.get_forced_tokens,
            initial_token_ids=header_token_ids,
        )

        generated_text = (
            '{"name": "' + self.model.decode(generated_token_ids)
        )

        call_data = json.loads(generated_text)

        if not isinstance(call_data, dict):
            raise ValueError(
                "The generated function call must be a JSON object."
            )

        return FunctionCall.model_validate(
            {
                "prompt": prompt,
                **call_data,
            }
        )
