
"""Provide a small interface to the language model SDK."""

from pathlib import Path
from typing import cast

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class LanguageModel:
    """Wrap the public API of the provided language model SDK."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        """Load the requested language model."""
        self.model = Small_LLM_Model(model_name)

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token IDs."""
        encoded = self.model.encode(text)
        return [int(token_id) for token_id in encoded.tolist()[0]]

    def get_next_token_logits(
        self,
        input_ids: list[int],
    ) -> list[float]:
        """Return the logits for the next token."""
        logits = self.model.get_logits_from_input_ids(input_ids)
        return cast(list[float], logits)

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text."""
        decoded = self.model.decode(token_ids)
        return cast(str, decoded)

    def get_vocab_file_path(self) -> Path:
        """Return the path to the model vocabulary file."""
        return Path(self.model.get_path_to_vocab_file())

    def get_tokenizer_file_path(self) -> Path:
        """Return the path to the tokenizer configuration file."""
        return Path(self.model.get_path_to_tokenizer_file())
