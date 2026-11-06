
"""Load and provide access to the tokenizer vocabulary."""

import json
from pathlib import Path
from typing import Any

from src.errors import VocabularyError


class Vocabulary:
    """Map token IDs to token strings and back."""

    def __init__(
        self,
        vocab_path: Path,
        tokenizer_path: Path,
    ) -> None:
        """Load the main vocabulary and additional tokenizer tokens."""
        vocab_data = self._read_json(vocab_path)
        tokenizer_data = self._read_json(tokenizer_path)

        if not isinstance(vocab_data, dict):
            raise VocabularyError(
                "The vocabulary file must contain a JSON object."
            )

        if not isinstance(tokenizer_data, dict):
            raise VocabularyError(
                "The tokenizer file must contain a JSON object."
            )

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._added_token_ids: set[int] = set()

        self._load_main_vocabulary(vocab_data)
        self._load_added_tokens(tokenizer_data)
        self._token_bytes_by_id = self._build_token_bytes()

    @staticmethod
    def _read_json(file_path: Path) -> Any:
        """Read and decode a JSON file."""
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise VocabularyError(
                f"Cannot load tokenizer data from '{file_path}': {error}"
            ) from error

    def _register_token(self, token: str, token_id: int) -> None:
        """Register a token in both lookup directions."""
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token

    def _load_main_vocabulary(
        self,
        vocab_data: dict[str, Any],
    ) -> None:
        """Load tokens from the main vocabulary file."""
        for token, token_id in vocab_data.items():
            if not isinstance(token, str):
                raise VocabularyError(
                    "Vocabulary token strings must be valid strings."
                )
            if type(token_id) is not int or token_id < 0:
                raise VocabularyError(
                    f"Invalid token ID for token '{token}'."
                )
            self._register_token(token, token_id)

    def _load_added_tokens(
        self,
        tokenizer_data: dict[str, Any],
    ) -> None:
        """Load additional tokens from the tokenizer configuration."""
        added_tokens = tokenizer_data.get("added_tokens", [])

        if not isinstance(added_tokens, list):
            raise VocabularyError(
                "The tokenizer added_tokens field must be a list."
            )

        for item in added_tokens:
            if not isinstance(item, dict):
                raise VocabularyError(
                    "Each added token must be a JSON object."
                )

            token = item.get("content")
            token_id = item.get("id")

            if not isinstance(token, str):
                raise VocabularyError(
                    "Each added token must have string content."
                )
            if type(token_id) is not int or token_id < 0:
                raise VocabularyError(
                    f"Invalid ID for added token '{token}'."
                )

            self._register_token(token, token_id)
            self._added_token_ids.add(token_id)

    def get_token_id(self, token: str) -> int | None:
        """Return a token ID, or None if the token is unknown."""
        return self.token_to_id.get(token)

    def get_token(self, token_id: int) -> str | None:
        """Return a token string, or None if the ID is unknown."""
        return self.id_to_token.get(token_id)

    def get_valid_token_ids(self) -> set[int]:
        """Return the IDs represented in the loaded vocabulary."""
        return set(self.id_to_token)

    @staticmethod
    def _build_byte_decoder() -> dict[str, int]:
        """Build the inverse GPT-style byte-to-Unicode mapping."""
        byte_values = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )

        unicode_values = byte_values.copy()
        next_unicode = 256

        for byte_value in range(256):
            if byte_value not in byte_values:
                byte_values.append(byte_value)
                unicode_values.append(next_unicode)
                next_unicode += 1

        return {
            chr(unicode_value): byte_value
            for byte_value, unicode_value in zip(
                byte_values,
                unicode_values,
            )
        }

    def _build_token_bytes(self) -> dict[int, bytes]:
        """Build the original byte representation of vocabulary tokens."""
        byte_decoder = self._build_byte_decoder()
        token_bytes_by_id: dict[int, bytes] = {}

        for token_id, token in self.id_to_token.items():
            if token_id in self._added_token_ids:
                continue

            try:
                token_bytes_by_id[token_id] = bytes(
                    byte_decoder[char]
                    for char in token
                )
            except KeyError:
                continue

        return token_bytes_by_id

    def get_token_bytes(self, token_id: int) -> bytes | None:
        """Return the original bytes of a regular vocabulary token."""
        return self._token_bytes_by_id.get(token_id)
