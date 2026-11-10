"""Constrain function-call generation to the declared JSON schema."""

import json

from src.llm import LanguageModel
from src.models import FunctionDefinition
from src.schema_parser import _SchemaParser
from src.vocabulary import Vocabulary


def is_plain_string_token(token_bytes: bytes) -> bool:
    """Return True if a token is valid inside a JSON string body."""
    if any(
        byte < 0x20 or byte in (0x22, 0x5C)
        for byte in token_bytes
    ):
        return False
    try:
        token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def get_generated_bytes(
    generated_token_ids: list[int],
    vocabulary: Vocabulary,
) -> bytes | None:
    """Return the concatenated bytes represented by generated tokens."""
    result = bytearray()
    for token_id in generated_token_ids:
        token_bytes = vocabulary.get_token_bytes(token_id)
        if token_bytes is None:
            return None
        result.extend(token_bytes)
    return bytes(result)


class FunctionCallConstraint:
    """Apply Teleport, Bridge Fast-Forward, and schema-based Type Shield."""

    def __init__(
        self,
        definitions: list[FunctionDefinition],
        model: LanguageModel,
        vocabulary: Vocabulary,
    ) -> None:
        """Prepare schema lookup and tokenization dependencies."""
        if not definitions:
            raise ValueError("At least one function definition is required.")

        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError("Function names must be unique.")

        for definition in definitions:
            for parameter in definition.parameters.values():
                if parameter.enum is None:
                    continue
                for value in parameter.enum:
                    if parameter.type == "string" and type(value) is not str:
                        raise ValueError(
                            "String enums must contain only strings."
                        )
                    if parameter.type == "number" and type(value) not in (
                        int,
                        float,
                    ):
                        raise ValueError(
                            "Number enums must contain only numbers."
                        )
                    if parameter.type == "integer" and type(value) is not int:
                        raise ValueError(
                            "Integer enums must contain only integers"
                        )

        self.definitions_by_name = {
            definition.name: definition for definition in definitions
        }
        self.name_bytes = {
            name: name.encode("utf-8") for name in names
        }
        self.enum_literals_by_name = {
            definition.name: {
                parameter_name: tuple(
                    sorted(
                        {
                            json.dumps(
                                float(value)
                                if parameter.type == "number"
                                else value,
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode("utf-8")
                            for value in parameter.enum or []
                        },
                        key=len,
                        reverse=True,
                    )
                )
                for parameter_name, parameter
                in definition.parameters.items()
            }
            for definition in definitions
        }
        self.model = model
        self.vocabulary = vocabulary
        self.bridge_bytes = b', "parameters": {'

        end_token_id = vocabulary.get_token_id("<|im_end|>")
        if end_token_id is None:
            raise ValueError("The message-ending token is missing.")
        self.end_token_ids: set[int] = {end_token_id}

        self.token_bytes_by_id = {
            token_id: token_bytes
            for token_id in vocabulary.get_valid_token_ids()
            if (token_bytes := vocabulary.get_token_bytes(token_id))
        }

        self.tokens_by_first_byte: dict[int, list[tuple[int, bytes]]] = {}
        self.special_by_first_byte: dict[int, list[tuple[int, bytes]]] = {}
        self.plain_token_ids: set[int] = set()

        for token_id, token_bytes in self.token_bytes_by_id.items():
            first_byte = token_bytes[0]
            self.tokens_by_first_byte.setdefault(first_byte, []).append(
                (token_id, token_bytes)
            )
            if is_plain_string_token(token_bytes):
                self.plain_token_ids.add(token_id)
            else:
                self.special_by_first_byte.setdefault(
                    first_byte, []
                ).append((token_id, token_bytes))

    def _encode_exact(self, data: bytes) -> list[int]:
        """Encode text only when token bytes reproduce it exactly."""
        if not data:
            return []
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return []
        token_ids = self.model.encode(text)
        if get_generated_bytes(token_ids, self.vocabulary) != data:
            return []
        return token_ids

    def _get_parser(self, data: bytes) -> _SchemaParser | None:
        """Return the schema parser for the identified function."""
        for name, raw_name in self.name_bytes.items():
            if data.startswith(raw_name + b'"'):
                return _SchemaParser(
                    data,
                    self.definitions_by_name[name],
                    self.enum_literals_by_name[name],
                )
        return None

    def _parse(self, data: bytes) -> bool:
        """Check whether bytes fit the matching function-call schema."""
        for raw_name in self.name_bytes.values():
            if raw_name.startswith(data):
                return True

        parser = self._get_parser(data)
        return parser is not None and parser.parse()

    def identify_function(
        self,
        generated_token_ids: list[int],
    ) -> str | None:
        """Return the function name once its name and JSON bridge exist."""
        current = get_generated_bytes(generated_token_ids, self.vocabulary)
        if current is None:
            return None

        for name, raw_name in self.name_bytes.items():
            if current.startswith(raw_name + b'"' + self.bridge_bytes):
                return name
        return None

    def get_forced_tokens(self, generated_token_ids: list[int]) -> list[int]:
        """Complete a unique function name and insert the JSON bridge."""
        current = get_generated_bytes(generated_token_ids, self.vocabulary)
        if current is None:
            return []

        for raw_name in self.name_bytes.values():
            name_and_quote = raw_name + b'"'
            if not current.startswith(name_and_quote):
                continue
            bridge_prefix = current[len(name_and_quote):]
            if not self.bridge_bytes.startswith(bridge_prefix):
                return []
            return self._encode_exact(
                self.bridge_bytes[len(bridge_prefix):]
            )

        candidates = [
            raw_name
            for raw_name in self.name_bytes.values()
            if raw_name.startswith(current)
        ]
        if len(candidates) != 1:
            return []

        remaining = candidates[0][len(current):] + b'"' + self.bridge_bytes
        return self._encode_exact(remaining)

    def get_allowed_tokens(self, generated_token_ids: list[int]) -> set[int]:
        """Return vocabulary tokens that preserve a valid schema prefix."""
        current = get_generated_bytes(generated_token_ids, self.vocabulary)
        if current is None:
            return set()

        parser = self._get_parser(current)
        in_string = False
        if parser is not None and parser.parse():
            if parser.complete:
                return self.end_token_ids.copy()
            in_string = parser.string_body_open

        allowed: set[int] = set()
        candidates = self.tokens_by_first_byte
        if in_string:
            allowed.update(self.plain_token_ids)
            candidates = self.special_by_first_byte

        for first_byte, tokens in candidates.items():
            if not self._parse(current + bytes([first_byte])):
                continue

            for token_id, token_bytes in tokens:
                if self._parse(current + token_bytes):
                    allowed.add(token_id)

        return allowed

    def _is_complete_call(self, data: bytes) -> bool:
        """Check whether bytes contain a complete serialized function call."""
        parser = self._get_parser(data)
        return parser is not None and parser.parse() and parser.complete
