"""Constrain function-call generation to the declared JSON schema."""

import json
from typing import Literal

from src.llm import LanguageModel
from src.models import FunctionDefinition, ParameterDefinition
from src.vocabulary import Vocabulary

ParseStatus = Literal["valid", "complete", "invalid"]


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


def _number_status(text: str) -> ParseStatus:
    """Classify a decimal JSON number prefix."""
    if not text:
        return "valid"
    index = 0
    if text[index] == "-":
        index += 1
        if index == len(text):
            return "valid"
    if index == len(text):
        return "valid"
    if text[index] == "0":
        index += 1
        if index < len(text) and text[index].isdigit():
            return "invalid"
    elif "1" <= text[index] <= "9":
        while index < len(text) and text[index].isdigit():
            index += 1
    else:
        return "invalid"
    if index == len(text):
        return "valid"
    if text[index] != ".":
        return "invalid"
    index += 1
    if index == len(text):
        return "valid"
    decimal_start = index
    while index < len(text) and text[index].isdigit():
        index += 1
    if index != len(text):
        return "invalid"
    return "complete" if index > decimal_start else "valid"


def _string_status(data: bytes) -> ParseStatus:
    """Classify bytes inside a JSON string, excluding its quotes."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason == "unexpected end of data" and error.end == len(data):
            text = data[:error.start].decode("utf-8")
            incomplete_utf8 = True
        else:
            return "invalid"
    else:
        incomplete_utf8 = False

    index = 0
    while index < len(text):
        char = text[index]
        if ord(char) < 0x20:
            return "invalid"
        if char != "\\":
            index += 1
            continue
        index += 1
        if index == len(text):
            return "valid"
        escape = text[index]
        if escape in '"\\/bfnrt':
            index += 1
            continue
        if escape != "u":
            return "invalid"
        index += 1
        for _ in range(4):
            if index == len(text):
                return "valid"
            if text[index] not in "0123456789abcdefABCDEF":
                return "invalid"
            index += 1
    return "valid" if incomplete_utf8 else "complete"


class _SchemaParser:
    """Validate a byte prefix against one function-call schema."""

    def __init__(self, data: bytes, definition: FunctionDefinition) -> None:
        self.data = data
        self.definition = definition
        self.position = 0

    def _literal(self, literal: bytes) -> bool:
        remaining = self.data[self.position:]
        common = min(len(remaining), len(literal))
        if remaining[:common] != literal[:common]:
            return False
        if len(remaining) < len(literal):
            self.position = len(self.data)
            return True
        self.position += len(literal)
        return True

    def _string(self) -> bool:
        if self.position >= len(self.data):
            return True
        if self.data[self.position:self.position + 1] != b'"':
            return False
        self.position += 1
        start = self.position
        escaped = False
        while self.position < len(self.data):
            byte = self.data[self.position]
            if escaped:
                escaped = False
                self.position += 1
                continue
            if byte == 0x5C:
                escaped = True
                self.position += 1
                continue
            if byte == 0x22:
                content = self.data[start:self.position]
                if _string_status(content) == "invalid":
                    return False
                self.position += 1
                return True
            self.position += 1
        content = self.data[start:self.position]
        return _string_status(content) != "invalid"

    def _value(self, parameter: ParameterDefinition) -> bool:
        if parameter.enum is not None:
            literals = {
                json.dumps(
                    float(value) if parameter.type == "number" else value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                for value in parameter.enum
            }
            remaining = self.data[self.position:]
            for literal in literals:
                if literal.startswith(remaining):
                    self.position = len(self.data)
                    return True
                if remaining.startswith(literal):
                    self.position += len(literal)
                    return True
            return False

        if parameter.type == "string":
            return self._string()

        start = self.position
        while self.position < len(self.data):
            byte = self.data[self.position]
            if byte in b",}":
                break
            self.position += 1
        raw = self.data[start:self.position]
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            return False
        status = _number_status(text)
        if self.position < len(self.data):
            return status == "complete"
        return status != "invalid"

    def parse(self) -> bool:
        """Return whether all supplied bytes are a valid schema prefix."""
        name = self.definition.name.encode("utf-8")
        if not self._literal(name + b'", "parameters": {'):
            return False
        parameters = list(self.definition.parameters.items())
        if not parameters:
            return self._literal(b"}}") and self.position == len(self.data)

        for index, (name, parameter) in enumerate(parameters):
            key_literal = (
                (b'"' if index == 0 else b', "')
                + name.encode("utf-8")
                + b'": '
            )
            if not self._literal(key_literal):
                return False
            if self.position == len(self.data):
                return True
            if not self._value(parameter):
                return False
            if self.position == len(self.data):
                return True

        if not self._literal(b"}}"):
            return False
        return self.position == len(self.data)


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
        self.definitions_by_name = {item.name: item for item in definitions}
        self.model = model
        self.vocabulary = vocabulary
        end_token_id = vocabulary.get_token_id("<|im_end|>")
        quote_token_id = vocabulary.get_token_id('"')
        if end_token_id is None or quote_token_id is None:
            raise ValueError(
                "Required JSON or message-ending token is missing."
            )
        self.end_token_ids = {end_token_id}
        self.quote_token_id = quote_token_id
        self.name_bytes = {name: name.encode("utf-8") for name in names}
        self.valid_token_ids = vocabulary.get_valid_token_ids()

    def _parse(self, data: bytes) -> bool:
        """Check whether bytes fit at least one function schema."""
        for definition in self.definitions_by_name.values():
            name = definition.name.encode("utf-8")
            if name.startswith(data):
                return True
            if data.startswith(name) and _SchemaParser(
                data,
                definition,
            ).parse():
                return True
        return any(
            _SchemaParser(data, definition).parse()
            for definition in self.definitions_by_name.values()
        )

    def get_forced_tokens(self, generated_token_ids: list[int]) -> list[int]:
        """Complete a unique function name and insert the JSON bridge."""
        current = get_generated_bytes(generated_token_ids, self.vocabulary)
        if current is None:
            return []
        candidates = [
            name
            for name, raw in self.name_bytes.items()
            if raw.startswith(current)
        ]
        if len(candidates) != 1:
            return []
        name = candidates[0]
        remaining = self.name_bytes[name][len(current):]
        encoded = self.model.encode(remaining.decode("utf-8"))
        bridge = self.model.encode(', "parameters": {')
        forced = encoded + [self.quote_token_id] + bridge
        forced_bytes = get_generated_bytes(forced, self.vocabulary)
        if forced_bytes != remaining + b'", "parameters": {':
            return []
        return forced

    def get_allowed_tokens(self, generated_token_ids: list[int]) -> set[int]:
        """Return vocabulary tokens that preserve a valid schema prefix."""
        current = get_generated_bytes(generated_token_ids, self.vocabulary)
        if current is None:
            return set()
        if self._is_complete_call(current):
            return self.end_token_ids.copy()
        allowed: set[int] = set()
        for token_id in self.valid_token_ids:
            token_bytes = self.vocabulary.get_token_bytes(token_id)
            if not token_bytes:
                continue
            if self._parse(current + token_bytes):
                allowed.add(token_id)
        if self._is_complete_call(current):
            allowed.update(self.end_token_ids)
        return allowed

    def _is_complete_call(self, data: bytes) -> bool:
        """Check whether bytes contain a complete serialized function call."""
        return any(
            _SchemaParser(data, definition).parse()
            and data.endswith(b"}}")
            for definition in self.definitions_by_name.values()
        )
