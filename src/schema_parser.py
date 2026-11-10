"""Validate generated JSON prefixes against function schemas."""

from typing import Literal

from src.models import FunctionDefinition, ParameterDefinition

ParseStatus = Literal["valid", "complete", "invalid"]


def _number_status(text: str) -> ParseStatus:
    """Classify a decimal JSON number prefix."""
    if not text:
        return "valid"

    index = 0
    if text[index] == "-":
        index += 1
        if index == len(text):
            return "valid"

    if text[index] == "0":
        index += 1
        if index < len(text) and "0" <= text[index] <= "9":
            return "invalid"
    elif "1" <= text[index] <= "9":
        while index < len(text) and "0" <= text[index] <= "9":
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
    while index < len(text) and "0" <= text[index] <= "9":
        index += 1

    if index != len(text):
        return "invalid"
    return "complete" if index > decimal_start else "valid"


def _integer_status(text: str) -> ParseStatus:
    """Classify an integer JSON number prefix."""
    if not text:
        return "valid"

    index = 0
    if text[index] == "-":
        index += 1
        if index == len(text):
            return "valid"

    if text[index] == "0":
        index += 1
        if index < len(text) and "0" <= text[index] <= "9":
            return "invalid"
    elif "1" <= text[index] <= "9":
        while index < len(text) and "0" <= text[index] <= "9":
            index += 1
    else:
        return "invalid"

    return "complete"


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

    def __init__(
        self,
        data: bytes,
        definition: FunctionDefinition,
        enum_literals: dict[str, tuple[bytes, ...]],
    ) -> None:
        self.data = data
        self.definition = definition
        self.enum_literals = enum_literals
        self.position = 0
        self.complete = False
        self.string_body_open = False

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
                if _string_status(content) != "complete":
                    return False
                self.position += 1
                return True
            self.position += 1

        content = self.data[start:self.position]
        status = _string_status(content)
        self.string_body_open = status == "complete"
        return status != "invalid"

    def _enum(self, parameter_name: str) -> bool:
        remaining = self.data[self.position:]
        for literal in self.enum_literals[parameter_name]:
            if literal.startswith(remaining):
                self.position = len(self.data)
                return True
            if remaining.startswith(literal):
                self.position += len(literal)
                return True
        return False

    def _value(
        self,
        parameter_name: str,
        parameter: ParameterDefinition,
    ) -> bool:
        if parameter.enum is not None:
            return self._enum(parameter_name)
        if parameter.type == "string":
            return self._string()

        start = self.position
        while self.position < len(self.data):
            if self.data[self.position] in b",}":
                break
            self.position += 1

        try:
            text = self.data[start:self.position].decode("ascii")
        except UnicodeDecodeError:
            return False

        status = (
            _integer_status(text)
            if parameter.type == "integer"
            else _number_status(text)
        )
        if self.position < len(self.data):
            return status == "complete"
        return status != "invalid"

    def parse(self) -> bool:
        """Return whether all supplied bytes are a valid schema prefix."""
        function_name = self.definition.name.encode("utf-8")
        if not self._literal(function_name + b'", "parameters": {'):
            return False

        if not self.definition.parameters:
            remaining_length = len(self.data) - self.position
            if not self._literal(b"}}"):
                return False
            self.complete = (
                remaining_length == 2 and self.position == len(self.data)
            )
            return self.position == len(self.data)

        for index, (parameter_name, parameter) in enumerate(
                    self.definition.parameters.items()):
            key_literal = (
                (b'"' if index == 0 else b', "')
                + parameter_name.encode("utf-8")
                + b'": '
            )
            if not self._literal(key_literal):
                return False
            if self.position == len(self.data):
                return True
            if not self._value(parameter_name, parameter):
                return False

            if self.position == len(self.data):
                return True

        remaining_length = len(self.data) - self.position
        if not self._literal(b"}}"):
            return False
        self.complete = (
            remaining_length == 2 and self.position == len(self.data)
        )
        return self.position == len(self.data)
