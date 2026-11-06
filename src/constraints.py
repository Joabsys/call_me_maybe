"""Define byte-level constraints for function-call generation."""

import codecs
import json
from typing import Literal

from src.llm import LanguageModel
from src.models import FunctionDefinition, ParameterDefinition
from src.vocabulary import Vocabulary

ValueStatus = Literal["INCOMPLETE", "COMPLETE", "INVALID"]


def get_generated_bytes(
    generated_token_ids: list[int],
    vocabulary: Vocabulary,
) -> bytes | None:
    """Reconstruct the bytes represented by generated token IDs."""
    generated_bytes = bytearray()

    for token_id in generated_token_ids:
        token_bytes = vocabulary.get_token_bytes(token_id)
        if token_bytes is None:
            return None
        generated_bytes.extend(token_bytes)

    return bytes(generated_bytes)


def get_function_name_state(
    generated_bytes: bytes,
    function_names: list[str],
) -> tuple[list[str], str | None]:
    """Return matching function names and a name followed by its quote."""
    candidates: list[str] = []
    completed_name: str | None = None

    for name in function_names:
        name_bytes = name.encode("utf-8")
        if name_bytes.startswith(generated_bytes):
            candidates.append(name)
        elif generated_bytes.startswith(name_bytes):
            if generated_bytes[len(name_bytes):].startswith(b'"'):
                completed_name = name

    return candidates, completed_name


def get_allowed_token_ids(
    current_bytes: bytes,
    continuations: list[bytes],
    vocabulary: Vocabulary,
) -> set[int]:
    """Return tokens that continue at least one supplied byte sequence."""
    allowed_token_ids: set[int] = set()

    for token_id in vocabulary.get_valid_token_ids():
        token_bytes = vocabulary.get_token_bytes(token_id)
        if not token_bytes:
            continue

        candidate = current_bytes + token_bytes
        if any(item.startswith(candidate) for item in continuations):
            allowed_token_ids.add(token_id)

    return allowed_token_ids


def get_string_prefix_status(text: str) -> ValueStatus:
    """Check whether text is a valid or incomplete JSON string body."""
    index = 0
    while index < len(text):
        char = text[index]
        if ord(char) < 0x20 or char == '"':
            return "INVALID"
        if char != "\\":
            index += 1
            continue

        index += 1
        if index == len(text):
            return "INCOMPLETE"

        escape_char = text[index]
        if escape_char in '"\\/bfnrt':
            index += 1
            continue
        if escape_char != "u":
            return "INVALID"

        index += 1
        for _ in range(4):
            if index == len(text):
                return "INCOMPLETE"
            if text[index] not in "0123456789abcdefABCDEF":
                return "INVALID"
            index += 1

    return "COMPLETE"


def get_float_prefix_status(text: str) -> ValueStatus:
    """Check whether text is an incomplete, complete, or invalid decimal."""
    if not text:
        return "INCOMPLETE"

    index = 0
    if text[index] == "-":
        index += 1
        if index == len(text):
            return "INCOMPLETE"

    if index == len(text):
        return "INCOMPLETE"

    if text[index] == "0":
        index += 1
        if index < len(text) and text[index].isdigit():
            return "INVALID"
    elif "1" <= text[index] <= "9":
        while index < len(text) and "0" <= text[index] <= "9":
            index += 1
    else:
        return "INVALID"

    if index == len(text):
        return "INCOMPLETE"
    if text[index] != ".":
        return "INVALID"

    index += 1
    decimal_start = index
    while index < len(text) and "0" <= text[index] <= "9":
        index += 1

    if index != len(text):
        return "INVALID"
    if index == decimal_start:
        return "INCOMPLETE"

    return "COMPLETE"


def get_json_value_status(
    value_bytes: bytes,
    parameter: ParameterDefinition,
) -> tuple[ValueStatus, int | None]:
    """Validate a typed JSON value and return its byte length if complete."""
    if parameter.enum is not None:
        serialized_values: list[bytes] = []
        for value in parameter.enum:
            if parameter.type == "number":
                value = float(value)
            serialized_values.append(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )

        for serialized in serialized_values:
            if value_bytes.startswith(serialized):
                return "COMPLETE", len(serialized)
        if any(value.startswith(value_bytes) for value in serialized_values):
            return "INCOMPLETE", None
        return "INVALID", None

    if parameter.type == "number":
        try:
            text = value_bytes.decode("ascii")
        except UnicodeDecodeError:
            return "INVALID", None

        number_end = 0
        while number_end < len(text) and text[number_end] in "0123456789.-":
            number_end += 1

        number_text = text[:number_end]
        trailing_text = text[number_end:]
        status = get_float_prefix_status(number_text)

        if status == "INVALID":
            return "INVALID", None
        if status == "INCOMPLETE":
            return ("INVALID", None) if trailing_text else ("INCOMPLETE", None)
        return "COMPLETE", number_end

    if parameter.type == "string":
        if not value_bytes:
            return "INCOMPLETE", None
        if value_bytes[0] != ord('"'):
            return "INVALID", None

        escaped = False
        for index in range(1, len(value_bytes)):
            byte = value_bytes[index]
            if byte < 0x20:
                return "INVALID", None
            if escaped:
                escaped = False
                continue
            if byte == ord("\\"):
                escaped = True
                continue
            if byte == ord('"'):
                try:
                    json.loads(value_bytes[:index + 1])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return "INVALID", None
                return "COMPLETE", index + 1

        try:
            decoder = codecs.getincrementaldecoder("utf-8")()
            text = decoder.decode(value_bytes[1:], final=False)
        except UnicodeDecodeError:
            return "INVALID", None

        if get_string_prefix_status(text) == "INVALID":
            return "INVALID", None
        return "INCOMPLETE", None

    return "INVALID", None


class FunctionCallConstraint:
    """Constrain generation to a function call matching its definition."""

    def __init__(
        self,
        definitions: list[FunctionDefinition],
        model: LanguageModel,
        vocabulary: Vocabulary,
    ) -> None:
        """Prepare function definitions and the tokens used for generation."""
        if not definitions:
            raise ValueError("At least one function definition is required.")

        names = [definition.name for definition in definitions]
        if len(set(names)) != len(names):
            raise ValueError("Function names must be unique.")

        quote_token_id = vocabulary.get_token_id('"')
        end_token_id = vocabulary.get_token_id("<|im_end|>")
        if quote_token_id is None:
            raise ValueError("The JSON quote token is missing.")
        if end_token_id is None:
            raise ValueError("The message end token is missing.")

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

        self.function_names = names
        self.definitions_by_name = {
            definition.name: definition for definition in definitions
        }
        self.model = model
        self.vocabulary = vocabulary
        self.quote_token_id = quote_token_id
        self.end_token_ids = {end_token_id}
        self.bridge = b', "parameters": {'
        self.token_bytes_by_id = {
            token_id: token_bytes
            for token_id in vocabulary.get_valid_token_ids()
            if (token_bytes := vocabulary.get_token_bytes(token_id))
        }

    def get_forced_tokens(self, generated_token_ids: list[int]) -> list[int]:
        """Teleport a unique function name and insert the parameter bridge."""
        current_bytes = get_generated_bytes(
            generated_token_ids,
            self.vocabulary,
        )
        if current_bytes is None:
            return []

        candidates = [
            name for name in self.function_names
            if name.encode("utf-8").startswith(current_bytes)
        ]
        if len(candidates) != 1:
            return []

        remaining = candidates[0].encode("utf-8")[len(current_bytes):]
        name_token_ids = self.model.encode(remaining.decode("utf-8"))
        bridge_token_ids = self.model.encode(self.bridge.decode("utf-8"))
        forced_ids = name_token_ids + [self.quote_token_id] + bridge_token_ids

        forced_bytes = get_generated_bytes(forced_ids, self.vocabulary)
        if forced_bytes != remaining + b'"' + self.bridge:
            return []
        return forced_ids

    def _get_parameter_prefix_status(
        self,
        current_bytes: bytes,
        parameters: list[tuple[str, ParameterDefinition]],
    ) -> tuple[bool, bool]:
        """Validate parameters and report whether the structure is complete."""
        offset = 0

        for index, (parameter_name, parameter) in enumerate(parameters):
            expected_key = f'"{parameter_name}": '.encode("utf-8")
            remaining = current_bytes[offset:]

            if len(remaining) < len(expected_key):
                return expected_key.startswith(remaining), False
            if not remaining.startswith(expected_key):
                return False, False

            offset += len(expected_key)
            status, value_length = get_json_value_status(
                current_bytes[offset:],
                parameter,
            )
            if status == "INVALID":
                return False, False
            if status == "INCOMPLETE":
                return True, False
            if value_length is None:
                return False, False

            offset += value_length
            following = b", " if index + 1 < len(parameters) else b"}}"

            remaining = current_bytes[offset:]
            if len(remaining) < len(following):
                return following.startswith(remaining), False
            if not remaining.startswith(following):
                return False, False
            offset += len(following)

        is_complete = offset == len(current_bytes)
        return is_complete, is_complete

    def _get_parameter_allowed_tokens(
        self,
        current_bytes: bytes,
        parameters: list[tuple[str, ParameterDefinition]],
    ) -> set[int]:
        """Apply the Type Shield to the next complete token byte sequence."""
        allowed_token_ids: set[int] = set()
        for token_id, token_bytes in self.token_bytes_by_id.items():
            is_valid, _ = self._get_parameter_prefix_status(
                current_bytes + token_bytes,
                parameters,
            )
            if is_valid:
                allowed_token_ids.add(token_id)
        return allowed_token_ids

    def get_allowed_tokens(
        self,
        generated_token_ids: list[int],
    ) -> set[int]:
        """Return tokens that preserve a valid function-call JSON prefix."""
        current_bytes = get_generated_bytes(
            generated_token_ids,
            self.vocabulary,
        )
        if current_bytes is None:
            return set()

        candidates, completed_name = get_function_name_state(
            current_bytes,
            self.function_names,
        )
        if completed_name is None:
            continuations = [
                name.encode("utf-8") + b'"'
                for name in candidates
            ]
            return get_allowed_token_ids(
                current_bytes,
                continuations,
                self.vocabulary,
            )

        name_end = len(completed_name.encode("utf-8")) + 1
        remainder = current_bytes[name_end:]
        if len(remainder) < len(self.bridge):
            return get_allowed_token_ids(
                remainder,
                [self.bridge],
                self.vocabulary,
            )
        if not remainder.startswith(self.bridge):
            return set()

        parameter_bytes = remainder[len(self.bridge):]
        parameters = list(
            self.definitions_by_name[completed_name].parameters.items()
        )
        if not parameters:
            if b"}}".startswith(parameter_bytes):
                if parameter_bytes == b"}}":
                    return self.end_token_ids
                return get_allowed_token_ids(
                    parameter_bytes,
                    [b"}}"],
                    self.vocabulary,
                )
            return set()

        is_valid, is_complete = self._get_parameter_prefix_status(
            parameter_bytes,
            parameters,
        )
        if not is_valid:
            return set()
        if is_complete:
            return self.end_token_ids
        return self._get_parameter_allowed_tokens(
            parameter_bytes,
            parameters,
        )
