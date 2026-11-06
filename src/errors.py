
"""Define application-specific exceptions."""


class FileReadError(Exception):
    """Raise when an input file cannot be read."""


class InvalidJSONError(Exception):
    """Raise when a file does not contain valid JSON."""


class DataValidationError(Exception):
    """Raise when JSON data does not match the expected structure."""


class FileWriteError(Exception):
    """Raise when an output file cannot be written."""


class VocabularyError(Exception):
    """Raise when the tokenizer vocabulary cannot be loaded."""
