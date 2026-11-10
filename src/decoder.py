
"""Provide constrained token selection and generation."""

from collections.abc import Callable
from math import isfinite

LogitsProvider = Callable[[list[int]], list[float]]
AllowedTokensProvider = Callable[[list[int]], set[int]]
ForcedTokensProvider = Callable[[list[int]], list[int]]
StopCondition = Callable[[list[int]], bool]


def mask_invalid_logits(
    logits: list[float],
    allowed_token_ids: set[int],
) -> list[float]:
    """Mask disallowed tokens by assigning negative infinity to logits."""
    if not logits:
        raise ValueError("Logits cannot be empty.")
    if not allowed_token_ids:
        raise ValueError("At least one token must be allowed.")
    if any(
        token_id < 0 or token_id >= len(logits)
        for token_id in allowed_token_ids
    ):
        raise ValueError("An allowed token ID is outside the logits range.")

    masked_logits = [
        score if token_id in allowed_token_ids else float("-inf")
        for token_id, score in enumerate(logits)
    ]

    if not any(
        isfinite(masked_logits[token_id])
        for token_id in allowed_token_ids
    ):
        raise ValueError("No allowed token has a finite logit.")

    return masked_logits


def generate_tokens(
    prompt_token_ids: list[int],
    get_logits: LogitsProvider,
    get_allowed_tokens: AllowedTokensProvider,
    stop_token_ids: set[int],
    max_new_tokens: int = 256,
    get_forced_tokens: ForcedTokensProvider | None = None,
    initial_token_ids: list[int] | None = None,
    should_stop: StopCondition | None = None
) -> list[int]:
    """Generate tokens while restricting or automatically inserting IDs."""
    if max_new_tokens < 1:
        raise ValueError("The maximum number of new tokens must be positive.")

    generated_token_ids: list[int] = list(initial_token_ids or [])

    while len(generated_token_ids) < max_new_tokens:
        if should_stop is not None and should_stop(generated_token_ids):
            return generated_token_ids
        if get_forced_tokens is not None:
            forced_token_ids = get_forced_tokens(generated_token_ids)

            if forced_token_ids:
                if len(generated_token_ids) + len(
                    forced_token_ids
                ) > max_new_tokens:
                    raise ValueError(
                        "Automatic insertion exceeds the maximum token limit."
                    )

                generated_token_ids.extend(forced_token_ids)
                continue

        context = prompt_token_ids + generated_token_ids
        logits = get_logits(context)
        allowed_ids = get_allowed_tokens(generated_token_ids)
        masked_logits = mask_invalid_logits(logits, allowed_ids)

        next_token_id = max(
            sorted(allowed_ids),
            key=lambda token_id: masked_logits[token_id],
        )

        if next_token_id in stop_token_ids:
            return generated_token_ids

        generated_token_ids.append(next_token_id)

    raise ValueError("Generation reached the maximum token limit.")
