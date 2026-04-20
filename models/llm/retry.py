"""Retry logic for LLM completions with validation."""

import logging
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

try:
    import litellm.exceptions as litellm_exceptions  # type: ignore[import-untyped]
except ImportError:
    litellm_exceptions = None  # type: ignore[assignment]

from models.llm.exceptions import (
    ExceptionCategory,
    LLMException,
    LLMTransientError,
)

P = ParamSpec("P")
R = TypeVar("R")
logger = logging.getLogger(__name__)

ANTHROPIC_MODEL_PREFIX = "anthropic/"
ANTHROPIC_RATE_LIMIT_SLEEP_SECONDS = 60.0

# Categories that should NOT be retried (fail hard immediately)
NON_RETRYABLE_CATEGORIES = {
    ExceptionCategory.AUTH_ERROR,
    ExceptionCategory.INVALID_REQUEST,
    ExceptionCategory.UNRECOVERABLE,
}


def _should_retry(exception: BaseException) -> bool:
    """Determine if an exception should be retried.

    Returns False for non-retryable exceptions (they fail hard immediately).
    Returns True for all other exceptions (they will be retried).

    Args:
        exception: The exception that was raised

    Returns:
        True if the exception should be retried, False otherwise
    """
    # Check if this is an internal LLM exception with a category
    if isinstance(exception, LLMException):
        return exception.category not in NON_RETRYABLE_CATEGORIES

    # For non-LLM exceptions (ValueError, ValidationError, etc.), retry by default
    # This maintains backward compatibility and allows retries on validation errors
    return True


def _model_from_retry_state(retry_state: RetryCallState) -> str | None:
    """Resolve `model` from LLMService structured-completion call signatures."""
    args = retry_state.args
    if len(args) >= 3:
        candidate = args[2]
        return candidate if isinstance(candidate, str) else None
    return retry_state.kwargs.get("model") if isinstance(retry_state.kwargs, dict) else None


def _is_anthropic_litellm_rate_limit(
    exc: BaseException | None, model: str | None
) -> bool:
    if model is None or not model.startswith(ANTHROPIC_MODEL_PREFIX):
        return False
    if not isinstance(exc, LLMTransientError) or exc.original_exception is None:
        return False
    if litellm_exceptions is None:
        return False
    return isinstance(
        exc.original_exception,
        litellm_exceptions.RateLimitError,  # type: ignore[attr-defined]
    )


def _wait_anthropic_rate_limit_else_exponential(
    *,
    initial_delay: float,
    max_delay: float,
) -> Callable[[RetryCallState], float]:
    jitter = wait_exponential_jitter(initial=initial_delay, max=max_delay)

    def wait(retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        exc: BaseException | None = None
        if outcome is not None and outcome.failed:
            exc = outcome.exception()
        model = _model_from_retry_state(retry_state)
        if _is_anthropic_litellm_rate_limit(exc, model):
            return ANTHROPIC_RATE_LIMIT_SLEEP_SECONDS
        return jitter(retry_state)

    return wait


def retry_llm_completion(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator for retrying LLM completions with exponential backoff.

    Retries on ANY exception EXCEPT non-retryable ones (authentication errors,
    invalid requests, etc.). This allows the model to "try again" if it returns
    invalid output, encounters transient HTTP errors, or has missing content.

    For LiteLLM ``RateLimitError`` on models whose id starts with ``anthropic/``,
    the wait before the next attempt is fixed at 60 seconds so the org TPM/RPM
    window can clear; all other failures use exponential jitter as before.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_delay: Initial delay in seconds before first retry (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 60.0)

    Returns:
        Decorated function with retry logic

    Example:
        @retry_llm_completion(max_retries=3)
        def _complete_and_validate_structured(self, ...):
            response = self._chat_completion(...)
            return self.handle_completion_response(response, response_model)
    """
    return retry(
        stop=stop_after_attempt(max_retries + 1),  # +1 for initial attempt
        wait=_wait_anthropic_rate_limit_else_exponential(
            initial_delay=initial_delay,
            max_delay=max_delay,
        ),
        retry=retry_if_exception(_should_retry),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Re-raise the exception after all retries exhausted
    )
