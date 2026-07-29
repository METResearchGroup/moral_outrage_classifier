"""Compatibility re-exports for team-llm."""

from team_llm.exceptions import (
    ExceptionCategory,
    LLMAuthError,
    LLMException,
    LLMInvalidRequestError,
    LLMPermissionDeniedError,
    LLMTransientError,
    LLMUnrecoverableError,
    standardize_litellm_exception,
)

__all__ = [
    "ExceptionCategory",
    "LLMAuthError",
    "LLMException",
    "LLMInvalidRequestError",
    "LLMPermissionDeniedError",
    "LLMTransientError",
    "LLMUnrecoverableError",
    "standardize_litellm_exception",
]
