"""Explicit application exceptions.

Each carries an HTTP status so `app.api` can translate it into a clean JSON
error without leaking a stack trace to the caller.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected, user-facing failures."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, str]:
        payload = {"error": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"


class AccountNotFoundError(AppError):
    status_code = 404
    code = "account_not_found"


class KnowledgeBaseError(AppError):
    status_code = 500
    code = "knowledge_base_error"


class DatasetError(AppError):
    status_code = 500
    code = "dataset_error"


class LLMUnavailableError(AppError):
    """No credentials, or the provider could not be reached."""

    status_code = 503
    code = "llm_unavailable"


class LLMResponseError(AppError):
    """The provider replied, but not with something we can use."""

    status_code = 502
    code = "llm_invalid_response"
