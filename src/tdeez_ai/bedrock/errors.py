"""Maps Bedrock failures to user-safe application errors.

Upstream messages can echo request content, so they are never copied into our messages.
"""

from tdeez_ai.core.errors import (
    AppError,
    BadRequest,
    ContentBlocked,
    Throttled,
    UpstreamError,
    UpstreamTimeout,
)

RETRYABLE_CODES: frozenset[str] = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
    }
)

_THROTTLE_CODES = frozenset({"ThrottlingException", "TooManyRequestsException"})
_QUOTA_CODES = frozenset({"ServiceQuotaExceededException"})
_FILTER_HINTS = ("blocked", "content filter", "content policy", "responsible ai", "moderation")

MSG_THROTTLED = "The AI service is busy right now. Please try again in a moment."
MSG_CONTENT = "That image or request could not be processed by our content safety checks."
MSG_BAD_INPUT = "The AI service could not use that image or request. Try a different image."
MSG_TIMEOUT = "The AI service took too long to respond. Please try again."
MSG_UPSTREAM = "The AI service is temporarily unavailable. Please try again later."
MSG_BAD_OUTPUT = "The AI service returned an unexpected response. Please try again."


def map_client_error(code: str, message: str) -> AppError:
    """Translate a Bedrock error code (and, for filter detection only, its message)."""
    if code in _THROTTLE_CODES or code in _QUOTA_CODES:
        return Throttled(MSG_THROTTLED)
    if code == "ValidationException":
        lowered = message.lower()
        if any(hint in lowered for hint in _FILTER_HINTS):
            return ContentBlocked(MSG_CONTENT)
        return BadRequest(MSG_BAD_INPUT)
    if code == "ModelTimeoutException":
        return UpstreamTimeout(MSG_TIMEOUT)
    return UpstreamError(MSG_UPSTREAM)
