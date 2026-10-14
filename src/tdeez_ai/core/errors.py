"""Application errors. Every message is user-safe by construction (no PII, no upstream text)."""


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequest(AppError):
    status_code = 400
    code = "bad_request"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class PayloadTooLarge(AppError):
    status_code = 413
    code = "payload_too_large"


class ValidationFailure(AppError):
    status_code = 422
    code = "validation_error"


class ContentBlocked(AppError):
    status_code = 422
    code = "content_blocked"


class Throttled(AppError):
    status_code = 429
    code = "throttled"


class UpstreamError(AppError):
    status_code = 502
    code = "upstream_error"


class UpstreamTimeout(AppError):
    status_code = 504
    code = "upstream_timeout"
