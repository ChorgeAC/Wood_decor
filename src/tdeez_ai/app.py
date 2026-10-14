"""FastAPI application factory, envelope-shaped error handling and request logging."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tdeez_ai.api import concepts, creations, customization, profile, public, quotes, rooms, slabs
from tdeez_ai.container import Container, build_container
from tdeez_ai.core.config import Settings, get_settings
from tdeez_ai.core.errors import AppError
from tdeez_ai.core.logging import configure_logging, log_event
from tdeez_ai.models.base import ApiResponse

logger = logging.getLogger(__name__)


def _error(status: int, message: str) -> JSONResponse:
    body = ApiResponse[None](is_success=False, message=message, data=None)
    return JSONResponse(status_code=status, content=body.model_dump(by_alias=True))


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    app = FastAPI(title="TDeez AI Backend", version="0.1.0")
    app.state.container = container or build_container(settings)

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        response = await call_next(request)
        route = request.scope.get("route")
        path_template = getattr(route, "path", "unmatched")
        log_event(
            logger,
            "request",
            request_id=request_id,
            method=request.method,
            route=str(path_template),
            status=response.status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log_event(logger, "app_error", error_code=exc.code, status=exc.status_code)
        return _error(exc.status_code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field names only: error "input" values may contain customer data.
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
        return _error(422, "Invalid request. Please check: " + ", ".join(fields))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        messages = {
            404: "Not found.",
            405: "Method not allowed.",
            401: "Please sign in to continue.",
        }
        return _error(exc.status_code, messages.get(exc.status_code, "Request failed."))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log_event(logger, "unhandled_error", level=logging.ERROR, error_type=type(exc).__name__)
        return _error(500, "Something went wrong. Please try again.")

    for module in (public, profile, creations, slabs, concepts, rooms, customization, quotes):
        app.include_router(module.router)
    return app
