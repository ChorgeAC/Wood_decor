"""Thin Bedrock Runtime wrapper: timeouts, bounded retry with backoff, error mapping."""

import json
import logging
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import boto3
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from pydantic import BaseModel, ValidationError

from tdeez_ai.bedrock.errors import (
    MSG_BAD_OUTPUT,
    MSG_TIMEOUT,
    MSG_UPSTREAM,
    RETRYABLE_CODES,
    map_client_error,
)
from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import AppError, UpstreamError, UpstreamTimeout
from tdeez_ai.core.images import ImageFormat
from tdeez_ai.core.logging import log_event
from tdeez_ai.core.types import JsonObject

logger = logging.getLogger(__name__)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    fmt: ImageFormat


class BedrockGateway(Protocol):
    def invoke_model(self, model_id: str, body: JsonObject) -> JsonObject: ...

    def converse(
        self,
        model_id: str,
        *,
        system: str,
        user_text: str,
        images: Sequence[ImageInput] = (),
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


class RuntimeApi(Protocol):
    """The subset of the boto3 ``bedrock-runtime`` client that we use."""

    def invoke_model(self, **kwargs: object) -> Mapping[str, object]: ...

    def converse(self, **kwargs: object) -> Mapping[str, object]: ...


class BedrockClient:
    def __init__(
        self,
        runtime: RuntimeApi,
        *,
        max_retries: int,
        backoff_base_s: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runtime = runtime
        self._max_retries = max_retries
        self._backoff = backoff_base_s
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> "BedrockClient":
        runtime = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            config=Config(
                read_timeout=settings.bedrock_read_timeout_s,
                connect_timeout=settings.bedrock_connect_timeout_s,
                retries={"max_attempts": 1, "mode": "standard"},  # we own the retry policy
            ),
        )
        return cls(
            runtime,
            max_retries=settings.bedrock_max_retries,
            backoff_base_s=settings.bedrock_backoff_base_s,
        )

    # -- public API -----------------------------------------------------------------------

    def invoke_model(self, model_id: str, body: JsonObject) -> JsonObject:
        payload = json.dumps(body)

        def call() -> JsonObject:
            response = self._runtime.invoke_model(
                modelId=model_id,
                body=payload,
                contentType="application/json",
                accept="application/json",
            )
            stream = response.get("body")
            read = getattr(stream, "read", None)
            if not callable(read):
                raise UpstreamError(MSG_BAD_OUTPUT)
            try:
                parsed: object = json.loads(read())
            except ValueError as exc:
                raise UpstreamError(MSG_BAD_OUTPUT) from exc
            if not isinstance(parsed, dict):
                raise UpstreamError(MSG_BAD_OUTPUT)
            return parsed

        return self._with_retry("invoke_model", call)

    def converse(
        self,
        model_id: str,
        *,
        system: str,
        user_text: str,
        images: Sequence[ImageInput] = (),
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        content: list[dict[str, object]] = [
            {"image": {"format": img.fmt, "source": {"bytes": img.data}}} for img in images
        ]
        content.append({"text": user_text})

        def call() -> str:
            response = self._runtime.converse(
                modelId=model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": content}],
                inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
            )
            return _extract_text(response)

        return self._with_retry("converse", call)

    # -- internals ------------------------------------------------------------------------

    def _with_retry(self, task: str, call: Callable[[], T]) -> T:
        attempt = 0
        while True:
            try:
                return call()
            except AppError:
                raise
            except ClientError as exc:
                error = exc.response.get("Error", {})
                code = str(error.get("Code", ""))
                message = str(error.get("Message", ""))
                if code in RETRYABLE_CODES and attempt < self._max_retries:
                    self._backoff_and_log(task, code, attempt)
                    attempt += 1
                    continue
                log_event(
                    logger, "bedrock_error", level=logging.WARNING, task=task, error_code=code
                )
                raise map_client_error(code, message) from exc
            except (ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError) as exc:
                if attempt < self._max_retries:
                    self._backoff_and_log(task, "timeout", attempt)
                    attempt += 1
                    continue
                if isinstance(exc, EndpointConnectionError):
                    raise UpstreamError(MSG_UPSTREAM) from exc
                raise UpstreamTimeout(MSG_TIMEOUT) from exc

    def _backoff_and_log(self, task: str, code: str, attempt: int) -> None:
        log_event(
            logger,
            "bedrock_retry",
            level=logging.WARNING,
            task=task,
            error_code=code,
            attempt=attempt,
        )
        delay = self._backoff * (2**attempt)
        self._sleep(delay + random.uniform(0, self._backoff))


def _extract_text(response: Mapping[str, object]) -> str:
    output = response.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    blocks = message.get("content") if isinstance(message, dict) else None
    if not isinstance(blocks, list):
        raise UpstreamError(MSG_BAD_OUTPUT)
    texts = [b["text"] for b in blocks if isinstance(b, dict) and isinstance(b.get("text"), str)]
    if not texts:
        raise UpstreamError(MSG_BAD_OUTPUT)
    return "".join(texts)


def parse_model_json(text: str, model: type[M]) -> M:
    """Extract the first JSON object from model text and validate it strictly."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise UpstreamError(MSG_BAD_OUTPUT)
    try:
        return model.model_validate_json(text[start : end + 1])
    except ValidationError as exc:
        raise UpstreamError(MSG_BAD_OUTPUT) from exc
