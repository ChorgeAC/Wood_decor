"""Bedrock wrapper: retries, timeouts, error mapping, payload shapes, cost guardrails."""

import base64
import io
import json
from typing import Any

import boto3
import pytest
from botocore.exceptions import ConnectTimeoutError, ReadTimeoutError
from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from tdeez_ai.app import create_app
from tdeez_ai.bedrock.canvas import NovaCanvasGenerator
from tdeez_ai.bedrock.client import BedrockClient, ImageInput, parse_model_json
from tdeez_ai.bedrock.embeddings import TitanEmbedder
from tdeez_ai.bedrock.errors import map_client_error
from tdeez_ai.bedrock.guard import CostGuard
from tdeez_ai.bedrock.vision import NovaVisionAnalyzer
from tdeez_ai.container import build_container
from tdeez_ai.core.errors import (
    BadRequest,
    ContentBlocked,
    PayloadTooLarge,
    Throttled,
    UpstreamError,
    UpstreamTimeout,
)
from tdeez_ai.core.keys import user_prefix
from tdeez_ai.core.types import JsonObject, JsonValue
from tdeez_ai.models.slab import SlabVisionResult
from tests.conftest import RecordingImages, make_settings

ALICE = user_prefix("mock-alice")
BOB = user_prefix("mock-bob")

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


def _stubbed() -> tuple[Any, Stubber]:
    runtime = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return runtime, Stubber(runtime)


def _body(payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload).encode()
    return {"body": StreamingBody(io.BytesIO(raw), len(raw)), "contentType": "application/json"}


def _client(runtime: Any, sleeps: list[float], retries: int = 2) -> BedrockClient:
    return BedrockClient(runtime, max_retries=retries, backoff_base_s=0.1, sleep=sleeps.append)


# -- error mapping -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "message", "expected"),
    [
        ("ThrottlingException", "Too many requests", Throttled),
        ("TooManyRequestsException", "x", Throttled),
        ("ServiceQuotaExceededException", "x", Throttled),
        ("ValidationException", "Your request was blocked by content filters", ContentBlocked),
        ("ValidationException", "Responsible AI policy violation", ContentBlocked),
        ("ValidationException", "image too small", BadRequest),
        ("ModelTimeoutException", "x", UpstreamTimeout),
        ("AccessDeniedException", "x", UpstreamError),
        ("SomethingNew", "x", UpstreamError),
    ],
)
def test_error_mapping(code: str, message: str, expected: type[Exception]) -> None:
    mapped = map_client_error(code, message)
    assert type(mapped) is expected


def test_mapped_messages_never_echo_upstream_text() -> None:
    mapped = map_client_error("ValidationException", "blocked: prompt contained alice@example.com")
    assert "alice@example.com" not in mapped.message
    assert "blocked" not in mapped.message.lower()


# -- retry / timeout -----------------------------------------------------------------------------


def test_throttling_retries_with_backoff_then_succeeds() -> None:
    runtime, stub = _stubbed()
    stub.add_client_error(
        "invoke_model", service_error_code="ThrottlingException", http_status_code=429
    )
    stub.add_client_error(
        "invoke_model", service_error_code="ThrottlingException", http_status_code=429
    )
    stub.add_response("invoke_model", _body({"ok": True}))
    sleeps: list[float] = []
    with stub:
        assert _client(runtime, sleeps).invoke_model("m", {"a": 1}) == {"ok": True}
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0] - 0.1  # exponential growth (plus jitter)


def test_throttling_exhausts_retries_then_maps() -> None:
    runtime, stub = _stubbed()
    for _ in range(3):  # initial attempt + 2 retries
        stub.add_client_error(
            "invoke_model", service_error_code="ThrottlingException", http_status_code=429
        )
    sleeps: list[float] = []
    with stub, pytest.raises(Throttled):
        _client(runtime, sleeps).invoke_model("m", {})
    assert len(sleeps) == 2
    stub.assert_no_pending_responses()


def test_validation_error_is_not_retried() -> None:
    runtime, stub = _stubbed()
    stub.add_client_error(
        "invoke_model", service_error_code="ValidationException", service_message="bad"
    )
    sleeps: list[float] = []
    with stub, pytest.raises(BadRequest):
        _client(runtime, sleeps).invoke_model("m", {})
    assert sleeps == []


def test_content_filter_is_mapped() -> None:
    runtime, stub = _stubbed()
    stub.add_client_error(
        "invoke_model",
        service_error_code="ValidationException",
        service_message="Request blocked by content filters",
    )
    with stub, pytest.raises(ContentBlocked):
        _client(runtime, []).invoke_model("m", {})


class _TimingOutRuntime:
    def __init__(self, exc: Exception) -> None:
        self.calls = 0
        self._exc = exc

    def invoke_model(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        raise self._exc

    def converse(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        raise self._exc


@pytest.mark.parametrize(
    "exc", [ReadTimeoutError(endpoint_url="x"), ConnectTimeoutError(endpoint_url="x")]
)
def test_network_timeouts_retry_then_map_to_timeout(exc: Exception) -> None:
    runtime = _TimingOutRuntime(exc)
    sleeps: list[float] = []
    with pytest.raises(UpstreamTimeout):
        _client(runtime, sleeps, retries=2).invoke_model("m", {})
    assert runtime.calls == 3  # bounded: 1 + max_retries
    assert len(sleeps) == 2


def test_zero_retries_means_single_attempt() -> None:
    runtime = _TimingOutRuntime(ReadTimeoutError(endpoint_url="x"))
    with pytest.raises(UpstreamTimeout):
        _client(runtime, [], retries=0).converse("m", system="s", user_text="u")
    assert runtime.calls == 1


# -- converse ---------------------------------------------------------------------------------------


def test_converse_sends_images_and_returns_text() -> None:
    runtime, stub = _stubbed()
    stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "metrics": {"latencyMs": 1},
        },
        expected_params={
            "modelId": "m",
            "system": [{"text": "sys"}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": {"format": "png", "source": {"bytes": PNG}}},
                        {"text": "look"},
                    ],
                }
            ],
            "inferenceConfig": {"maxTokens": 10, "temperature": 0.0},
        },
    )
    with stub:
        text = _client(runtime, []).converse(
            "m", system="sys", user_text="look", images=[ImageInput(PNG, "png")], max_tokens=10
        )
    assert text == "hello"


def test_parse_model_json_strict() -> None:
    good = (
        '```json\n{"shape_description":"s","length_to_width_ratio":2,"defects":[],'
        '"species":"cherry","species_confidence":0.7,"summary":"x"}\n```'
    )
    assert parse_model_json(good, SlabVisionResult).species == "cherry"
    for bad in (
        "no json at all",
        '{"shape_description": "s"}',  # missing required fields
        '{"shape_description":"s","species_confidence":1.7,"summary":"x"}',  # out of range
    ):
        with pytest.raises(UpstreamError):
            parse_model_json(bad, SlabVisionResult)


class _FakeGateway:
    def __init__(self, response: JsonObject | None = None, text: str = "") -> None:
        self.response = response or {}
        self.text = text
        self.last_body: JsonObject = {}
        self.last_model = ""

    def invoke_model(self, model_id: str, body: JsonObject) -> JsonObject:
        self.last_model, self.last_body = model_id, body
        return self.response

    def converse(self, model_id: str, **kwargs: object) -> str:
        return self.text


GUARD = CostGuard(max_images=3, max_image_bytes=1000)


# -- Nova Canvas payloads ------------------------------------------------------------------------


def test_canvas_image_variation_payload_and_decode() -> None:
    out = base64.b64encode(PNG).decode()
    gw = _FakeGateway({"images": [out]})
    canvas = NovaCanvasGenerator(gw, "amazon.nova-canvas-v1:0", GUARD)
    assert canvas.variation(PNG, "p", "n", similarity=0.85, seed=5) == PNG
    body = gw.last_body
    assert body["taskType"] == "IMAGE_VARIATION"
    params: Any = body["imageVariationParams"]
    assert params["images"] == [base64.b64encode(PNG).decode()]
    assert params["similarityStrength"] == 0.85
    assert params["text"] == "p" and params["negativeText"] == "n"
    cfg: Any = body["imageGenerationConfig"]
    assert cfg["numberOfImages"] == 1


def test_canvas_other_task_types() -> None:
    out = base64.b64encode(PNG).decode()
    gw = _FakeGateway({"images": [out]})
    canvas = NovaCanvasGenerator(gw, "m", GUARD)
    canvas.remove_background(PNG)
    assert gw.last_body == {
        "taskType": "BACKGROUND_REMOVAL",
        "backgroundRemovalParams": {"image": base64.b64encode(PNG).decode()},
    }
    canvas.inpaint(PNG, "floor", "table", "people", seed=1)
    assert gw.last_body["taskType"] == "INPAINTING"
    cfg: Any = gw.last_body["imageGenerationConfig"]
    assert "width" not in cfg and "height" not in cfg  # not allowed for inpainting
    canvas.outpaint(PNG, "bg", "room", "people", seed=1)
    assert gw.last_body["taskType"] == "OUTPAINTING"
    canvas.text_to_image("a table", "people", seed=1)
    assert gw.last_body["taskType"] == "TEXT_IMAGE"


def test_canvas_prompts_are_clipped_to_limit() -> None:
    gw = _FakeGateway({"images": [base64.b64encode(PNG).decode()]})
    NovaCanvasGenerator(gw, "m", GUARD).variation(PNG, "x" * 5000, "y", similarity=0.5, seed=1)
    params: Any = gw.last_body["imageVariationParams"]
    assert len(params["text"]) <= 1024


def test_canvas_moderated_response_maps_to_content_blocked() -> None:
    gw = _FakeGateway({"images": [], "error": "blocked by RAI policy"})
    with pytest.raises(ContentBlocked):
        NovaCanvasGenerator(gw, "m", GUARD).variation(PNG, "p", "n", similarity=0.5, seed=1)


def test_canvas_empty_response_is_upstream_error() -> None:
    with pytest.raises(UpstreamError):
        NovaCanvasGenerator(_FakeGateway({}), "m", GUARD).remove_background(PNG)


def test_canvas_garbage_image_is_upstream_error() -> None:
    gw = _FakeGateway({"images": ["!!not base64!!"]})
    with pytest.raises(UpstreamError):
        NovaCanvasGenerator(gw, "m", GUARD).remove_background(PNG)


# -- cost guardrails -------------------------------------------------------------------------------


def test_cost_guard_blocks_oversize_input_before_any_call() -> None:
    gw = _FakeGateway({"images": [base64.b64encode(PNG).decode()]})
    canvas = NovaCanvasGenerator(gw, "m", GUARD)
    with pytest.raises(PayloadTooLarge):
        canvas.remove_background(b"\x89PNG\r\n\x1a\n" + b"0" * 2000)
    assert gw.last_body == {}  # never reached Bedrock


def test_cost_guard_image_count() -> None:
    guard = CostGuard(max_images=3, max_image_bytes=10)
    guard.check_count(3)
    for bad in (0, 4):
        with pytest.raises(BadRequest):
            guard.check_count(bad)


# -- Titan + vision ----------------------------------------------------------------------------------


def test_titan_request_and_validation() -> None:
    vec: list[JsonValue] = [0.1] * 256
    gw = _FakeGateway({"embedding": vec})
    titan = TitanEmbedder(gw, "amazon.titan-embed-image-v1", 256, GUARD)
    assert titan.embed_image(PNG) == vec
    assert gw.last_model == "amazon.titan-embed-image-v1"
    assert gw.last_body == {
        "inputImage": base64.b64encode(PNG).decode(),
        "embeddingConfig": {"outputEmbeddingLength": 256},
    }
    bad_responses: list[JsonObject] = [{}, {"embedding": [0.1] * 10}, {"embedding": ["x"] * 256}]
    for bad in bad_responses:
        with pytest.raises(UpstreamError):
            TitanEmbedder(_FakeGateway(bad), "m", 256, GUARD).embed_image(PNG)


def test_vision_rejects_malformed_model_output() -> None:
    vision = NovaVisionAnalyzer(_FakeGateway(text="I think it's walnut!"), "m", GUARD)
    with pytest.raises(UpstreamError):
        vision.analyze_slab(PNG, "png")


def test_vision_parses_valid_output() -> None:
    text = json.dumps(
        {
            "shape_description": "elongated",
            "length_to_width_ratio": 2.5,
            "defects": [{"kind": "knot", "severity": "low", "location_hint": "left"}],
            "species": "white_oak",
            "species_confidence": 0.8,
            "summary": "ok",
        }
    )
    result = NovaVisionAnalyzer(_FakeGateway(text=text), "m", GUARD).analyze_slab(PNG, "png")
    assert result.species == "white_oak"
    assert result.defects[0].kind == "knot"


def test_vision_rejects_unknown_defect_kind() -> None:
    text = json.dumps(
        {
            "shape_description": "x",
            "defects": [{"kind": "rot", "severity": "low"}],
            "species_confidence": 0.5,
            "summary": "x",
        }
    )
    with pytest.raises(UpstreamError):
        NovaVisionAnalyzer(_FakeGateway(text=text), "m", GUARD).analyze_slab(PNG, "png")


# -- end to end: a Bedrock failure surfaces as a user-safe envelope --------------------------------


def test_api_maps_bedrock_failure_to_safe_envelope() -> None:
    class Throttling(RecordingImages):
        def remove_background(self, image: bytes) -> bytes:
            raise Throttled("The AI service is busy right now. Please try again in a moment.")

    container = build_container(make_settings(), images=Throttling())
    client = TestClient(create_app(container.settings, container))
    key = ALICE + "slabs/a.png"
    container.store.put_bytes(key, PNG, "image/png")
    r = client.post(
        "/slabs/analyze", headers={"Authorization": "Bearer mock-alice"}, json={"imageKey": key}
    )
    assert r.status_code == 429
    assert r.json() == {
        "isSuccess": False,
        "message": "The AI service is busy right now. Please try again in a moment.",
        "data": None,
    }
