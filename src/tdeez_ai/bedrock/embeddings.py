"""Amazon Titan Multimodal Embeddings G1 (``inputImage`` -> ``embedding``)."""

import base64
from typing import Protocol

from tdeez_ai.bedrock.client import BedrockGateway
from tdeez_ai.bedrock.errors import MSG_BAD_OUTPUT
from tdeez_ai.bedrock.guard import CostGuard
from tdeez_ai.core.errors import UpstreamError
from tdeez_ai.core.types import JsonObject


class ImageEmbedder(Protocol):
    def embed_image(self, image: bytes) -> list[float]: ...


class TitanEmbedder:
    def __init__(
        self, client: BedrockGateway, model_id: str, dimensions: int, guard: CostGuard
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._dimensions = dimensions
        self._guard = guard

    def embed_image(self, image: bytes) -> list[float]:
        self._guard.check_image(image)
        body: JsonObject = {
            "inputImage": base64.b64encode(image).decode("ascii"),
            "embeddingConfig": {"outputEmbeddingLength": self._dimensions},
        }
        response = self._client.invoke_model(self._model_id, body)
        raw = response.get("embedding")
        if (
            not isinstance(raw, list)
            or len(raw) != self._dimensions
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in raw)
        ):
            raise UpstreamError(MSG_BAD_OUTPUT)
        return [float(v) for v in raw if isinstance(v, int | float)]
