"""Amazon Nova Canvas image generation / editing (payloads verified against the Bedrock docs)."""

import base64
from typing import Protocol

from tdeez_ai.bedrock.client import BedrockGateway
from tdeez_ai.bedrock.errors import MSG_BAD_OUTPUT, MSG_CONTENT
from tdeez_ai.bedrock.guard import CostGuard
from tdeez_ai.bedrock.prompts import clip
from tdeez_ai.core.errors import ContentBlocked, UpstreamError
from tdeez_ai.core.types import JsonObject, JsonValue


class ImageGenerator(Protocol):
    def variation(
        self,
        source: bytes,
        prompt: str,
        negative: str,
        *,
        similarity: float,
        seed: int,
    ) -> bytes:
        """IMAGE_VARIATION conditioned on ``source``; returns one PNG/JPEG image."""
        ...

    def inpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        """INPAINTING of the region described by ``mask_prompt``; returns one image."""
        ...

    def outpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        """OUTPAINTING around the region described by ``mask_prompt``; returns one image."""
        ...

    def text_to_image(self, prompt: str, negative: str, *, seed: int) -> bytes: ...

    def remove_background(self, image: bytes) -> bytes:
        """BACKGROUND_REMOVAL; returns a transparent PNG."""
        ...


# Canvas seeds are bounded (0 to 858,993,459); confirm against current docs when changing models.
_MAX_SEED = 858_993_460


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class NovaCanvasGenerator:
    def __init__(self, client: BedrockGateway, model_id: str, guard: CostGuard) -> None:
        self._client = client
        self._model_id = model_id
        self._guard = guard

    def variation(
        self, source: bytes, prompt: str, negative: str, *, similarity: float, seed: int
    ) -> bytes:
        self._guard.check_image(source)
        body: JsonObject = {
            "taskType": "IMAGE_VARIATION",
            "imageVariationParams": {
                "images": [_b64(source)],
                "similarityStrength": max(0.2, min(1.0, similarity)),
                "text": clip(prompt),
                "negativeText": clip(negative),
            },
            "imageGenerationConfig": self._config(seed),
        }
        return self._one_image(body)

    def inpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        self._guard.check_image(image)
        body: JsonObject = {
            "taskType": "INPAINTING",
            "inPaintingParams": {
                "image": _b64(image),
                "maskPrompt": clip(mask_prompt),
                "text": clip(prompt),
                "negativeText": clip(negative),
            },
            # Docs: width/height must not be sent for INPAINTING.
            "imageGenerationConfig": self._config(seed),
        }
        return self._one_image(body)

    def outpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        self._guard.check_image(image)
        body: JsonObject = {
            "taskType": "OUTPAINTING",
            "outPaintingParams": {
                "image": _b64(image),
                "maskPrompt": clip(mask_prompt),
                "outPaintingMode": "DEFAULT",
                "text": clip(prompt),
                "negativeText": clip(negative),
            },
            "imageGenerationConfig": self._config(seed),
        }
        return self._one_image(body)

    def text_to_image(self, prompt: str, negative: str, *, seed: int) -> bytes:
        body: JsonObject = {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {"text": clip(prompt), "negativeText": clip(negative)},
            "imageGenerationConfig": self._config(seed),
        }
        return self._one_image(body)

    def remove_background(self, image: bytes) -> bytes:
        self._guard.check_image(image)
        body: JsonObject = {
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {"image": _b64(image)},
        }
        return self._one_image(body)

    # -- internals ------------------------------------------------------------------------

    def _config(self, seed: int) -> JsonObject:
        """Common config. No width/height is sent: Canvas defaults to 1024x1024, and the docs
        forbid sizes for INPAINTING/OUTPAINTING."""
        self._guard.check_count(1)
        return {
            "numberOfImages": 1,
            "quality": "standard",
            "cfgScale": 7.0,
            "seed": seed % _MAX_SEED,
        }

    def _one_image(self, body: JsonObject) -> bytes:
        response = self._client.invoke_model(self._model_id, body)
        images: JsonValue = response.get("images")
        if isinstance(images, list) and images and isinstance(images[0], str):
            try:
                return base64.b64decode(images[0], validate=True)
            except ValueError as exc:
                raise UpstreamError(MSG_BAD_OUTPUT) from exc
        # No image came back: Canvas reports RAI moderation through the `error` field.
        if response.get("error"):
            raise ContentBlocked(MSG_CONTENT)
        raise UpstreamError(MSG_BAD_OUTPUT)
