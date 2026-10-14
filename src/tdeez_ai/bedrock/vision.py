"""Amazon Nova multimodal understanding via the Converse API (strict JSON, validated)."""

from typing import Protocol

from pydantic import BaseModel, Field

from tdeez_ai.bedrock import prompts
from tdeez_ai.bedrock.client import BedrockGateway, ImageInput, parse_model_json
from tdeez_ai.bedrock.guard import CostGuard
from tdeez_ai.core.images import ImageFormat
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.room import RoomSuitability
from tdeez_ai.models.slab import SlabVisionResult


class RawLegChoice(BaseModel):
    leg_style: str
    rationale: str = Field(max_length=500)


class VisionAnalyzer(Protocol):
    def analyze_slab(self, image: bytes, fmt: ImageFormat) -> SlabVisionResult: ...

    def assess_room(self, image: bytes, fmt: ImageFormat) -> RoomSuitability: ...

    def recommend_legs(
        self, furniture: FurnitureType, ratio: float | None, summary: str | None
    ) -> RawLegChoice: ...


class NovaVisionAnalyzer:
    def __init__(self, client: BedrockGateway, model_id: str, guard: CostGuard) -> None:
        self._client = client
        self._model_id = model_id
        self._guard = guard

    def analyze_slab(self, image: bytes, fmt: ImageFormat) -> SlabVisionResult:
        self._guard.check_image(image)
        text = self._client.converse(
            self._model_id,
            system=prompts.VISION_SYSTEM,
            user_text=prompts.slab_analysis_prompt(),
            images=[ImageInput(image, fmt)],
            max_tokens=900,
        )
        return parse_model_json(text, SlabVisionResult)

    def assess_room(self, image: bytes, fmt: ImageFormat) -> RoomSuitability:
        self._guard.check_image(image)
        text = self._client.converse(
            self._model_id,
            system=prompts.VISION_SYSTEM,
            user_text=prompts.room_prompt_check(),
            images=[ImageInput(image, fmt)],
            max_tokens=300,
        )
        return parse_model_json(text, RoomSuitability)

    def recommend_legs(
        self, furniture: FurnitureType, ratio: float | None, summary: str | None
    ) -> RawLegChoice:
        text = self._client.converse(
            self._model_id,
            system=prompts.VISION_SYSTEM,
            user_text=prompts.leg_prompt(furniture, ratio, summary),
            max_tokens=300,
        )
        return parse_model_json(text, RawLegChoice)
