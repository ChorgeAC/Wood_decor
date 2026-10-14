"""Room photo (TBC-009): suitability check + furniture-in-room visualization. Never blocks the flow."""

import logging

from tdeez_ai.bedrock import prompts
from tdeez_ai.bedrock.canvas import ImageGenerator
from tdeez_ai.bedrock.vision import VisionAnalyzer
from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import AppError
from tdeez_ai.core.logging import log_event
from tdeez_ai.models.room import (
    RoomSuitability,
    RoomValidateRequest,
    RoomValidation,
    RoomVisualization,
    RoomVisualizeRequest,
)
from tdeez_ai.services.image_access import ImageAccess
from tdeez_ai.services.text_guard import guard_text

logger = logging.getLogger(__name__)

MSG_UNSUITABLE = (
    "We couldn't see enough floor and wall in that photo. Try another photo, or continue "
    "without a room image."
)
MSG_FAILED = (
    "We couldn't process the room photo right now. You can retry, or continue without a room image."
)
MSG_OK = "Room photo looks good."
_ROOM_NEGATIVE = "floating furniture, distorted room, extra furniture, text, watermark, people"


class RoomService:
    def __init__(
        self,
        access: ImageAccess,
        images: ImageGenerator,
        vision: VisionAnalyzer,
        settings: Settings,
    ) -> None:
        self._access = access
        self._images = images
        self._vision = vision
        self._settings = settings

    def validate(self, owner: str, req: RoomValidateRequest) -> RoomValidation:
        # Auth/ownership errors must surface; only processing failures degrade to the fallback.
        image = self._access.read(owner, req.image_key)
        try:
            verdict = self._vision.assess_room(image.data, image.fmt)
        except AppError as exc:
            log_event(logger, "room_validation_failed", error_code=exc.code)
            return self._fallback(MSG_FAILED, confidence=0.0)
        return self._decide(verdict)

    def visualize(self, owner: str, req: RoomVisualizeRequest) -> RoomVisualization:
        validation = self.validate(owner, RoomValidateRequest(image_key=req.room_image_key))
        if validation.fallback_to_standalone:
            return RoomVisualization(fallback_to_standalone=True, message=validation.message)
        image = self._access.read(owner, req.room_image_key)
        try:
            png = self._images.inpaint(
                image.data,
                "empty floor area in the middle of the room",
                prompts.room_prompt(req.furniture_type, req.customization),
                _ROOM_NEGATIVE,
                seed=7,
            )
        except AppError as exc:
            log_event(logger, "room_visualization_failed", error_code=exc.code)
            return RoomVisualization(fallback_to_standalone=True, message=MSG_FAILED)
        key = self._access.save(owner, "rooms", png, "png")
        return RoomVisualization(
            fallback_to_standalone=False,
            image_key=key,
            image_url=self._access.url(key),
            message="Here's a draft of your design in your room.",
        )

    def _decide(self, verdict: RoomSuitability) -> RoomValidation:
        usable = (
            verdict.suitable
            and verdict.has_floor
            and verdict.has_wall
            and verdict.confidence >= self._settings.room_min_confidence
        )
        if not usable:
            return self._fallback(
                MSG_UNSUITABLE, confidence=verdict.confidence, reason=verdict.reason
            )
        return RoomValidation(
            suitable=True,
            confidence=verdict.confidence,
            fallback_to_standalone=False,
            recoverable=True,
            message=MSG_OK,
        )

    @staticmethod
    def _fallback(message: str, *, confidence: float, reason: str = "") -> RoomValidation:
        hint = f" ({guard_text(reason)})" if reason else ""
        return RoomValidation(
            suitable=False,
            confidence=confidence,
            fallback_to_standalone=True,
            recoverable=True,
            message=message + hint,
        )
