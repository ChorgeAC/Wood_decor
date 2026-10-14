"""Room photo schemas."""

from pydantic import Field

from tdeez_ai.models.base import DISCLAIMER, CamelModel
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.customization import CustomizationValues


class RoomValidateRequest(CamelModel):
    image_key: str = Field(max_length=300)


class RoomSuitability(CamelModel):
    """Strict shape the multimodal model must return."""

    suitable: bool
    confidence: float = Field(ge=0, le=1)
    has_wall: bool
    has_floor: bool
    reason: str = Field(default="", max_length=300)


class RoomValidation(CamelModel):
    suitable: bool
    confidence: float
    fallback_to_standalone: bool
    recoverable: bool
    message: str
    disclaimer: str = DISCLAIMER


class RoomVisualizeRequest(CamelModel):
    room_image_key: str = Field(max_length=300)
    furniture_type: FurnitureType
    customization: CustomizationValues | None = None


class RoomVisualization(CamelModel):
    fallback_to_standalone: bool
    image_key: str | None = None
    image_url: str | None = None
    message: str
    disclaimer: str = DISCLAIMER
