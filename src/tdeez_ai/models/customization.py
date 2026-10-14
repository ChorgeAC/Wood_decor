"""Customization schemas."""

from enum import StrEnum

from pydantic import Field

from tdeez_ai.models.base import DISCLAIMER, CamelModel
from tdeez_ai.models.common import FurnitureType


class EdgeType(StrEnum):
    LIVE = "live"
    STRAIGHT = "straight"
    CHAMFERED = "chamfered"
    ROUNDED = "rounded"


class EpoxyType(StrEnum):
    NONE = "none"
    CLEAR = "clear"
    TINTED = "tinted"
    RIVER = "river"


class LegStyle(StrEnum):
    HAIRPIN = "hairpin"
    TRESTLE = "trestle"
    U_SHAPED = "u_shaped"
    SLED = "sled"
    X_FRAME = "x_frame"


class WoodOrientation(StrEnum):
    LENGTHWISE = "lengthwise"
    CROSSWISE = "crosswise"


class Dimensions(CamelModel):
    """Requested (target) dimensions in inches. Not a measurement of any image."""

    length_in: float
    width_in: float
    height_in: float


class CustomizationValues(CamelModel):
    edge_type: EdgeType = EdgeType.LIVE
    epoxy_type: EpoxyType = EpoxyType.NONE
    epoxy_color: str | None = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")
    leg_style: LegStyle = LegStyle.HAIRPIN
    wood_orientation: WoodOrientation = WoodOrientation.LENGTHWISE
    dimensions: Dimensions | None = None


class ValidateCustomizationRequest(CamelModel):
    furniture_type: FurnitureType
    customization: CustomizationValues


class ValidationIssue(CamelModel):
    field: str
    message: str


class ValidationResult(CamelModel):
    valid: bool
    issues: list[ValidationIssue]


class LegRecommendationRequest(CamelModel):
    furniture_type: FurnitureType
    dimensions: Dimensions | None = None
    slab_summary: str | None = Field(default=None, max_length=1000)


class LegRecommendation(CamelModel):
    leg_style: LegStyle
    rationale: str
    disclaimer: str = DISCLAIMER


class RenderRequest(CamelModel):
    furniture_type: FurnitureType
    customization: CustomizationValues
    base_image_key: str = Field(max_length=300)


class PriceRange(CamelModel):
    low: float
    high: float
    currency: str
    note: str = "Rough estimate only. This is not a quote."


class RenderResponse(CamelModel):
    image_key: str
    image_url: str
    validation: ValidationResult
    price_range: PriceRange | None = None
    disclaimer: str = DISCLAIMER
