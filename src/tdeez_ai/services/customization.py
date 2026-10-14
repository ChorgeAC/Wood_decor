"""Customization (TBC-012): deterministic validation, AI leg recommendation, preview re-render."""

import logging
from dataclasses import dataclass

from tdeez_ai.bedrock import prompts
from tdeez_ai.bedrock.canvas import ImageGenerator
from tdeez_ai.bedrock.vision import VisionAnalyzer
from tdeez_ai.core.errors import AppError, ValidationFailure
from tdeez_ai.core.keys import assert_readable
from tdeez_ai.core.logging import log_event
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.customization import (
    CustomizationValues,
    EpoxyType,
    LegRecommendation,
    LegRecommendationRequest,
    LegStyle,
    RenderRequest,
    RenderResponse,
    ValidationIssue,
    ValidationResult,
)
from tdeez_ai.services.image_access import ImageAccess
from tdeez_ai.services.pricing import PricingService
from tdeez_ai.services.text_guard import guard_text

logger = logging.getLogger(__name__)

SIMILARITY_STRENGTH = 0.9


@dataclass(frozen=True)
class Range:
    low: float
    high: float


@dataclass(frozen=True)
class DimensionRanges:
    length: Range
    width: Range
    height: Range


# Inches. Deterministic business ranges, not engineering limits.
DIMENSION_RANGES: dict[FurnitureType, DimensionRanges] = {
    FurnitureType.DINING_TABLE: DimensionRanges(Range(48, 120), Range(30, 48), Range(28, 31)),
    FurnitureType.COFFEE_TABLE: DimensionRanges(Range(30, 60), Range(16, 36), Range(14, 20)),
    FurnitureType.DESK: DimensionRanges(Range(40, 80), Range(20, 36), Range(28, 31)),
}

_DEFAULT_LEGS: dict[FurnitureType, tuple[LegStyle, str]] = {
    FurnitureType.DINING_TABLE: (
        LegStyle.TRESTLE,
        "Trestle legs give a long tabletop steady support.",
    ),
    FurnitureType.COFFEE_TABLE: (LegStyle.HAIRPIN, "Hairpin legs keep a low table light and airy."),
    FurnitureType.DESK: (LegStyle.U_SHAPED, "U-shaped legs leave generous legroom at a desk."),
}


def validate_customization(
    furniture: FurnitureType, values: CustomizationValues
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    ranges = DIMENSION_RANGES[furniture]
    if values.dimensions is not None:
        d = values.dimensions
        for name, value, rng in (
            ("dimensions.lengthIn", d.length_in, ranges.length),
            ("dimensions.widthIn", d.width_in, ranges.width),
            ("dimensions.heightIn", d.height_in, ranges.height),
        ):
            if not rng.low <= value <= rng.high:
                issues.append(
                    ValidationIssue(
                        field=name,
                        message=f"Must be between {rng.low:g} and {rng.high:g} inches.",
                    )
                )
    if values.epoxy_type is EpoxyType.TINTED and values.epoxy_color is None:
        issues.append(
            ValidationIssue(field="epoxyColor", message="Choose a color for tinted epoxy.")
        )
    if values.epoxy_type is EpoxyType.NONE and values.epoxy_color is not None:
        issues.append(
            ValidationIssue(field="epoxyColor", message="Epoxy color needs an epoxy type.")
        )
    return ValidationResult(valid=not issues, issues=issues)


class CustomizationService:
    def __init__(
        self,
        access: ImageAccess,
        images: ImageGenerator,
        vision: VisionAnalyzer,
        pricing: PricingService,
    ) -> None:
        self._access = access
        self._images = images
        self._vision = vision
        self._pricing = pricing

    def recommend_legs(self, req: LegRecommendationRequest) -> LegRecommendation:
        ratio = None
        if req.dimensions is not None and req.dimensions.width_in > 0:
            ratio = req.dimensions.length_in / req.dimensions.width_in
        try:
            raw = self._vision.recommend_legs(req.furniture_type, ratio, req.slab_summary)
            style = LegStyle(raw.leg_style.strip().lower())
            return LegRecommendation(leg_style=style, rationale=guard_text(raw.rationale))
        except (AppError, ValueError) as exc:
            # A recommendation is a nicety: fall back to a deterministic default.
            log_event(logger, "leg_recommendation_fallback", error_type=type(exc).__name__)
            style, why = _DEFAULT_LEGS[req.furniture_type]
            return LegRecommendation(leg_style=style, rationale=why)

    def validate(self, furniture: FurnitureType, values: CustomizationValues) -> ValidationResult:
        return validate_customization(furniture, values)

    def render(self, owner: str, req: RenderRequest) -> RenderResponse:
        validation = validate_customization(req.furniture_type, req.customization)
        if not validation.valid:
            raise ValidationFailure(
                "Some customization values are out of range. Please adjust them."
            )
        base_key = assert_readable(owner, req.base_image_key)
        base = self._access.read(owner, base_key)
        png = self._images.variation(
            base.data,
            prompts.customization_prompt(req.furniture_type, req.customization),
            prompts.GEOMETRY_NEGATIVE,
            similarity=SIMILARITY_STRENGTH,
            seed=11,
        )
        key = self._access.save(owner, "renders", png, "png")
        return RenderResponse(
            image_key=key,
            image_url=self._access.url(key),
            validation=validation,
            price_range=self._pricing.estimate(req.furniture_type, req.customization),
        )
