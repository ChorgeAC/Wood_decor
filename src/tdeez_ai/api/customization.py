from fastapi import APIRouter

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.customization import (
    LegRecommendation,
    LegRecommendationRequest,
    RenderRequest,
    RenderResponse,
    ValidateCustomizationRequest,
    ValidationResult,
)

router = APIRouter(tags=["customization"])


@router.post("/customization/validate")
def validate_customization(
    body: ValidateCustomizationRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[ValidationResult]:
    """TBC-012: deterministic range checks."""
    return ok(c.customization.validate(body.furniture_type, body.customization))


@router.post("/customization/recommend-legs")
def recommend_legs(
    body: LegRecommendationRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[LegRecommendation]:
    """TBC-012"""
    return ok(c.customization.recommend_legs(body))


@router.post("/customization/render")
def render_preview(
    body: RenderRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[RenderResponse]:
    """TBC-012: validate, then re-render the preview."""
    return ok(c.customization.render(user.sub, body))
