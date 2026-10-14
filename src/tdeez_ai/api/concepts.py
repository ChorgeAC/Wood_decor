from fastapi import APIRouter

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.concept import ConceptRequest, ConceptResponse

router = APIRouter(tags=["concepts"])


@router.post("/concepts/generate")
def generate_concepts(
    body: ConceptRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[ConceptResponse]:
    """TBC-011: 2-3 concepts; also used for regenerate and to change furniture type."""
    return ok(c.concepts.generate(user.sub, body))
