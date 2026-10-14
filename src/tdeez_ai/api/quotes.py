from fastapi import APIRouter

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.quote import QuoteConfirmation, QuoteRequest

router = APIRouter(tags=["quotes"])


@router.post("/quotes")
def submit_quote(
    body: QuoteRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[QuoteConfirmation]:
    """TBC-013"""
    return ok(c.quotes.submit(user.sub, body), "Quote request received")
