from fastapi import APIRouter

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.room import (
    RoomValidateRequest,
    RoomValidation,
    RoomVisualization,
    RoomVisualizeRequest,
)
from tdeez_ai.models.slab import UploadUrlRequest, UploadUrlResponse

router = APIRouter(tags=["rooms"])


@router.post("/rooms/upload-url")
def room_upload_url(
    body: UploadUrlRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[UploadUrlResponse]:
    """TBC-009"""
    return ok(c.slab_source.create_upload(user.sub, "rooms", body))


@router.post("/rooms/validate")
def validate_room(
    body: RoomValidateRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[RoomValidation]:
    """TBC-009: suitability check; unsuitable photos return fallbackToStandalone=true."""
    return ok(c.rooms.validate(user.sub, body))


@router.post("/rooms/visualize")
def visualize_room(
    body: RoomVisualizeRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[RoomVisualization]:
    """TBC-009"""
    return ok(c.rooms.visualize(user.sub, body))
