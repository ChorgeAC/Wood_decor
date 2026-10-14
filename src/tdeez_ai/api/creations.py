from fastapi import APIRouter

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.creation import (
    CreationView,
    SaveCreationRequest,
    ShareLink,
    UpdateCreationRequest,
)

router = APIRouter(tags=["creations"])


@router.post("/creations")
def save_creation(
    body: SaveCreationRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[CreationView]:
    """POC-006 / TBC-013"""
    return ok(c.creations.save(user.sub, body), "Creation saved")


@router.get("/creations")
def list_creations(user: UserDep, c: ContainerDep) -> ApiResponse[list[CreationView]]:
    """POC-006"""
    return ok(c.creations.list(user.sub))


@router.get("/creations/{creation_id}")
def get_creation(creation_id: str, user: UserDep, c: ContainerDep) -> ApiResponse[CreationView]:
    """POC-006"""
    return ok(c.creations.get(user.sub, creation_id))


@router.put("/creations/{creation_id}")
def update_creation(
    creation_id: str, body: UpdateCreationRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[CreationView]:
    """POC-006"""
    return ok(c.creations.update(user.sub, creation_id, body), "Creation updated")


@router.delete("/creations/{creation_id}")
def delete_creation(creation_id: str, user: UserDep, c: ContainerDep) -> ApiResponse[None]:
    """POC-006"""
    c.creations.delete(user.sub, creation_id)
    return ok(None, "Creation deleted")


@router.post("/creations/{creation_id}/share")
def enable_share(creation_id: str, user: UserDep, c: ContainerDep) -> ApiResponse[ShareLink]:
    """Public share"""
    return ok(c.creations.enable_share(user.sub, creation_id), "Share link created")


@router.delete("/creations/{creation_id}/share")
def disable_share(creation_id: str, user: UserDep, c: ContainerDep) -> ApiResponse[None]:
    """Public share"""
    c.creations.disable_share(user.sub, creation_id)
    return ok(None, "Share link removed")
