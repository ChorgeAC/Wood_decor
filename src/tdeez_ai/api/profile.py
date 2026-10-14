from fastapi import APIRouter

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.profile import Profile, UpdateProfileRequest

router = APIRouter(tags=["profile"])


@router.get("/profile")
def get_profile(user: UserDep, c: ContainerDep) -> ApiResponse[Profile]:
    """POC-031"""
    return ok(c.profiles.get(user))


@router.put("/profile")
def update_profile(
    body: UpdateProfileRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[Profile]:
    """POC-031"""
    return ok(c.profiles.update(user, body), "Profile updated")
