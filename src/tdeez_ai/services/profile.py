"""Profile (POC-031). Cognito owns credentials; DynamoDB holds the extra fields."""

from tdeez_ai.core.auth import Principal
from tdeez_ai.models.profile import Profile, UpdateProfileRequest
from tdeez_ai.repositories.base import ProfileRepository


class ProfileService:
    def __init__(self, repo: ProfileRepository) -> None:
        self._repo = repo

    def get(self, user: Principal) -> Profile:
        stored = self._repo.get_profile(user.sub)
        if stored is not None:
            return stored
        return Profile(email=user.email)  # first visit: prefill from the verified token

    def update(self, user: Principal, req: UpdateProfileRequest) -> Profile:
        profile = Profile(name=req.name, email=req.email, phone=req.phone, extras=req.extras)
        self._repo.put_profile(user.sub, profile)
        return profile
