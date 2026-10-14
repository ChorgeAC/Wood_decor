"""Profile schemas."""

from typing import Self

from pydantic import Field, model_validator

from tdeez_ai.models.base import CamelModel
from tdeez_ai.models.quote import EMAIL_PATTERN, PHONE_PATTERN


class Profile(CamelModel):
    name: str | None = Field(default=None, max_length=100)
    email: str | None = Field(default=None, max_length=254, pattern=EMAIL_PATTERN)
    phone: str | None = Field(default=None, pattern=PHONE_PATTERN)
    extras: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bound_extras(self) -> Self:
        if len(self.extras) > 20 or any(
            len(k) > 50 or len(v) > 200 for k, v in self.extras.items()
        ):
            raise ValueError("extras are limited to 20 entries (key 50, value 200 chars)")
        return self


class UpdateProfileRequest(Profile):
    pass
