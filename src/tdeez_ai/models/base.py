"""Response envelope and shared base model (camelCase on the wire)."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

DISCLAIMER = (
    "AI-generated draft for visualization only. It is not a formal quote, an exact measurement "
    "or a structural validation. TDeez will confirm all details with you."
)


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class ApiResponse[T](CamelModel):
    is_success: bool
    message: str
    data: T


def ok[T](data: T, message: str = "OK") -> ApiResponse[T]:
    return ApiResponse[T](is_success=True, message=message, data=data)
