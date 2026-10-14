"""Concept generation schemas."""

from pydantic import Field

from tdeez_ai.models.base import DISCLAIMER, CamelModel
from tdeez_ai.models.common import FurnitureType


class ConceptRequest(CamelModel):
    furniture_type: FurnitureType
    slab_image_key: str | None = Field(default=None, max_length=300)
    count: int = Field(default=3, ge=2, le=3)
    regenerate: bool = False


class Concept(CamelModel):
    id: str
    style: str
    title: str
    furniture_type: FurnitureType
    image_key: str
    image_url: str


class ConceptResponse(CamelModel):
    furniture_type: FurnitureType
    generation: int
    concepts: list[Concept]
    disclaimer: str = DISCLAIMER
