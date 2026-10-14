"""Slab inventory, upload, similarity and analysis schemas."""

from typing import Literal, Self

from pydantic import Field, model_validator

from tdeez_ai.models.base import DISCLAIMER, CamelModel
from tdeez_ai.models.common import Availability, FurnitureType, Species


class SlabDimensions(CamelModel):
    length_in: float = Field(gt=0, le=400)
    width_in: float = Field(gt=0, le=200)
    thickness_in: float = Field(gt=0, le=20)


class Slab(CamelModel):
    id: str
    name: str
    species: Species
    availability: Availability
    dimensions: SlabDimensions | None = None
    image_key: str
    description: str = ""


class SlabView(Slab):
    image_url: str


class UploadUrlRequest(CamelModel):
    content_type: str
    size_bytes: int = Field(gt=0)


class UploadUrlResponse(CamelModel):
    upload_url: str
    image_key: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str]
    expires_in_seconds: int
    max_bytes: int


class SlabSource(CamelModel):
    """A slab reference: either an inventory slab or an uploaded image (exactly one)."""

    slab_id: str | None = Field(default=None, max_length=100)
    image_key: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.slab_id is None) == (self.image_key is None):
            raise ValueError("Provide exactly one of slabId or imageKey")
        return self


class ActiveSlab(CamelModel):
    source: Literal["inventory", "upload"]
    slab_id: str | None = None
    image_key: str
    isolated_image_key: str | None = None


class SessionMeta(CamelModel):
    active_slab: ActiveSlab | None = None
    furniture_type: FurnitureType | None = None
    concept_generation: int = 0


class SimilarityRequest(CamelModel):
    image_key: str = Field(max_length=300)
    top_k: int = Field(default=5, ge=3, le=5)


class SimilarSlab(CamelModel):
    slab: SlabView
    score: float = Field(ge=0, le=1)


class SimilarityResponse(CamelModel):
    matches: list[SimilarSlab]
    disclaimer: str = DISCLAIMER


DefectKind = Literal["knot", "crack", "bark_inclusion"]


class Defect(CamelModel):
    kind: DefectKind
    severity: Literal["low", "medium", "high"]
    location_hint: str = Field(default="", max_length=200)


class SlabVisionResult(CamelModel):
    """Strict shape the multimodal model must return (validated, never trusted)."""

    shape_description: str = Field(max_length=500)
    length_to_width_ratio: float | None = Field(default=None, gt=0, le=30)
    defects: list[Defect] = Field(default_factory=list, max_length=20)
    species: str | None = None
    species_confidence: float = Field(ge=0, le=1)
    summary: str = Field(max_length=1200)


class SpeciesGuess(CamelModel):
    species: Species | None
    confidence: float = Field(ge=0, le=1)
    supported_species: list[Species]


class Proportions(CamelModel):
    length_to_width_ratio: float | None
    basis: Literal["image_estimate", "inventory_listing"]
    known_dimensions: SlabDimensions | None = None
    note: str


class SlabAnalysis(CamelModel):
    isolated_image_key: str
    isolated_image_url: str
    outline: str
    proportions: Proportions
    defects: list[Defect]
    species: SpeciesGuess
    needs_review: bool
    review_message: str | None = None
    summary: str
    limitations: str
    disclaimer: str = DISCLAIMER
