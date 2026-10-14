"""My Creations and public share schemas."""

from datetime import datetime

from pydantic import Field

from tdeez_ai.models.base import DISCLAIMER, CamelModel
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.customization import CustomizationValues
from tdeez_ai.models.slab import SlabSource


class ConceptRef(CamelModel):
    id: str = Field(max_length=100)
    style: str = Field(max_length=100)
    furniture_type: FurnitureType
    image_key: str = Field(max_length=300)


class SaveCreationRequest(CamelModel):
    name: str = Field(min_length=1, max_length=80)
    slab: SlabSource
    concept: ConceptRef | None = None
    customization: CustomizationValues
    visualization_image_key: str | None = Field(default=None, max_length=300)
    room_visualization_image_key: str | None = Field(default=None, max_length=300)


class UpdateCreationRequest(CamelModel):
    """Only these fields are editable after saving."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    concept: ConceptRef | None = None
    customization: CustomizationValues | None = None
    visualization_image_key: str | None = Field(default=None, max_length=300)
    room_visualization_image_key: str | None = Field(default=None, max_length=300)


class Creation(CamelModel):
    """Stored record. The owner id is a repository key and is never part of this model."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    slab: SlabSource
    concept: ConceptRef | None = None
    customization: CustomizationValues
    visualization_image_key: str | None = None
    room_visualization_image_key: str | None = None
    share_token: str | None = None


class CreationView(CamelModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    slab: SlabSource
    concept: ConceptRef | None
    customization: CustomizationValues
    slab_image_url: str | None
    concept_image_url: str | None
    visualization_url: str | None
    room_visualization_url: str | None
    share_token: str | None
    disclaimer: str = DISCLAIMER


class ShareLink(CamelModel):
    token: str
    path: str


class SharedCreation(CamelModel):
    """Public, read-only projection: no owner, no keys, no customer data."""

    name: str
    furniture_type: FurnitureType | None
    customization: CustomizationValues
    concept_image_url: str | None
    visualization_url: str | None
    room_visualization_url: str | None
    disclaimer: str = DISCLAIMER
