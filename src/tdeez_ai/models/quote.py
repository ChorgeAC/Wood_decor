"""Quote request schemas."""

from datetime import datetime

from pydantic import Field

from tdeez_ai.models.base import CamelModel
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.customization import CustomizationValues
from tdeez_ai.models.slab import SlabSource

QUOTE_DISCLAIMER = (
    "This is a request for a quote, not a quote. TDeez will review your design and contact you "
    "with confirmed details and pricing."
)

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE_PATTERN = r"^\+?[0-9 ()\-.]{7,20}$"


class CustomerInfo(CamelModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=254, pattern=EMAIL_PATTERN)
    phone: str | None = Field(default=None, pattern=PHONE_PATTERN)
    notes: str | None = Field(default=None, max_length=1000)


class DesignConfig(CamelModel):
    furniture_type: FurnitureType
    customization: CustomizationValues


class QuoteRequest(CamelModel):
    customer: CustomerInfo
    slab: SlabSource
    design: DesignConfig
    visualization_image_key: str | None = Field(default=None, max_length=300)
    creation_id: str | None = Field(default=None, max_length=100)


class QuoteRecord(CamelModel):
    reference: str
    submitted_at: datetime
    request: QuoteRequest
    email_sent: bool


class QuoteConfirmation(CamelModel):
    reference: str
    submitted_at: datetime
    message: str
    disclaimer: str = QUOTE_DISCLAIMER
