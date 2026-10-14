"""Optional rough price range. Absent unless a pricing rules config is provided."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from tdeez_ai.core.config import Settings
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.customization import CustomizationValues, EpoxyType, PriceRange


class PricingRules(BaseModel):
    currency: str = "USD"
    base: dict[FurnitureType, float]
    epoxy_surcharge: float = Field(default=0, ge=0)
    spread_pct: float = Field(default=0.15, gt=0, lt=1)


class PricingService:
    def __init__(self, rules: PricingRules | None) -> None:
        self._rules = rules

    @classmethod
    def from_settings(cls, settings: Settings) -> "PricingService":
        if not settings.pricing_rules_provided or not settings.pricing_rules_path:
            return cls(None)
        raw = Path(settings.pricing_rules_path).read_text(encoding="utf-8")
        return cls(PricingRules.model_validate(json.loads(raw)))

    def estimate(self, furniture: FurnitureType, values: CustomizationValues) -> PriceRange | None:
        rules = self._rules
        if rules is None or furniture not in rules.base:
            return None
        mid = rules.base[furniture]
        if values.epoxy_type is not EpoxyType.NONE:
            mid += rules.epoxy_surcharge
        return PriceRange(
            low=round(mid * (1 - rules.spread_pct), 2),
            high=round(mid * (1 + rules.spread_pct), 2),
            currency=rules.currency,
        )
