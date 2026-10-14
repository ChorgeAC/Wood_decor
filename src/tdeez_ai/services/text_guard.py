"""Hard rule 1: AI text must not imply exact dimensions, structural validation or a quote."""

import re

_MEASUREMENT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:-|to)?\s?(?:\d+(?:[.,]\d+)?\s?)?"
    r"(?:inches|inch|in|cm|mm|ft|feet|foot|meters|metres|meter|metre|m)\b|\b\d+(?:\.\d+)?\s?[\"']",
    re.IGNORECASE,
)
_STRUCTURAL = re.compile(
    r"structurally\s+(?:sound|safe|certified)|load[- ]bearing|weight\s+(?:rating|capacity)|"
    r"rated\s+for|will\s+(?:safely\s+)?support|engineer(?:ed|ing)\s+(?:approved|certified)",
    re.IGNORECASE,
)
_PRICE = re.compile(r"[$€£]\s?\d|\b\d[\d,.]*\s?(?:usd|dollars|eur|gbp)\b", re.IGNORECASE)


def guard_text(text: str) -> str:
    """Neutralise measurements, structural claims and prices in model-written text."""
    cleaned = _MEASUREMENT.sub("[measurement omitted]", text)
    cleaned = _STRUCTURAL.sub("[claim omitted]", cleaned)
    cleaned = _PRICE.sub("[price omitted]", cleaned)
    return cleaned.strip()
