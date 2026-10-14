"""Offline fakes for the Bedrock-backed interfaces (selected by USE_MOCK=true)."""

import hashlib
import math

from tdeez_ai.bedrock.prompts import FURNITURE_LABEL
from tdeez_ai.bedrock.vision import RawLegChoice
from tdeez_ai.core.images import ImageFormat, make_png
from tdeez_ai.models.common import FurnitureType, Species
from tdeez_ai.models.room import RoomSuitability
from tdeez_ai.models.slab import Defect, SlabVisionResult

_SPECIES = list(Species)


def _digest(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


class MockImageGenerator:
    """Returns small deterministic PNGs; records nothing and never calls AWS."""

    def _png(self, *parts: object) -> bytes:
        d = _digest(repr(parts).encode())
        return make_png(64, 64, (d[0], d[1], d[2]))

    def variation(
        self, source: bytes, prompt: str, negative: str, *, similarity: float, seed: int
    ) -> bytes:
        return self._png("variation", _digest(source), prompt, seed)

    def inpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        return self._png("inpaint", _digest(image), prompt, seed)

    def outpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        return self._png("outpaint", _digest(image), prompt, seed)

    def text_to_image(self, prompt: str, negative: str, *, seed: int) -> bytes:
        return self._png("text", prompt, seed)

    def remove_background(self, image: bytes) -> bytes:
        return self._png("bgremoved", _digest(image))


class MockVisionAnalyzer:
    def analyze_slab(self, image: bytes, fmt: ImageFormat) -> SlabVisionResult:
        d = _digest(image)
        species = _SPECIES[d[0] % len(_SPECIES)]
        return SlabVisionResult(
            shape_description="Elongated slab with a flowing natural live edge on both sides.",
            length_to_width_ratio=round(2.0 + d[1] / 255, 2),
            defects=[Defect(kind="knot", severity="low", location_hint="near the center")],
            species=species.value,
            species_confidence=0.82,
            summary="A characterful live-edge slab with warm grain and a gently curving outline.",
        )

    def assess_room(self, image: bytes, fmt: ImageFormat) -> RoomSuitability:
        return RoomSuitability(
            suitable=True,
            confidence=0.9,
            has_wall=True,
            has_floor=True,
            reason="Floor and wall visible.",
        )

    def recommend_legs(
        self, furniture: FurnitureType, ratio: float | None, summary: str | None
    ) -> RawLegChoice:
        style = "trestle" if furniture is FurnitureType.DINING_TABLE else "hairpin"
        label = FURNITURE_LABEL[furniture]
        return RawLegChoice(
            leg_style=style, rationale=f"{style.capitalize()} legs suit a {label} with this slab."
        )


class MockEmbedder:
    """Deterministic pseudo-embedding derived from the image hash (unit length)."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    def embed_image(self, image: bytes) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self._dimensions:
            block = hashlib.sha256(image + counter.to_bytes(4, "big")).digest()
            values.extend((b - 127.5) / 127.5 for b in block)
            counter += 1
        vec = values[: self._dimensions]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
