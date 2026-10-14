"""Slab analysis (TBC-010): background removal, outline, proportions, defects, species, summary."""

from tdeez_ai.bedrock.canvas import ImageGenerator
from tdeez_ai.bedrock.vision import VisionAnalyzer
from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import NotFound
from tdeez_ai.models.common import Species
from tdeez_ai.models.slab import (
    ActiveSlab,
    Proportions,
    SlabAnalysis,
    SlabDimensions,
    SlabSource,
    SpeciesGuess,
)
from tdeez_ai.repositories.base import ProfileRepository, SlabRepository
from tdeez_ai.services.image_access import ImageAccess
from tdeez_ai.services.text_guard import guard_text

LIMITATIONS = (
    "This analysis is an AI-generated draft based on one photo. Proportions are relative "
    "estimates, defect detection is limited to visible knots, cracks and bark inclusions, and "
    "species identification may be wrong. It is not an exact measurement or a structural "
    "assessment. TDeez will confirm the details."
)
REVIEW_MESSAGE = "We're not fully sure about the species, so TDeez will review it for you."
_PROPORTIONS_NOTE_IMAGE = "Relative estimate from the photo only. Exact dimensions are not known."
_PROPORTIONS_NOTE_INVENTORY = (
    "Nominal dimensions from the TDeez inventory listing; subject to confirmation."
)


class SlabIsolator:
    """Removes the background once per slab image and remembers it in the session."""

    def __init__(
        self,
        access: ImageAccess,
        images: ImageGenerator,
        profiles: ProfileRepository,
    ) -> None:
        self._access = access
        self._images = images
        self._profiles = profiles

    def isolate(self, owner: str, source_key: str) -> str:
        image = self._access.read(owner, source_key)
        png = self._images.remove_background(image.data)
        key = self._access.save(owner, "derived", png, "png")
        session = self._profiles.get_session(owner)
        active = session.active_slab
        if active is not None and active.image_key == source_key:
            session = session.model_copy(
                update={"active_slab": active.model_copy(update={"isolated_image_key": key})}
            )
            self._profiles.put_session(owner, session)
        return key


class SlabAnalysisService:
    def __init__(
        self,
        access: ImageAccess,
        isolator: SlabIsolator,
        vision: VisionAnalyzer,
        slabs: SlabRepository,
        profiles: ProfileRepository,
        settings: Settings,
    ) -> None:
        self._access = access
        self._isolator = isolator
        self._vision = vision
        self._slabs = slabs
        self._profiles = profiles
        self._settings = settings

    def analyze(self, owner: str, source: SlabSource) -> SlabAnalysis:
        known: SlabDimensions | None = None
        if source.slab_id is not None:
            slab = self._slabs.get(source.slab_id)
            if slab is None:
                raise NotFound("That slab could not be found.")
            image_key, known = slab.image_key, slab.dimensions
            self._ensure_active(
                owner, ActiveSlab(source="inventory", slab_id=slab.id, image_key=image_key)
            )
        else:
            assert source.image_key is not None
            image_key = source.image_key
            self._ensure_active(owner, ActiveSlab(source="upload", image_key=image_key))

        image = self._access.read(owner, image_key)
        isolated_key = self._isolator.isolate(owner, image_key)
        result = self._vision.analyze_slab(image.data, image.fmt)

        species = _match_species(result.species)
        confidence = result.species_confidence if species is not None else 0.0
        needs_review = species is None or confidence < self._settings.species_review_threshold

        return SlabAnalysis(
            isolated_image_key=isolated_key,
            isolated_image_url=self._access.url(isolated_key),
            outline=guard_text(result.shape_description),
            proportions=Proportions(
                length_to_width_ratio=result.length_to_width_ratio,
                basis="inventory_listing" if known else "image_estimate",
                known_dimensions=known,
                note=_PROPORTIONS_NOTE_INVENTORY if known else _PROPORTIONS_NOTE_IMAGE,
            ),
            defects=result.defects,
            species=SpeciesGuess(
                species=species, confidence=round(confidence, 3), supported_species=list(Species)
            ),
            needs_review=needs_review,
            review_message=REVIEW_MESSAGE if needs_review else None,
            summary=guard_text(result.summary),
            limitations=LIMITATIONS,
        )

    def _ensure_active(self, owner: str, active: ActiveSlab) -> None:
        session = self._profiles.get_session(owner)
        current = session.active_slab
        if current is None or current.image_key != active.image_key:
            self._profiles.put_session(owner, session.model_copy(update={"active_slab": active}))


def _match_species(raw: str | None) -> Species | None:
    if raw is None:
        return None
    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return Species(normalized)
    except ValueError:
        return None
