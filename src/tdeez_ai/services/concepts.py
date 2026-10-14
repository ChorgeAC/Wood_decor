"""Furniture type + concepts (TBC-011): 2-3 distinct concepts conditioned on the isolated slab."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from tdeez_ai.bedrock import prompts
from tdeez_ai.bedrock.canvas import ImageGenerator
from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import AppError, BadRequest, UpstreamError
from tdeez_ai.core.keys import assert_readable, user_prefix
from tdeez_ai.core.logging import log_event
from tdeez_ai.models.concept import Concept, ConceptRequest, ConceptResponse
from tdeez_ai.repositories.base import ProfileRepository
from tdeez_ai.services.image_access import ImageAccess
from tdeez_ai.services.slab_analysis import SlabIsolator

logger = logging.getLogger(__name__)

# High similarity keeps the slab geometry; the style prompt only changes the setting and base.
SIMILARITY_STRENGTH = 0.85
_SEED_STRIDE = 1000


class ConceptService:
    def __init__(
        self,
        access: ImageAccess,
        isolator: SlabIsolator,
        images: ImageGenerator,
        profiles: ProfileRepository,
        settings: Settings,
    ) -> None:
        self._access = access
        self._isolator = isolator
        self._images = images
        self._profiles = profiles
        self._settings = settings

    def generate(self, owner: str, req: ConceptRequest) -> ConceptResponse:
        if req.count > self._settings.max_images_per_request:
            raise BadRequest("Too many concepts requested.")
        session = self._profiles.get_session(owner)

        if req.slab_image_key is not None:
            source_key = assert_readable(owner, req.slab_image_key)
            isolated_key = source_key
            if not source_key.startswith(f"{user_prefix(owner)}derived/"):
                isolated_key = self._isolator.isolate(owner, source_key)
        elif session.active_slab is not None:
            active = session.active_slab
            isolated_key = active.isolated_image_key or self._isolator.isolate(
                owner, active.image_key
            )
        else:
            raise BadRequest("Please select or upload a slab first.")

        generation = session.concept_generation + (1 if req.regenerate else 0)
        base = self._access.read(owner, isolated_key)
        styles = prompts.CONCEPT_STYLES[: req.count]

        def render(index: int) -> Concept:
            style_key, title, descriptor = styles[index]
            png = self._images.variation(
                base.data,
                prompts.concept_prompt(req.furniture_type, descriptor),
                prompts.GEOMETRY_NEGATIVE,
                similarity=SIMILARITY_STRENGTH,
                seed=generation * _SEED_STRIDE + index,
            )
            key = self._access.save(owner, "concepts", png, "png")
            return Concept(
                id=uuid.uuid4().hex,
                style=style_key,
                title=title,
                furniture_type=req.furniture_type,
                image_key=key,
                image_url=self._access.url(key),
            )

        concepts: list[Concept] = []
        failures: list[AppError] = []
        with ThreadPoolExecutor(max_workers=len(styles)) as pool:
            futures = [pool.submit(render, i) for i in range(len(styles))]
            for future in futures:
                try:
                    concepts.append(future.result())
                except AppError as exc:
                    failures.append(exc)
                    log_event(logger, "concept_failed", error_code=exc.code)

        if len(concepts) < 2:  # the product promise is 2-3 distinct concepts
            raise failures[0] if failures else UpstreamError("Concepts could not be generated.")

        # Persist: new type/generation, keeping the slab (the session is never reset).
        self._profiles.put_session(
            owner,
            self._profiles.get_session(owner).model_copy(
                update={"furniture_type": req.furniture_type, "concept_generation": generation}
            ),
        )
        return ConceptResponse(
            furniture_type=req.furniture_type, generation=generation, concepts=concepts
        )
