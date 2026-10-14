"""Shared fixtures. Everything runs offline: fakes stand in for Bedrock, AWS is never called."""

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from tdeez_ai.app import create_app
from tdeez_ai.bedrock.mock import MockImageGenerator, MockVisionAnalyzer
from tdeez_ai.bedrock.vision import RawLegChoice
from tdeez_ai.container import Container, build_container
from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import AppError
from tdeez_ai.core.images import ImageFormat, make_png
from tdeez_ai.core.keys import new_key
from tdeez_ai.models.common import FurnitureType
from tdeez_ai.models.room import RoomSuitability
from tdeez_ai.models.slab import SlabVisionResult
from tdeez_ai.repositories.memory import InMemoryMailer


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"use_mock": True, "seed_on_startup": True}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


@dataclass
class FakeVision:
    """Scriptable multimodal model."""

    species: str | None = "black_walnut"
    confidence: float = 0.9
    summary: str = "A warm slab with a flowing natural edge."
    room: RoomSuitability | AppError = field(
        default_factory=lambda: RoomSuitability(
            suitable=True, confidence=0.9, has_wall=True, has_floor=True, reason="ok"
        )
    )
    legs: RawLegChoice | Exception = field(
        default_factory=lambda: RawLegChoice(leg_style="hairpin", rationale="Light and airy.")
    )

    def analyze_slab(self, image: bytes, fmt: ImageFormat) -> SlabVisionResult:
        return SlabVisionResult(
            shape_description="Elongated slab with a curving live edge.",
            length_to_width_ratio=2.4,
            defects=[],
            species=self.species,
            species_confidence=self.confidence,
            summary=self.summary,
        )

    def assess_room(self, image: bytes, fmt: ImageFormat) -> RoomSuitability:
        if isinstance(self.room, AppError):
            raise self.room
        return self.room

    def recommend_legs(
        self, furniture: FurnitureType, ratio: float | None, summary: str | None
    ) -> RawLegChoice:
        if isinstance(self.legs, Exception):
            raise self.legs
        return self.legs


@dataclass
class RecordingImages(MockImageGenerator):
    """Mock generator that records every call and can be told to fail."""

    variation_calls: list[tuple[str, str, int]] = field(default_factory=list)
    inpaint_calls: int = 0
    fail_variations_after: int | None = None
    inpaint_error: AppError | None = None

    def variation(
        self, source: bytes, prompt: str, negative: str, *, similarity: float, seed: int
    ) -> bytes:
        if (
            self.fail_variations_after is not None
            and len(self.variation_calls) >= self.fail_variations_after
        ):
            raise AppError("boom")
        self.variation_calls.append((prompt, negative, seed))
        return super().variation(source, prompt, negative, similarity=similarity, seed=seed)

    def inpaint(
        self, image: bytes, mask_prompt: str, prompt: str, negative: str, *, seed: int
    ) -> bytes:
        self.inpaint_calls += 1
        if self.inpaint_error is not None:
            raise self.inpaint_error
        return super().inpaint(image, mask_prompt, prompt, negative, seed=seed)


@dataclass
class Env:
    client: TestClient
    container: Container
    mailer: InMemoryMailer
    images: RecordingImages
    vision: FakeVision

    def headers(self, user: str = "alice") -> dict[str, str]:
        return {"Authorization": f"Bearer mock-{user}"}

    def put_image(self, user: str = "alice", purpose: str = "slabs") -> str:
        """Place a valid image in the store under the user's prefix (simulates the S3 upload)."""
        key = new_key(f"mock-{user}", purpose, "png")
        self.container.store.put_bytes(key, make_png(8, 8, (120, 80, 40)), "image/png")
        return key


@pytest.fixture
def make_env() -> Callable[..., Env]:
    def factory(
        *,
        vision: FakeVision | None = None,
        mailer: InMemoryMailer | None = None,
        **settings: object,
    ) -> Env:
        fake_vision = vision or FakeVision()
        mail = mailer or InMemoryMailer()
        images = RecordingImages()
        container = build_container(
            make_settings(**settings), vision=fake_vision, images=images, mailer=mail
        )
        app = create_app(container.settings, container)
        return Env(
            TestClient(app, raise_server_exceptions=False), container, mail, images, fake_vision
        )

    return factory


@pytest.fixture
def env(make_env: Callable[..., Env]) -> Env:
    return make_env()


def default_mock_vision() -> MockVisionAnalyzer:
    return MockVisionAnalyzer()
