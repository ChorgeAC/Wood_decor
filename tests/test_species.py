"""Slab analysis: species confidence threshold, disclaimer and no-exact-dimension rules."""

from collections.abc import Callable
from typing import Any

import pytest

from tdeez_ai.core.keys import user_prefix
from tdeez_ai.services.text_guard import guard_text
from tests.conftest import Env, FakeVision

ALICE = user_prefix("mock-alice")
BOB = user_prefix("mock-bob")


def _analyze(env: Env, **body: str) -> Any:
    payload = body or {"imageKey": env.put_image()}
    r = env.client.post("/slabs/analyze", headers=env.headers(), json=payload)
    assert r.status_code == 200, r.text
    return r.json()["data"]


def test_high_confidence_is_not_flagged(env: Env) -> None:
    data = _analyze(env)
    assert data["needsReview"] is False
    assert data["reviewMessage"] is None
    assert data["species"]["species"] == "black_walnut"


@pytest.mark.parametrize(("confidence", "flagged"), [(0.59, True), (0.6, False), (0.1, True)])
def test_threshold_boundary(make_env: Callable[..., Env], confidence: float, flagged: bool) -> None:
    env = make_env(vision=FakeVision(confidence=confidence))
    data = _analyze(env)
    assert data["needsReview"] is flagged
    if flagged:
        assert "TDeez will review" in str(data["reviewMessage"])


def test_threshold_is_configurable(make_env: Callable[..., Env]) -> None:
    env = make_env(vision=FakeVision(confidence=0.7), species_review_threshold=0.8)
    assert _analyze(env)["needsReview"] is True


def test_unsupported_species_needs_review(make_env: Callable[..., Env]) -> None:
    env = make_env(vision=FakeVision(species="teak", confidence=0.99))
    data = _analyze(env)
    assert data["needsReview"] is True
    assert data["species"]["species"] is None


def test_response_has_disclaimer_and_limitations(env: Env) -> None:
    data = _analyze(env)
    assert "not a formal quote" in str(data["disclaimer"])
    assert "not an exact measurement" in str(data["limitations"])


def test_no_exact_dimensions_for_uploads(env: Env) -> None:
    data = _analyze(env)
    proportions = data["proportions"]
    assert proportions["basis"] == "image_estimate"
    assert proportions["knownDimensions"] is None
    assert "Exact dimensions are not known" in proportions["note"]


def test_inventory_slab_uses_listing_dimensions(env: Env) -> None:
    data = _analyze(env, slabId="slab-001")
    proportions = data["proportions"]
    assert proportions["basis"] == "inventory_listing"
    assert proportions["knownDimensions"]["lengthIn"] == 96
    assert "subject to confirmation" in proportions["note"]


def test_model_text_is_scrubbed_of_measurements_and_claims(make_env: Callable[..., Env]) -> None:
    env = make_env(
        vision=FakeVision(
            summary="This 84 inches slab is structurally sound and worth $2,400 at 30cm wide."
        )
    )
    summary = str(_analyze(env)["summary"])
    for leaked in ("84", "30cm", "structurally sound", "$2,400"):
        assert leaked not in summary


def test_guard_text_leaves_plain_text_alone() -> None:
    text = "A warm slab with two small knots and a flowing edge."
    assert guard_text(text) == text


def test_analysis_stores_isolated_slab_in_session(env: Env) -> None:
    _analyze(env)
    session = env.client.get("/session", headers=env.headers()).json()["data"]
    assert session["activeSlab"]["isolatedImageKey"].startswith(ALICE + "derived/")
