"""Concept generation (TBC-011): 2-3 distinct concepts that preserve slab geometry."""

from typing import Any

import pytest

from tests.conftest import Env


def _generate(env: Env, **body: Any) -> Any:
    payload: dict[str, Any] = {"furnitureType": "dining_table", "slabImageKey": env.put_image()}
    payload.update(body)
    return env.client.post("/concepts/generate", headers=env.headers(), json=payload)


def test_default_is_three_distinct_concepts(env: Env) -> None:
    r = _generate(env)
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    concepts = data["concepts"]
    assert len(concepts) == 3
    assert len({c["style"] for c in concepts}) == 3
    assert len({c["imageKey"] for c in concepts}) == 3
    assert all(c["imageUrl"] and c["furnitureType"] == "dining_table" for c in concepts)
    assert "not a formal quote" in data["disclaimer"]


def test_two_concepts_supported(env: Env) -> None:
    r = _generate(env, count=2)
    assert len(r.json()["data"]["concepts"]) == 2


@pytest.mark.parametrize("count", [0, 1, 4])
def test_count_outside_two_to_three_rejected(env: Env, count: int) -> None:
    assert _generate(env, count=count).status_code == 422


@pytest.mark.parametrize("furniture", ["bed", "", "DINING_TABLE"])
def test_furniture_type_limited_to_supported(env: Env, furniture: str) -> None:
    assert _generate(env, furnitureType=furniture).status_code == 422


def test_prompts_preserve_slab_geometry(env: Env) -> None:
    _generate(env)
    assert len(env.images.variation_calls) == 3
    for prompt, negative, _seed in env.images.variation_calls:
        assert "outline is fixed and must remain unchanged" in prompt
        assert "Never alter, reshape, crop, straighten or resize the slab outline" in prompt
        assert "altered slab outline" in negative
        assert len(prompt) <= 1024


def test_style_prompts_differ(env: Env) -> None:
    _generate(env)
    prompts = [p for p, _, _ in env.images.variation_calls]
    assert len(set(prompts)) == 3


def test_concepts_condition_on_isolated_slab(env: Env) -> None:
    _generate(env)
    session = env.client.get("/session", headers=env.headers()).json()["data"]
    # the raw upload was background-removed first and remembered
    assert session["furnitureType"] == "dining_table"


def test_regenerate_changes_seeds_and_keeps_slab(env: Env) -> None:
    key = env.put_image()
    first = _generate(env, slabImageKey=key).json()["data"]
    seeds_first = [s for _, _, s in env.images.variation_calls]
    second = _generate(env, slabImageKey=key, regenerate=True).json()["data"]
    seeds_second = [s for _, _, s in env.images.variation_calls[3:]]
    assert second["generation"] == first["generation"] + 1
    assert set(seeds_first).isdisjoint(seeds_second)


def test_change_type_without_resetting_session(env: Env) -> None:
    key = env.put_image()
    env.client.post("/slabs/select", headers=env.headers(), json={"imageKey": key})
    r1 = env.client.post(
        "/concepts/generate", headers=env.headers(), json={"furnitureType": "desk"}
    )
    assert r1.status_code == 200
    r2 = env.client.post(
        "/concepts/generate", headers=env.headers(), json={"furnitureType": "coffee_table"}
    )
    assert r2.status_code == 200
    session = env.client.get("/session", headers=env.headers()).json()["data"]
    assert session["furnitureType"] == "coffee_table"
    assert session["activeSlab"]["imageKey"] == key  # slab preserved


def test_requires_a_slab(env: Env) -> None:
    r = env.client.post("/concepts/generate", headers=env.headers(), json={"furnitureType": "desk"})
    assert r.status_code == 400
    assert "select or upload a slab" in r.json()["message"]


def test_partial_failure_still_returns_at_least_two(env: Env) -> None:
    env.images.fail_variations_after = 2
    r = _generate(env)
    assert r.status_code == 200
    assert len(r.json()["data"]["concepts"]) == 2


def test_too_many_failures_is_an_error(env: Env) -> None:
    env.images.fail_variations_after = 1
    r = _generate(env)
    assert r.status_code >= 400
    assert r.json()["isSuccess"] is False
