"""Input validation: uploads, similarity, customization ranges, pricing, profile."""

from collections.abc import Callable
from typing import Any

import pytest

from tdeez_ai.bedrock.vision import RawLegChoice
from tdeez_ai.core.errors import NotFound
from tdeez_ai.core.images import make_png
from tdeez_ai.core.keys import assert_owned, assert_readable, user_prefix
from tests.conftest import Env, FakeVision, make_settings

ALICE = user_prefix("mock-alice")
BOB = user_prefix("mock-bob")


# -- uploads ---------------------------------------------------------------------------------


def _upload(env: Env, content_type: str, size: int, path: str = "/slabs/upload-url") -> Any:
    return env.client.post(
        path, headers=env.headers(), json={"contentType": content_type, "sizeBytes": size}
    )


def test_upload_url_for_valid_image(env: Env) -> None:
    r = _upload(env, "image/png", 1000)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["imageKey"].startswith(ALICE + "slabs/")
    assert data["method"] == "PUT"
    assert data["expiresInSeconds"] <= 900  # short-lived
    assert data["headers"]["Content-Type"] == "image/png"


@pytest.mark.parametrize(
    "content_type", ["image/gif", "application/pdf", "text/html", "image/webp"]
)
def test_upload_rejects_unsupported_types(env: Env, content_type: str) -> None:
    r = _upload(env, content_type, 1000)
    assert r.status_code == 422
    assert "JPEG or PNG" in r.json()["message"]


def test_upload_rejects_oversize(make_env: Callable[..., Env]) -> None:
    env = make_env(max_image_bytes=1000)
    assert _upload(env, "image/jpeg", 1001).status_code == 413
    assert _upload(env, "image/jpeg", 1000).status_code == 200


@pytest.mark.parametrize("size", [0, -5])
def test_upload_rejects_nonpositive_size(env: Env, size: int) -> None:
    assert _upload(env, "image/png", size).status_code == 422


def test_room_upload_url(env: Env) -> None:
    r = _upload(env, "image/png", 10, "/rooms/upload-url")
    assert r.json()["data"]["imageKey"].startswith(ALICE + "rooms/")


def test_mock_upload_round_trip(env: Env) -> None:
    data = _upload(env, "image/png", 100).json()["data"]
    path = data["uploadUrl"].removeprefix("http://localhost:8000")
    png = make_png(8, 8, (1, 2, 3))
    assert env.client.put(path, content=png).status_code == 200
    r = env.client.post("/slabs/select", headers=env.headers(), json={"imageKey": data["imageKey"]})
    assert r.status_code == 200
    # a key we never issued is refused
    assert (
        env.client.put("/mock-upload/" + ALICE + "slabs/other.png", content=png).status_code == 404
    )


def test_uploaded_non_image_is_rejected_and_removed(env: Env) -> None:
    key = ALICE + "slabs/fake.png"
    env.container.store.put_bytes(key, b"<html>not an image</html>", "image/png")
    r = env.client.post("/slabs/analyze", headers=env.headers(), json={"imageKey": key})
    assert r.status_code == 400
    with pytest.raises(NotFound):
        env.container.store.get_bytes(key, 10_000)


def test_stored_image_over_limit_rejected(make_env: Callable[..., Env]) -> None:
    env = make_env(max_image_bytes=50)
    key = env.put_image()
    r = env.client.post("/slabs/analyze", headers=env.headers(), json={"imageKey": key})
    assert r.status_code == 413


@pytest.mark.parametrize("key", ["../etc/passwd", BOB + "slabs/a.png", ALICE + "../x", "a b"])
def test_key_scoping_helpers(key: str) -> None:
    with pytest.raises(NotFound):
        assert_owned("mock-alice", key)
    with pytest.raises(NotFound):
        assert_readable("mock-alice", key)


def test_inventory_images_are_readable_not_owned() -> None:
    assert assert_readable("mock-alice", "inventory/slab-001.png")
    with pytest.raises(NotFound):
        assert_owned("mock-alice", "inventory/slab-001.png")


# -- slab source / similarity -----------------------------------------------------------------


def test_inventory_filters(env: Env) -> None:
    all_slabs = env.client.get("/slabs", headers=env.headers()).json()["data"]
    assert len(all_slabs) == 8
    walnut = env.client.get("/slabs?species=black_walnut", headers=env.headers()).json()["data"]
    assert {s["species"] for s in walnut} == {"black_walnut"}
    available = env.client.get("/slabs?availability=available", headers=env.headers()).json()[
        "data"
    ]
    assert all(s["availability"] == "available" for s in available)
    both = env.client.get(
        "/slabs?species=big_leaf_maple&availability=reserved", headers=env.headers()
    ).json()["data"]
    assert [s["id"] for s in both] == ["slab-004"]


def test_invalid_filter_value_rejected(env: Env) -> None:
    assert env.client.get("/slabs?species=plastic", headers=env.headers()).status_code == 422


def test_select_inventory_slab_sets_active_slab(env: Env) -> None:
    r = env.client.post("/slabs/select", headers=env.headers(), json={"slabId": "slab-003"})
    assert r.json()["data"]["activeSlab"]["slabId"] == "slab-003"
    # other users are unaffected
    other = env.client.get("/session", headers=env.headers("bob")).json()["data"]
    assert other["activeSlab"] is None


def test_cannot_select_sold_or_unknown_slab(env: Env) -> None:
    assert (
        env.client.post(
            "/slabs/select", headers=env.headers(), json={"slabId": "slab-008"}
        ).status_code
        == 422
    )
    assert (
        env.client.post("/slabs/select", headers=env.headers(), json={"slabId": "zzz"}).status_code
        == 404
    )


def test_select_needs_exactly_one_source(env: Env) -> None:
    assert env.client.post("/slabs/select", headers=env.headers(), json={}).status_code == 422
    both = {"slabId": "slab-001", "imageKey": ALICE + "slabs/a.png"}
    assert env.client.post("/slabs/select", headers=env.headers(), json=both).status_code == 422


@pytest.mark.parametrize("top_k", [2, 6, 0])
def test_similarity_top_k_range(env: Env, top_k: int) -> None:
    r = env.client.post(
        "/slabs/similar", headers=env.headers(), json={"imageKey": env.put_image(), "topK": top_k}
    )
    assert r.status_code == 422


@pytest.mark.parametrize("top_k", [3, 4, 5])
def test_similarity_returns_ranked_matches(env: Env, top_k: int) -> None:
    r = env.client.post(
        "/slabs/similar", headers=env.headers(), json={"imageKey": env.put_image(), "topK": top_k}
    )
    assert r.status_code == 200, r.text
    matches = r.json()["data"]["matches"]
    assert len(matches) == top_k
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 1 for s in scores)
    assert all(m["slab"]["availability"] != "sold" for m in matches)


def test_similarity_finds_identical_inventory_image(env: Env) -> None:
    store = env.container.store
    slab = env.container.slabs.get("slab-005")
    assert slab is not None
    data = store.get_bytes(slab.image_key, 10_000_000)
    key = ALICE + "slabs/copy.png"
    store.put_bytes(key, data, "image/png")
    r = env.client.post("/slabs/similar", headers=env.headers(), json={"imageKey": key, "topK": 3})
    top = r.json()["data"]["matches"][0]
    assert top["slab"]["id"] == "slab-005"
    assert top["score"] == pytest.approx(1.0, abs=1e-3)


# -- customization ---------------------------------------------------------------------------


def _validate(env: Env, furniture: str, customization: dict[str, Any]) -> Any:
    return env.client.post(
        "/customization/validate",
        headers=env.headers(),
        json={"furnitureType": furniture, "customization": customization},
    )


def test_valid_customization(env: Env) -> None:
    c = {"dimensions": {"lengthIn": 72, "widthIn": 36, "heightIn": 30}}
    data = _validate(env, "dining_table", c).json()["data"]
    assert data == {"valid": True, "issues": []}


@pytest.mark.parametrize(
    ("furniture", "dims", "bad_field"),
    [
        ("dining_table", {"lengthIn": 200, "widthIn": 36, "heightIn": 30}, "dimensions.lengthIn"),
        ("dining_table", {"lengthIn": 72, "widthIn": 10, "heightIn": 30}, "dimensions.widthIn"),
        ("coffee_table", {"lengthIn": 40, "widthIn": 20, "heightIn": 30}, "dimensions.heightIn"),
        ("desk", {"lengthIn": 20, "widthIn": 24, "heightIn": 29}, "dimensions.lengthIn"),
    ],
)
def test_out_of_range_dimensions(
    env: Env, furniture: str, dims: dict[str, float], bad_field: str
) -> None:
    data = _validate(env, furniture, {"dimensions": dims}).json()["data"]
    assert data["valid"] is False
    assert [i["field"] for i in data["issues"]] == [bad_field]


def test_range_edges_are_inclusive(env: Env) -> None:
    c = {"dimensions": {"lengthIn": 48, "widthIn": 48, "heightIn": 28}}
    assert _validate(env, "dining_table", c).json()["data"]["valid"] is True


def test_epoxy_rules(env: Env) -> None:
    tinted = _validate(env, "desk", {"epoxyType": "tinted"}).json()["data"]
    assert tinted["issues"][0]["field"] == "epoxyColor"
    orphan = _validate(env, "desk", {"epoxyType": "none", "epoxyColor": "#112233"}).json()["data"]
    assert orphan["valid"] is False
    ok = _validate(env, "desk", {"epoxyType": "tinted", "epoxyColor": "#112233"}).json()["data"]
    assert ok["valid"] is True


@pytest.mark.parametrize(
    "bad",
    [
        {"epoxyColor": "blue"},
        {"edgeType": "jagged"},
        {"legStyle": "wheels"},
        {"woodOrientation": "up"},
    ],
)
def test_invalid_enum_and_color_values(env: Env, bad: dict[str, Any]) -> None:
    assert _validate(env, "desk", bad).status_code == 422


def test_render_requires_valid_customization(env: Env) -> None:
    base = env.put_image(purpose="concepts")
    r = env.client.post(
        "/customization/render",
        headers=env.headers(),
        json={
            "furnitureType": "desk",
            "baseImageKey": base,
            "customization": {"dimensions": {"lengthIn": 500, "widthIn": 24, "heightIn": 29}},
        },
    )
    assert r.status_code == 422
    assert env.images.variation_calls == []


def test_render_preview_preserves_geometry_in_prompt(env: Env) -> None:
    base = env.put_image(purpose="concepts")
    r = env.client.post(
        "/customization/render",
        headers=env.headers(),
        json={
            "furnitureType": "dining_table",
            "baseImageKey": base,
            "customization": {
                "legStyle": "trestle",
                "epoxyType": "tinted",
                "epoxyColor": "#1E90FF",
            },
        },
    )
    assert r.status_code == 200, r.text
    prompt = env.images.variation_calls[0][0]
    assert "trestle" in prompt
    assert "#1E90FF" in prompt
    assert "outline is fixed and must remain unchanged" in prompt
    assert r.json()["data"]["imageUrl"]


def test_leg_recommendation(env: Env) -> None:
    r = env.client.post(
        "/customization/recommend-legs", headers=env.headers(), json={"furnitureType": "desk"}
    )
    data = r.json()["data"]
    assert data["legStyle"] == "hairpin"
    assert "not a formal quote" in data["disclaimer"]


@pytest.mark.parametrize("legs", [ValueError("x"), None], ids=["model-error", "unknown-style"])
def test_leg_recommendation_falls_back(
    make_env: Callable[..., Env], legs: Exception | None
) -> None:
    fake = FakeVision(legs=legs or RawLegChoice(leg_style="wheels", rationale="x"))
    env = make_env(vision=fake)
    r = env.client.post(
        "/customization/recommend-legs", headers=env.headers(), json={"furnitureType": "desk"}
    )
    assert r.status_code == 200
    assert r.json()["data"]["legStyle"] == "u_shaped"


# -- pricing rule ------------------------------------------------------------------------------


def _render(env: Env) -> Any:
    base = env.put_image(purpose="concepts")
    return env.client.post(
        "/customization/render",
        headers=env.headers(),
        json={"furnitureType": "desk", "baseImageKey": base, "customization": {}},
    ).json()["data"]


def test_price_range_omitted_by_default(env: Env) -> None:
    assert _render(env)["priceRange"] is None


def test_price_range_with_rules(make_env: Callable[..., Env], tmp_path: Any) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text('{"currency": "USD", "base": {"desk": 1000}, "spread_pct": 0.2}')
    env = make_env(pricing_rules_provided=True, pricing_rules_path=str(rules))
    price = _render(env)["priceRange"]
    assert (price["low"], price["high"], price["currency"]) == (800, 1200, "USD")
    assert "not a quote" in price["note"]


def test_pricing_flag_requires_path() -> None:
    with pytest.raises(ValueError, match="PRICING_RULES_PATH"):
        make_settings(pricing_rules_provided=True)


# -- profile -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"email": "nope"},
        {"phone": "abc"},
        {"extras": {f"k{i}": "v" for i in range(21)}},
        {"extras": {"k": "v" * 201}},
    ],
)
def test_profile_validation(env: Env, body: dict[str, Any]) -> None:
    assert env.client.put("/profile", headers=env.headers(), json=body).status_code == 422


def test_profile_round_trip(env: Env) -> None:
    body = {
        "name": "Alice",
        "email": "a@example.com",
        "phone": "+1 555 123 4567",
        "extras": {"k": "v"},
    }
    assert env.client.put("/profile", headers=env.headers(), json=body).status_code == 200
    assert env.client.get("/profile", headers=env.headers()).json()["data"] == body


def test_real_mode_requires_aws_settings() -> None:
    with pytest.raises(ValueError, match="S3_BUCKET"):
        make_settings(use_mock=False)


def test_mock_files_route_serves_images_in_mock_mode(env: Env) -> None:
    key = env.put_image()
    r = env.client.get(f"/mock-files/{key}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
