"""Room photo: unsuitable / low-confidence / failed processing never blocks and never guesses."""

from collections.abc import Callable
from typing import Any

import pytest

from tdeez_ai.core.errors import ContentBlocked, UpstreamError
from tdeez_ai.core.keys import user_prefix
from tdeez_ai.models.room import RoomSuitability
from tests.conftest import Env, FakeVision

ALICE = user_prefix("mock-alice")
BOB = user_prefix("mock-bob")


def _room(**kw: Any) -> RoomSuitability:
    base: dict[str, Any] = {
        "suitable": True,
        "confidence": 0.9,
        "has_wall": True,
        "has_floor": True,
        "reason": "",
    }
    base.update(kw)
    return RoomSuitability(**base)


def _validate(env: Env) -> Any:
    key = env.put_image(purpose="rooms")
    r = env.client.post("/rooms/validate", headers=env.headers(), json={"imageKey": key})
    assert r.status_code == 200, r.text
    return r.json()["data"]


def _visualize(env: Env) -> Any:
    key = env.put_image(purpose="rooms")
    r = env.client.post(
        "/rooms/visualize",
        headers=env.headers(),
        json={"roomImageKey": key, "furnitureType": "desk"},
    )
    assert r.status_code == 200, r.text
    return r.json()["data"]


def test_suitable_room_is_visualized(env: Env) -> None:
    data = _visualize(env)
    assert data["fallbackToStandalone"] is False
    assert data["imageUrl"]
    assert env.images.inpaint_calls == 1


@pytest.mark.parametrize(
    "verdict",
    [
        _room(suitable=False),
        _room(has_floor=False),
        _room(has_wall=False),
        _room(confidence=0.3),
    ],
    ids=["unsuitable", "no-floor", "no-wall", "low-confidence"],
)
def test_unsuitable_room_falls_back(make_env: Callable[..., Env], verdict: RoomSuitability) -> None:
    env = make_env(vision=FakeVision(room=verdict))
    validation = _validate(env)
    assert validation["fallbackToStandalone"] is True
    assert validation["recoverable"] is True
    assert validation["suitable"] is False

    visual = _visualize(env)
    assert visual["fallbackToStandalone"] is True
    assert visual["imageUrl"] is None
    assert visual["imageKey"] is None
    assert env.images.inpaint_calls == 0  # no guessed room image


def test_vision_failure_falls_back_instead_of_erroring(make_env: Callable[..., Env]) -> None:
    env = make_env(vision=FakeVision(room=UpstreamError("down")))
    data = _validate(env)
    assert data["fallbackToStandalone"] is True
    assert "continue without a room image" in data["message"]


def test_generation_failure_falls_back(env: Env) -> None:
    env.images.inpaint_error = ContentBlocked("blocked")
    data = _visualize(env)
    assert data["fallbackToStandalone"] is True
    assert data["imageUrl"] is None


def test_min_confidence_is_configurable(make_env: Callable[..., Env]) -> None:
    env = make_env(vision=FakeVision(room=_room(confidence=0.7)), room_min_confidence=0.9)
    assert _validate(env)["fallbackToStandalone"] is True


def test_missing_room_image_is_a_clear_error(env: Env) -> None:
    r = env.client.post(
        "/rooms/validate",
        headers=env.headers(),
        json={"imageKey": ALICE + "rooms/missing.png"},
    )
    assert r.status_code == 404
    assert r.json()["isSuccess"] is False
