"""Response envelope: { isSuccess, message, data } for success and for every kind of error."""

from tests.conftest import Env


def test_success_envelope_shape(env: Env) -> None:
    r = env.client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"isSuccess": True, "message": "OK", "data": {"status": "ok"}}


def test_success_payload_is_camel_case(env: Env) -> None:
    r = env.client.get("/slabs", headers=env.headers())
    body = r.json()
    assert body["isSuccess"] is True
    first = body["data"][0]
    assert {"imageUrl", "imageKey"} <= first.keys()
    assert "image_url" not in first


def test_app_error_uses_envelope(env: Env) -> None:
    r = env.client.get("/creations/does-not-exist", headers=env.headers())
    assert r.status_code == 404
    body = r.json()
    assert body == {
        "isSuccess": False,
        "message": "That creation could not be found.",
        "data": None,
    }


def test_unknown_route_uses_envelope(env: Env) -> None:
    r = env.client.get("/nope")
    assert r.status_code == 404
    assert r.json()["isSuccess"] is False


def test_validation_error_hides_input_values(env: Env) -> None:
    secret = "do-not-echo-this@example.com"
    r = env.client.put("/profile", headers=env.headers(), json={"email": secret + "x y"})
    assert r.status_code == 422
    body = r.json()
    assert body["isSuccess"] is False
    assert body["data"] is None
    assert secret not in r.text
    assert "email" in body["message"]


def test_unhandled_exception_is_generic(env: Env) -> None:
    def boom(*_: object, **__: object) -> None:
        raise RuntimeError("contains alice@example.com")

    env.container.profiles.get = boom  # type: ignore[method-assign,assignment]
    r = env.client.get("/profile", headers=env.headers())
    assert r.status_code == 500
    assert r.json() == {
        "isSuccess": False,
        "message": "Something went wrong. Please try again.",
        "data": None,
    }
    assert "alice@example.com" not in r.text
