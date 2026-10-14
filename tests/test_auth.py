"""Auth: Cognito JWT verification and customer scoping."""

import time
from dataclasses import dataclass

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

from tdeez_ai.core.auth import CognitoJwtVerifier
from tdeez_ai.core.errors import Unauthorized
from tests.conftest import Env

REGION, POOL, CLIENT = "us-east-1", "us-east-1_TEST", "client-123"
ISSUER = f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}"


@dataclass
class StaticKeys:
    key: RSAPublicKey

    def resolve(self, token: str) -> RSAPublicKey:
        return self.key


@pytest.fixture(scope="module")
def private_key() -> RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(key: RSAPrivateKey, **overrides: object) -> str:
    now = int(time.time())
    claims: dict[str, object] = {
        "sub": "user-1",
        "iss": ISSUER,
        "token_use": "access",
        "client_id": CLIENT,
        "exp": now + 300,
        "iat": now,
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256")


def _verifier(key: RSAPrivateKey) -> CognitoJwtVerifier:
    return CognitoJwtVerifier(REGION, POOL, CLIENT, keys=StaticKeys(key.public_key()))


def test_valid_access_token(private_key: RSAPrivateKey) -> None:
    principal = _verifier(private_key).verify(_token(private_key))
    assert principal.sub == "user-1"


def test_valid_id_token_carries_email(private_key: RSAPrivateKey) -> None:
    token = _token(private_key, token_use="id", aud=CLIENT, client_id=None, email="a@example.com")
    principal = _verifier(private_key).verify(token)
    assert principal.email == "a@example.com"


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": int(time.time()) - 10},
        {"iss": "https://evil.example.com"},
        {"client_id": "other-client"},
        {"token_use": "refresh"},
        {"sub": None},
    ],
)
def test_bad_claims_rejected(private_key: RSAPrivateKey, overrides: dict[str, object]) -> None:
    with pytest.raises(Unauthorized):
        _verifier(private_key).verify(_token(private_key, **overrides))


def test_wrong_signing_key_rejected(private_key: RSAPrivateKey) -> None:
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(Unauthorized):
        _verifier(private_key).verify(_token(other))


def test_garbage_token_rejected(private_key: RSAPrivateKey) -> None:
    with pytest.raises(Unauthorized):
        _verifier(private_key).verify("not-a-jwt")


def test_missing_token_is_401(env: Env) -> None:
    r = env.client.get("/creations")
    assert r.status_code == 401
    assert r.json()["isSuccess"] is False


def test_invalid_token_is_401(env: Env) -> None:
    r = env.client.get("/creations", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_share_route_is_public(env: Env) -> None:
    r = env.client.get("/share/unknown-token")
    assert r.status_code == 404  # reached the handler without auth
    assert r.json()["message"] == "This shared design is not available."


# -- customer scoping ---------------------------------------------------------------------


def _save(env: Env, user: str) -> str:
    key = env.put_image(user, "renders")
    r = env.client.post(
        "/creations",
        headers=env.headers(user),
        json={
            "name": "My table",
            "slab": {"slabId": "slab-001"},
            "customization": {},
            "visualizationImageKey": key,
        },
    )
    assert r.status_code == 200, r.text
    return str(r.json()["data"]["id"])


def test_other_customers_creation_is_invisible(env: Env) -> None:
    creation_id = _save(env, "alice")
    assert (
        env.client.get(f"/creations/{creation_id}", headers=env.headers("alice")).status_code == 200
    )
    assert (
        env.client.get(f"/creations/{creation_id}", headers=env.headers("bob")).status_code == 404
    )
    assert (
        env.client.delete(f"/creations/{creation_id}", headers=env.headers("bob")).status_code
        == 404
    )
    assert env.client.get("/creations", headers=env.headers("bob")).json()["data"] == []
    # still intact for the owner
    assert (
        env.client.get(f"/creations/{creation_id}", headers=env.headers("alice")).status_code == 200
    )


def test_client_supplied_user_id_is_ignored(env: Env) -> None:
    key = env.put_image("alice", "renders")
    env.client.post(
        "/creations",
        headers=env.headers("alice"),
        json={
            "name": "x",
            "slab": {"slabId": "slab-001"},
            "customization": {},
            "visualizationImageKey": key,
            "userId": "mock-bob",
            "sub": "mock-bob",
        },
    )
    assert len(env.client.get("/creations", headers=env.headers("alice")).json()["data"]) == 1
    assert env.client.get("/creations", headers=env.headers("bob")).json()["data"] == []


def test_cannot_reference_another_customers_image(env: Env) -> None:
    bobs_key = env.put_image("bob", "renders")
    r = env.client.post(
        "/creations",
        headers=env.headers("alice"),
        json={
            "name": "x",
            "slab": {"slabId": "slab-001"},
            "customization": {},
            "visualizationImageKey": bobs_key,
        },
    )
    assert r.status_code == 404
    r = env.client.post("/slabs/analyze", headers=env.headers("alice"), json={"imageKey": bobs_key})
    assert r.status_code == 404


def test_profiles_are_per_user(env: Env) -> None:
    env.client.put("/profile", headers=env.headers("alice"), json={"name": "Alice"})
    assert (
        env.client.get("/profile", headers=env.headers("alice")).json()["data"]["name"] == "Alice"
    )
    assert env.client.get("/profile", headers=env.headers("bob")).json()["data"]["name"] is None


def test_share_link_exposes_no_owner_data(env: Env) -> None:
    creation_id = _save(env, "alice")
    link = env.client.post(f"/creations/{creation_id}/share", headers=env.headers("alice"))
    token = link.json()["data"]["token"]
    public = env.client.get(f"/share/{token}")
    assert public.status_code == 200
    text = public.text
    assert "alice" not in text
    assert "mock-alice" not in text  # the sub never appears in shared URLs
    assert public.json()["data"]["name"] == "My table"
    # revoke
    env.client.delete(f"/creations/{creation_id}/share", headers=env.headers("alice"))
    assert env.client.get(f"/share/{token}").status_code == 404
