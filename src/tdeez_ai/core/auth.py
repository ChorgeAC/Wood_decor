"""Cognito JWT verification. Identity comes only from verified claims, never from the client."""

import re
from dataclasses import dataclass
from typing import Protocol, cast

import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

from tdeez_ai.core.errors import Unauthorized

_SIGN_IN_AGAIN = "Please sign in again."


@dataclass(frozen=True)
class Principal:
    sub: str
    email: str | None = None


class TokenVerifier(Protocol):
    def verify(self, token: str) -> Principal: ...


class KeyResolver(Protocol):
    def resolve(self, token: str) -> RSAPublicKey: ...


class JwksKeyResolver:
    """Fetches (and caches) the user pool's signing keys."""

    def __init__(self, jwks_url: str) -> None:
        self._client = jwt.PyJWKClient(jwks_url, cache_keys=True)

    def resolve(self, token: str) -> RSAPublicKey:
        return cast(RSAPublicKey, self._client.get_signing_key_from_jwt(token).key)


class CognitoJwtVerifier:
    def __init__(
        self,
        region: str,
        user_pool_id: str,
        app_client_id: str,
        keys: KeyResolver | None = None,
    ) -> None:
        self._issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
        self._client_id = app_client_id
        self._keys = keys or JwksKeyResolver(f"{self._issuer}/.well-known/jwks.json")

    def verify(self, token: str) -> Principal:
        try:
            key = self._keys.resolve(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self._issuer,
                options={"require": ["exp", "iss", "sub"], "verify_aud": False},
            )
        except (jwt.PyJWTError, ValueError) as exc:
            raise Unauthorized(_SIGN_IN_AGAIN) from exc

        use = claims.get("token_use")
        # Access tokens carry client_id; ID tokens carry aud.
        audience = claims.get("client_id") if use == "access" else claims.get("aud")
        if use not in {"access", "id"} or audience != self._client_id:
            raise Unauthorized(_SIGN_IN_AGAIN)
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub:
            raise Unauthorized(_SIGN_IN_AGAIN)
        email = claims.get("email")
        return Principal(sub=sub, email=email if isinstance(email, str) else None)


class MockTokenVerifier:
    """Mock mode only: ``Bearer mock-<name>`` authenticates as user ``mock-<name>``."""

    def verify(self, token: str) -> Principal:
        if not re.fullmatch(r"mock-[A-Za-z0-9_-]{1,58}", token):
            raise Unauthorized(_SIGN_IN_AGAIN)
        return Principal(sub=token)
