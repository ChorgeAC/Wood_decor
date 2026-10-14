"""Object-key helpers. Keys are always minted server-side under the caller's own prefix."""

import hashlib
import re
import uuid

from tdeez_ai.core.errors import NotFound

INVENTORY_PREFIX = "inventory/"
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_\-./]{1,300}$")


def user_prefix(sub: str) -> str:
    """Opaque per-customer prefix. Hashed so URLs (incl. public share URLs) never reveal the sub."""
    return f"users/{hashlib.sha256(sub.encode()).hexdigest()[:24]}/"


def new_key(sub: str, purpose: str, ext: str) -> str:
    return f"{user_prefix(sub)}{purpose}/{uuid.uuid4().hex}.{ext}"


def _well_formed(key: str) -> bool:
    return bool(_SAFE_KEY.match(key)) and ".." not in key


def assert_owned(sub: str, key: str) -> str:
    """Return ``key`` if it lives under the caller's prefix; otherwise behave like not-found."""
    if not _well_formed(key) or not key.startswith(user_prefix(sub)):
        raise NotFound("Image not found.")
    return key


def assert_readable(sub: str, key: str) -> str:
    """Owned keys plus the shared, read-only inventory images."""
    if _well_formed(key) and key.startswith(INVENTORY_PREFIX):
        return key
    return assert_owned(sub, key)
