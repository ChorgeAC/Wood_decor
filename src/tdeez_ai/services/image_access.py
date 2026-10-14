"""Validated, customer-scoped image reads and writes."""

from dataclasses import dataclass

from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import BadRequest
from tdeez_ai.core.images import ImageFormat, content_type_for, detect_format
from tdeez_ai.core.keys import assert_owned, assert_readable, new_key, user_prefix
from tdeez_ai.repositories.base import ImageStore


@dataclass(frozen=True)
class LoadedImage:
    key: str
    data: bytes
    fmt: ImageFormat


class ImageAccess:
    def __init__(self, store: ImageStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings

    def read(self, owner: str, key: str) -> LoadedImage:
        """Load an image the caller owns (or a shared inventory image), checking type and size."""
        assert_readable(owner, key)
        data = self._store.get_bytes(key, self._settings.max_image_bytes)
        fmt = detect_format(data)
        if fmt is None:
            if key.startswith(user_prefix(owner)):
                self._store.delete(key)
            raise BadRequest("Unsupported image. Please upload a JPEG or PNG.")
        return LoadedImage(key=key, data=data, fmt=fmt)

    def save(self, owner: str, purpose: str, data: bytes, fmt: ImageFormat = "png") -> str:
        key = new_key(owner, purpose, "png" if fmt == "png" else "jpg")
        self._store.put_bytes(key, data, content_type_for(fmt))
        return key

    def url(self, key: str) -> str:
        return self._store.presign_get(key, self._settings.read_url_expiry_s)

    def url_if_owned_or_shared(self, owner: str, key: str | None) -> str | None:
        return self.url(assert_readable(owner, key)) if key else None

    def check_owned(self, owner: str, key: str | None) -> str | None:
        return assert_owned(owner, key) if key else None
