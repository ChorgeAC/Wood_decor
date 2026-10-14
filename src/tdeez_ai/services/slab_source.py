"""Slab source (TBC-008): uploads, inventory browsing, active slab selection."""

from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import NotFound, PayloadTooLarge, ValidationFailure
from tdeez_ai.core.images import ALLOWED_CONTENT_TYPES
from tdeez_ai.core.keys import assert_owned, new_key
from tdeez_ai.models.common import Availability, Species
from tdeez_ai.models.slab import (
    ActiveSlab,
    SessionMeta,
    Slab,
    SlabSource,
    SlabView,
    UploadUrlRequest,
    UploadUrlResponse,
)
from tdeez_ai.repositories.base import ImageStore, ProfileRepository, SlabRepository
from tdeez_ai.services.image_access import ImageAccess


class SlabSourceService:
    def __init__(
        self,
        store: ImageStore,
        slabs: SlabRepository,
        profiles: ProfileRepository,
        access: ImageAccess,
        settings: Settings,
    ) -> None:
        self._store = store
        self._slabs = slabs
        self._profiles = profiles
        self._access = access
        self._settings = settings

    def create_upload(self, owner: str, purpose: str, req: UploadUrlRequest) -> UploadUrlResponse:
        fmt = ALLOWED_CONTENT_TYPES.get(req.content_type.lower())
        if fmt is None:
            raise ValidationFailure("Please upload a JPEG or PNG image.")
        if req.size_bytes > self._settings.max_image_bytes:
            raise PayloadTooLarge("That image is too large. Please use a smaller image.")
        key = new_key(owner, purpose, "png" if fmt == "png" else "jpg")
        expires = self._settings.upload_url_expiry_s
        signed = self._store.presign_put(key, req.content_type.lower(), expires)
        return UploadUrlResponse(
            upload_url=signed.url,
            image_key=key,
            headers=signed.headers,
            expires_in_seconds=expires,
            max_bytes=self._settings.max_image_bytes,
        )

    def view(self, slab: Slab) -> SlabView:
        return SlabView(**slab.model_dump(), image_url=self._access.url(slab.image_key))

    def list_slabs(
        self, species: Species | None, availability: Availability | None
    ) -> list[SlabView]:
        return [self.view(s) for s in self._slabs.list(species, availability)]

    def get_session(self, owner: str) -> SessionMeta:
        return self._profiles.get_session(owner)

    def select(self, owner: str, source: SlabSource) -> SessionMeta:
        if source.slab_id is not None:
            slab = self._slabs.get(source.slab_id)
            if slab is None:
                raise NotFound("That slab could not be found.")
            if slab.availability is Availability.SOLD:
                raise ValidationFailure("That slab is no longer available.")
            active = ActiveSlab(source="inventory", slab_id=slab.id, image_key=slab.image_key)
        else:
            assert source.image_key is not None
            key = assert_owned(owner, source.image_key)
            self._access.read(owner, key)  # validates existence, type and size
            active = ActiveSlab(source="upload", image_key=key)
        session = self._profiles.get_session(owner).model_copy(update={"active_slab": active})
        self._profiles.put_session(owner, session)
        return session
