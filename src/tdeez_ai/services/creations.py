"""My Creations (POC-006, TBC-013) and the public share view."""

import secrets
import uuid
from datetime import UTC, datetime

from tdeez_ai.core.errors import NotFound
from tdeez_ai.core.keys import assert_owned, assert_readable
from tdeez_ai.models.creation import (
    Creation,
    CreationView,
    SaveCreationRequest,
    SharedCreation,
    ShareLink,
    UpdateCreationRequest,
)
from tdeez_ai.models.slab import SlabSource
from tdeez_ai.repositories.base import CreationRepository, SlabRepository
from tdeez_ai.services.image_access import ImageAccess

_NOT_FOUND = "That creation could not be found."


class CreationService:
    def __init__(
        self, repo: CreationRepository, slabs: SlabRepository, access: ImageAccess
    ) -> None:
        self._repo = repo
        self._slabs = slabs
        self._access = access

    # -- CRUD -----------------------------------------------------------------------------

    def save(self, owner: str, req: SaveCreationRequest) -> CreationView:
        self._check_slab(owner, req.slab)
        self._check_keys(
            owner,
            req.concept.image_key if req.concept else None,
            req.visualization_image_key,
            req.room_visualization_image_key,
        )
        now = datetime.now(UTC)
        creation = Creation(
            id=uuid.uuid4().hex,
            name=req.name,
            created_at=now,
            updated_at=now,
            slab=req.slab,
            concept=req.concept,
            customization=req.customization,
            visualization_image_key=req.visualization_image_key,
            room_visualization_image_key=req.room_visualization_image_key,
        )
        self._repo.put(owner, creation)
        return self._view(owner, creation)

    def list(self, owner: str) -> list[CreationView]:
        return [self._view(owner, c) for c in self._repo.list(owner)]

    def get(self, owner: str, creation_id: str) -> CreationView:
        return self._view(owner, self._load(owner, creation_id))

    def update(self, owner: str, creation_id: str, req: UpdateCreationRequest) -> CreationView:
        current = self._load(owner, creation_id)
        self._check_keys(
            owner,
            req.concept.image_key if req.concept else None,
            req.visualization_image_key,
            req.room_visualization_image_key,
        )
        changes = req.model_dump(exclude_unset=True, exclude_none=True)
        updated = current.model_copy(
            update={
                **{k: getattr(req, k) for k in changes},
                "updated_at": datetime.now(UTC),
            }
        )
        self._repo.put(owner, updated)
        return self._view(owner, updated)

    def delete(self, owner: str, creation_id: str) -> None:
        if not self._repo.delete(owner, creation_id):
            raise NotFound(_NOT_FOUND)

    # -- sharing --------------------------------------------------------------------------

    def enable_share(self, owner: str, creation_id: str) -> ShareLink:
        current = self._load(owner, creation_id)
        token = current.share_token or secrets.token_urlsafe(24)
        self._repo.put(owner, current.model_copy(update={"share_token": token}))
        return ShareLink(token=token, path=f"/share/{token}")

    def disable_share(self, owner: str, creation_id: str) -> None:
        current = self._load(owner, creation_id)
        self._repo.put(owner, current.model_copy(update={"share_token": None}))

    def get_shared(self, token: str) -> SharedCreation:
        creation = self._repo.get_by_share_token(token)
        if creation is None:
            raise NotFound("This shared design is not available.")
        concept = creation.concept
        return SharedCreation(
            name=creation.name,
            furniture_type=concept.furniture_type if concept else None,
            customization=creation.customization,
            concept_image_url=self._url(concept.image_key if concept else None),
            visualization_url=self._url(creation.visualization_image_key),
            room_visualization_url=self._url(creation.room_visualization_image_key),
        )

    # -- helpers --------------------------------------------------------------------------

    def _load(self, owner: str, creation_id: str) -> Creation:
        creation = self._repo.get(owner, creation_id)
        if creation is None:  # also covers other customers' ids: no existence leak
            raise NotFound(_NOT_FOUND)
        return creation

    def _check_slab(self, owner: str, slab: SlabSource) -> None:
        if slab.slab_id is not None:
            if self._slabs.get(slab.slab_id) is None:
                raise NotFound("That slab could not be found.")
        elif slab.image_key is not None:
            assert_readable(owner, slab.image_key)

    def _check_keys(self, owner: str, *keys: str | None) -> None:
        for key in keys:
            if key is not None:
                assert_owned(owner, key)

    def _url(self, key: str | None) -> str | None:
        return self._access.url(key) if key else None

    def _view(self, owner: str, c: Creation) -> CreationView:
        slab_key = c.slab.image_key
        if c.slab.slab_id is not None:
            slab = self._slabs.get(c.slab.slab_id)
            slab_key = slab.image_key if slab else None
        return CreationView(
            id=c.id,
            name=c.name,
            created_at=c.created_at,
            updated_at=c.updated_at,
            slab=c.slab,
            concept=c.concept,
            customization=c.customization,
            slab_image_url=self._url(slab_key),
            concept_image_url=self._url(c.concept.image_key if c.concept else None),
            visualization_url=self._url(c.visualization_image_key),
            room_visualization_url=self._url(c.room_visualization_image_key),
            share_token=c.share_token,
        )
