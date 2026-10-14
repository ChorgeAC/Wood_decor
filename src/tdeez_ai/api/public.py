"""Unauthenticated routes: health, public share, and the mock-mode upload sink."""

from fastapi import APIRouter, Request, Response

from tdeez_ai.api.deps import ContainerDep
from tdeez_ai.core.errors import NotFound, PayloadTooLarge
from tdeez_ai.core.images import content_type_for, detect_format
from tdeez_ai.models.base import ApiResponse, CamelModel, ok
from tdeez_ai.models.creation import SharedCreation
from tdeez_ai.repositories.memory import InMemoryImageStore

router = APIRouter(tags=["public"])


class Health(CamelModel):
    status: str


@router.get("/health")
def health() -> ApiResponse[Health]:
    return ok(Health(status="ok"))


@router.get("/share/{token}")
def get_shared(token: str, c: ContainerDep) -> ApiResponse[SharedCreation]:
    """Public share: read-only, no auth."""
    return ok(c.creations.get_shared(token))


@router.get("/mock-files/{key:path}")
def mock_file(key: str, c: ContainerDep) -> Response:
    """Mock mode only. Serves images behind the fake presigned GET URLs."""
    if not c.settings.use_mock or not isinstance(c.store, InMemoryImageStore):
        raise NotFound("Not found.")
    data = c.store.get_bytes(key, c.settings.max_image_bytes)
    fmt = detect_format(data)
    return Response(data, media_type=content_type_for(fmt) if fmt else "application/octet-stream")


@router.put("/mock-upload/{key:path}")
async def mock_upload(key: str, request: Request, c: ContainerDep) -> ApiResponse[None]:
    """Mock mode only. Stands in for the S3 presigned PUT; only accepts keys we issued."""
    store = c.store
    if not c.settings.use_mock or not isinstance(store, InMemoryImageStore):
        raise NotFound("Not found.")
    data = await request.body()
    if len(data) > c.settings.max_image_bytes:
        raise PayloadTooLarge("That image is too large. Please use a smaller image.")
    if not store.accept_pending_upload(key, data):
        raise NotFound("Not found.")
    return ok(None, "Uploaded")
