from typing import Annotated

from fastapi import APIRouter, Query

from tdeez_ai.api.deps import ContainerDep, UserDep
from tdeez_ai.models.base import ApiResponse, ok
from tdeez_ai.models.common import Availability, Species
from tdeez_ai.models.slab import (
    SessionMeta,
    SimilarityRequest,
    SimilarityResponse,
    SlabAnalysis,
    SlabSource,
    SlabView,
    UploadUrlRequest,
    UploadUrlResponse,
)

router = APIRouter(tags=["slabs"])


@router.post("/slabs/upload-url")
def slab_upload_url(
    body: UploadUrlRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[UploadUrlResponse]:
    """TBC-008: presigned PUT for a slab / reference image."""
    return ok(c.slab_source.create_upload(user.sub, "slabs", body))


@router.get("/slabs")
def list_slabs(
    user: UserDep,
    c: ContainerDep,
    species: Annotated[Species | None, Query()] = None,
    availability: Annotated[Availability | None, Query()] = None,
) -> ApiResponse[list[SlabView]]:
    """TBC-008: inventory browse with filters."""
    return ok(c.slab_source.list_slabs(species, availability))


@router.post("/slabs/select")
def select_slab(body: SlabSource, user: UserDep, c: ContainerDep) -> ApiResponse[SessionMeta]:
    """TBC-008: store the active slab in session metadata."""
    return ok(c.slab_source.select(user.sub, body), "Slab selected")


@router.get("/session")
def get_session(user: UserDep, c: ContainerDep) -> ApiResponse[SessionMeta]:
    """TBC-008 / TBC-011"""
    return ok(c.slab_source.get_session(user.sub))


@router.post("/slabs/similar")
def similar_slabs(
    body: SimilarityRequest, user: UserDep, c: ContainerDep
) -> ApiResponse[SimilarityResponse]:
    """TBC-008: Top 3-5 similar inventory slabs with scores."""
    return ok(c.similarity.find(user.sub, body))


@router.post("/slabs/analyze")
def analyze_slab(body: SlabSource, user: UserDep, c: ContainerDep) -> ApiResponse[SlabAnalysis]:
    """TBC-010"""
    return ok(c.analysis.analyze(user.sub, body))
