"""AI similarity matching (TBC-008): embed the upload, k-NN the inventory."""

from tdeez_ai.bedrock.embeddings import ImageEmbedder
from tdeez_ai.models.slab import SimilarityRequest, SimilarityResponse, SimilarSlab
from tdeez_ai.repositories.base import SimilarityIndex, SlabRepository
from tdeez_ai.services.image_access import ImageAccess
from tdeez_ai.services.slab_source import SlabSourceService


class SimilarityService:
    def __init__(
        self,
        access: ImageAccess,
        embedder: ImageEmbedder,
        index: SimilarityIndex,
        slabs: SlabRepository,
        source: SlabSourceService,
    ) -> None:
        self._access = access
        self._embedder = embedder
        self._index = index
        self._slabs = slabs
        self._source = source

    def find(self, owner: str, req: SimilarityRequest) -> SimilarityResponse:
        image = self._access.read(owner, req.image_key)
        vector = self._embedder.embed_image(image.data)
        matches: list[SimilarSlab] = []
        # Over-fetch so removed/sold slabs do not shrink the result below top_k.
        for hit in self._index.query(vector, req.top_k * 2):
            slab = self._slabs.get(hit.slab_id)
            if slab is None or slab.availability.value == "sold":
                continue
            matches.append(SimilarSlab(slab=self._source.view(slab), score=round(hit.score, 4)))
            if len(matches) == req.top_k:
                break
        return SimilarityResponse(matches=matches)
