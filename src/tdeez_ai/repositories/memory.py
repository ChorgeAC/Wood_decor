"""In-memory implementations (USE_MOCK=true and tests). Process-local, not persistent."""

import math
from dataclasses import dataclass

from tdeez_ai.core.errors import NotFound, PayloadTooLarge
from tdeez_ai.models.common import Availability, Species
from tdeez_ai.models.creation import Creation
from tdeez_ai.models.profile import Profile
from tdeez_ai.models.quote import QuoteRecord
from tdeez_ai.models.slab import SessionMeta, Slab
from tdeez_ai.repositories.base import IndexHit, PresignedPut


class InMemoryCreationRepository:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], Creation] = {}

    def put(self, owner: str, creation: Creation) -> None:
        self._items[(owner, creation.id)] = creation.model_copy(deep=True)

    def get(self, owner: str, creation_id: str) -> Creation | None:
        item = self._items.get((owner, creation_id))
        return item.model_copy(deep=True) if item else None

    def list(self, owner: str) -> list[Creation]:
        items = [c for (o, _), c in self._items.items() if o == owner]
        return sorted(
            (c.model_copy(deep=True) for c in items), key=lambda c: c.updated_at, reverse=True
        )

    def delete(self, owner: str, creation_id: str) -> bool:
        return self._items.pop((owner, creation_id), None) is not None

    def get_by_share_token(self, token: str) -> Creation | None:
        for creation in self._items.values():
            if creation.share_token == token:
                return creation.model_copy(deep=True)
        return None


class InMemorySlabRepository:
    def __init__(self) -> None:
        self._items: dict[str, Slab] = {}

    def put(self, slab: Slab) -> None:
        self._items[slab.id] = slab

    def get(self, slab_id: str) -> Slab | None:
        return self._items.get(slab_id)

    def list(
        self, species: Species | None = None, availability: Availability | None = None
    ) -> list[Slab]:
        return sorted(
            (
                s
                for s in self._items.values()
                if (species is None or s.species == species)
                and (availability is None or s.availability == availability)
            ),
            key=lambda s: s.id,
        )


class InMemoryProfileRepository:
    def __init__(self) -> None:
        self._profiles: dict[str, Profile] = {}
        self._sessions: dict[str, SessionMeta] = {}

    def get_profile(self, owner: str) -> Profile | None:
        profile = self._profiles.get(owner)
        return profile.model_copy(deep=True) if profile else None

    def put_profile(self, owner: str, profile: Profile) -> None:
        self._profiles[owner] = profile.model_copy(deep=True)

    def get_session(self, owner: str) -> SessionMeta:
        return self._sessions.get(owner, SessionMeta()).model_copy(deep=True)

    def put_session(self, owner: str, session: SessionMeta) -> None:
        self._sessions[owner] = session.model_copy(deep=True)


class InMemoryQuoteRepository:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], QuoteRecord] = {}

    def put(self, owner: str, record: QuoteRecord) -> None:
        self._items[(owner, record.reference)] = record

    def get(self, owner: str, reference: str) -> QuoteRecord | None:
        return self._items.get((owner, reference))


class InMemoryImageStore:
    """Stores bytes in a dict. ``presign_put`` points at the dev-only /mock-upload route."""

    def __init__(self, public_base_url: str = "http://localhost:8000") -> None:
        self._base = public_base_url.rstrip("/")
        self._blobs: dict[str, bytes] = {}
        self._pending: set[str] = set()

    def presign_put(self, key: str, content_type: str, expires_s: int) -> PresignedPut:
        self._pending.add(key)
        return PresignedPut(
            url=f"{self._base}/mock-upload/{key}", headers={"Content-Type": content_type}
        )

    def accept_pending_upload(self, key: str, data: bytes) -> bool:
        """Used by the dev upload route: only keys we issued a URL for are accepted."""
        if key not in self._pending:
            return False
        self._pending.discard(key)
        self._blobs[key] = data
        return True

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self._blobs[key] = data

    def get_bytes(self, key: str, max_bytes: int) -> bytes:
        data = self._blobs.get(key)
        if data is None:
            raise NotFound("Image not found. Please upload it again.")
        if len(data) > max_bytes:
            raise PayloadTooLarge("That image is too large. Please use a smaller image.")
        return data

    def delete(self, key: str) -> None:
        self._blobs.pop(key, None)

    def presign_get(self, key: str, expires_s: int) -> str:
        return f"{self._base}/mock-files/{key}?expires={expires_s}"


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemorySimilarityIndex:
    """Brute-force cosine search; the mock-mode stand-in for OpenSearch k-NN."""

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, slab_id: str, vector: list[float]) -> None:
        self._vectors[slab_id] = list(vector)

    def query(self, vector: list[float], k: int) -> list[IndexHit]:
        scored = [
            IndexHit(sid, max(0.0, min(1.0, cosine(vector, vec))))
            for sid, vec in self._vectors.items()
            if len(vec) == len(vector)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


@dataclass(frozen=True)
class SentEmail:
    to: str
    subject: str
    body: str


class InMemoryMailer:
    def __init__(self) -> None:
        self.outbox: list[SentEmail] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.outbox.append(SentEmail(to, subject, body))
