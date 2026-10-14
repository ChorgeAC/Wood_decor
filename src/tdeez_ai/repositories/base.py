"""Interfaces for every external dependency (DynamoDB, S3, OpenSearch, SES).

All per-customer methods take the owner id (the verified JWT ``sub``) as an explicit argument.
"""

from dataclasses import dataclass
from typing import Protocol

from tdeez_ai.models.common import Availability, Species
from tdeez_ai.models.creation import Creation
from tdeez_ai.models.profile import Profile
from tdeez_ai.models.quote import QuoteRecord
from tdeez_ai.models.slab import SessionMeta, Slab


class CreationRepository(Protocol):
    def put(self, owner: str, creation: Creation) -> None: ...

    def get(self, owner: str, creation_id: str) -> Creation | None: ...

    def list(self, owner: str) -> list[Creation]: ...

    def delete(self, owner: str, creation_id: str) -> bool: ...

    def get_by_share_token(self, token: str) -> Creation | None: ...


class SlabRepository(Protocol):
    def put(self, slab: Slab) -> None: ...

    def get(self, slab_id: str) -> Slab | None: ...

    def list(
        self, species: Species | None = None, availability: Availability | None = None
    ) -> list[Slab]: ...


class ProfileRepository(Protocol):
    def get_profile(self, owner: str) -> Profile | None: ...

    def put_profile(self, owner: str, profile: Profile) -> None: ...

    def get_session(self, owner: str) -> SessionMeta: ...

    def put_session(self, owner: str, session: SessionMeta) -> None: ...


class QuoteRepository(Protocol):
    def put(self, owner: str, record: QuoteRecord) -> None: ...

    def get(self, owner: str, reference: str) -> QuoteRecord | None: ...


@dataclass(frozen=True)
class PresignedPut:
    url: str
    headers: dict[str, str]


class ImageStore(Protocol):
    def presign_put(self, key: str, content_type: str, expires_s: int) -> PresignedPut: ...

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None: ...

    def get_bytes(self, key: str, max_bytes: int) -> bytes:
        """Raises ``NotFound`` if absent and ``PayloadTooLarge`` if over ``max_bytes``."""
        ...

    def delete(self, key: str) -> None: ...

    def presign_get(self, key: str, expires_s: int) -> str: ...


@dataclass(frozen=True)
class IndexHit:
    slab_id: str
    score: float  # cosine similarity, clamped to [0, 1]


class SimilarityIndex(Protocol):
    def upsert(self, slab_id: str, vector: list[float]) -> None: ...

    def query(self, vector: list[float], k: int) -> list[IndexHit]: ...


class Mailer(Protocol):
    def send(self, to: str, subject: str, body: str) -> None: ...
