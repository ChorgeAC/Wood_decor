"""OpenSearch Serverless k-NN similarity index.

Vectors are L2-normalised and indexed with the faiss/hnsw ``l2`` space. For unit vectors,
cosine = 1 - d^2 / 2, and OpenSearch's l2 score is 1 / (1 + d^2), so we invert it to get cosine.
"""

from collections.abc import Mapping
from typing import Protocol

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from tdeez_ai.core.errors import UpstreamError
from tdeez_ai.repositories.base import IndexHit


class OpenSearchApi(Protocol):
    def index(self, **kwargs: object) -> Mapping[str, object]: ...

    def search(self, **kwargs: object) -> Mapping[str, object]: ...


def _unit(vector: list[float]) -> list[float]:
    norm = sum(v * v for v in vector) ** 0.5 or 1.0
    return [v / norm for v in vector]


def index_mapping(dimensions: int) -> dict[str, object]:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "slab_id": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {"name": "hnsw", "engine": "faiss", "space_type": "l2"},
                },
            }
        },
    }


def create_client(endpoint: str, region: str) -> OpenSearch:
    credentials = boto3.Session().get_credentials()
    host = endpoint.removeprefix("https://").rstrip("/")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=AWSV4SignerAuth(credentials, region, "aoss"),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=15,
    )


def ensure_index(client: OpenSearch, index: str, dimensions: int) -> None:
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=index_mapping(dimensions))


class OpenSearchSimilarityIndex:
    def __init__(self, client: OpenSearchApi, index: str) -> None:
        self._client = client
        self._index = index

    def upsert(self, slab_id: str, vector: list[float]) -> None:
        # Serverless vector collections do not accept custom document ids, so the slab id is a field.
        self._client.index(index=self._index, body={"slab_id": slab_id, "embedding": _unit(vector)})

    def query(self, vector: list[float], k: int) -> list[IndexHit]:
        response = self._client.search(
            index=self._index,
            body={
                "size": k,
                "_source": ["slab_id"],
                "query": {"knn": {"embedding": {"vector": _unit(vector), "k": k}}},
            },
        )
        outer = response.get("hits")
        inner = outer.get("hits") if isinstance(outer, dict) else None
        if not isinstance(inner, list):
            raise UpstreamError("Search is temporarily unavailable. Please try again.")
        hits: list[IndexHit] = []
        for hit in inner:
            if not isinstance(hit, dict):
                continue
            source = hit.get("_source")
            score = hit.get("_score")
            slab_id = source.get("slab_id") if isinstance(source, dict) else None
            if isinstance(slab_id, str) and isinstance(score, int | float) and score > 0:
                cosine = 1.0 - (1.0 / float(score) - 1.0) / 2.0
                hits.append(IndexHit(slab_id, max(0.0, min(1.0, cosine))))
        return hits
