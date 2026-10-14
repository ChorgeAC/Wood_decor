"""DynamoDB repositories.

Key design (one table per domain, string keys):
  creations : pk=USER#<sub>  sk=CREATION#<id>   GSI shareToken-index (sparse) hash=shareToken
  slabs     : pk=SLAB#<id>   sk=META            GSI species-availability-index hash=species range=availability
  profiles  : pk=USER#<sub>  sk=PROFILE | SESSION
  quotes    : pk=USER#<sub>  sk=QUOTE#<reference>
The domain object is stored as a JSON string in ``data`` so floats never hit DynamoDB's Decimal rules.
"""

from collections.abc import Mapping
from typing import Protocol, TypeVar

from boto3.dynamodb.conditions import Attr, Key
from pydantic import BaseModel

from tdeez_ai.models.common import Availability, Species
from tdeez_ai.models.creation import Creation
from tdeez_ai.models.profile import Profile
from tdeez_ai.models.quote import QuoteRecord
from tdeez_ai.models.slab import SessionMeta, Slab

M = TypeVar("M", bound=BaseModel)

type Item = Mapping[str, object]


class DynamoTable(Protocol):
    """The subset of ``boto3.resource('dynamodb').Table`` we use."""

    def put_item(self, **kwargs: object) -> Item: ...

    def get_item(self, **kwargs: object) -> Item: ...

    def delete_item(self, **kwargs: object) -> Item: ...

    def query(self, **kwargs: object) -> Item: ...

    def scan(self, **kwargs: object) -> Item: ...


def _items(response: Item) -> list[Item]:
    raw = response.get("Items")
    return [i for i in raw if isinstance(i, dict)] if isinstance(raw, list) else []


def _decode(item: Item | None, model: type[M]) -> M | None:
    if not item:
        return None
    data = item.get("data")
    return model.model_validate_json(data) if isinstance(data, str) else None


def _encode(model: BaseModel) -> str:
    return model.model_dump_json(by_alias=True)


def _get(table: DynamoTable, pk: str, sk: str) -> Item | None:
    item = table.get_item(Key={"pk": pk, "sk": sk}).get("Item")
    return item if isinstance(item, dict) else None


class DynamoCreationRepository:
    def __init__(self, table: DynamoTable) -> None:
        self._t = table

    def put(self, owner: str, creation: Creation) -> None:
        item: dict[str, object] = {
            "pk": f"USER#{owner}",
            "sk": f"CREATION#{creation.id}",
            "data": _encode(creation),
            "updatedAt": creation.updated_at.isoformat(),
        }
        if creation.share_token:
            item["shareToken"] = creation.share_token
        self._t.put_item(Item=item)

    def get(self, owner: str, creation_id: str) -> Creation | None:
        return _decode(_get(self._t, f"USER#{owner}", f"CREATION#{creation_id}"), Creation)

    def list(self, owner: str) -> list[Creation]:
        response = self._t.query(
            KeyConditionExpression=Key("pk").eq(f"USER#{owner}")
            & Key("sk").begins_with("CREATION#")
        )
        found = [c for i in _items(response) if (c := _decode(i, Creation))]
        return sorted(found, key=lambda c: c.updated_at, reverse=True)

    def delete(self, owner: str, creation_id: str) -> bool:
        existed = self.get(owner, creation_id) is not None
        self._t.delete_item(Key={"pk": f"USER#{owner}", "sk": f"CREATION#{creation_id}"})
        return existed

    def get_by_share_token(self, token: str) -> Creation | None:
        response = self._t.query(
            IndexName="shareToken-index",
            KeyConditionExpression=Key("shareToken").eq(token),
            Limit=1,
        )
        items = _items(response)
        return _decode(items[0], Creation) if items else None


class DynamoSlabRepository:
    def __init__(self, table: DynamoTable) -> None:
        self._t = table

    def put(self, slab: Slab) -> None:
        self._t.put_item(
            Item={
                "pk": f"SLAB#{slab.id}",
                "sk": "META",
                "species": slab.species.value,
                "availability": slab.availability.value,
                "data": _encode(slab),
            }
        )

    def get(self, slab_id: str) -> Slab | None:
        return _decode(_get(self._t, f"SLAB#{slab_id}", "META"), Slab)

    def list(
        self, species: Species | None = None, availability: Availability | None = None
    ) -> list[Slab]:
        items: list[Item] = []
        kwargs: dict[str, object]
        if species is not None:
            cond = Key("species").eq(species.value)
            if availability is not None:
                cond = cond & Key("availability").eq(availability.value)
            kwargs = {"IndexName": "species-availability-index", "KeyConditionExpression": cond}
            method = self._t.query
        else:
            kwargs = {"FilterExpression": Attr("sk").eq("META")}
            if availability is not None:
                kwargs["FilterExpression"] = Attr("sk").eq("META") & Attr("availability").eq(
                    availability.value
                )
            method = self._t.scan
        while True:
            response = method(**kwargs)
            items.extend(_items(response))
            last = response.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
        return sorted((s for i in items if (s := _decode(i, Slab))), key=lambda s: s.id)


class DynamoProfileRepository:
    def __init__(self, table: DynamoTable) -> None:
        self._t = table

    def get_profile(self, owner: str) -> Profile | None:
        return _decode(_get(self._t, f"USER#{owner}", "PROFILE"), Profile)

    def put_profile(self, owner: str, profile: Profile) -> None:
        self._t.put_item(Item={"pk": f"USER#{owner}", "sk": "PROFILE", "data": _encode(profile)})

    def get_session(self, owner: str) -> SessionMeta:
        return _decode(_get(self._t, f"USER#{owner}", "SESSION"), SessionMeta) or SessionMeta()

    def put_session(self, owner: str, session: SessionMeta) -> None:
        self._t.put_item(Item={"pk": f"USER#{owner}", "sk": "SESSION", "data": _encode(session)})


class DynamoQuoteRepository:
    def __init__(self, table: DynamoTable) -> None:
        self._t = table

    def put(self, owner: str, record: QuoteRecord) -> None:
        self._t.put_item(
            Item={"pk": f"USER#{owner}", "sk": f"QUOTE#{record.reference}", "data": _encode(record)}
        )

    def get(self, owner: str, reference: str) -> QuoteRecord | None:
        return _decode(_get(self._t, f"USER#{owner}", f"QUOTE#{reference}"), QuoteRecord)
