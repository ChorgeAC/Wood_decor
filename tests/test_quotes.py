"""Quote request (TBC-013)."""

import re
from collections.abc import Callable
from typing import Any

from tdeez_ai.core.errors import UpstreamError
from tdeez_ai.core.keys import user_prefix
from tdeez_ai.repositories.memory import InMemoryMailer
from tests.conftest import Env

ALICE = user_prefix("mock-alice")
BOB = user_prefix("mock-bob")


class FailingMailer(InMemoryMailer):
    def send(self, to: str, subject: str, body: str) -> None:
        raise UpstreamError("smtp down")


def _payload(env: Env, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "customer": {
            "name": "Alice Example",
            "email": "alice@example.com",
            "phone": "+1 555 123 4567",
        },
        "slab": {"slabId": "slab-001"},
        "design": {"furnitureType": "dining_table", "customization": {}},
        "visualizationImageKey": env.put_image("alice", "renders"),
    }
    body.update(overrides)
    return body


def test_quote_returns_reference_and_persists(env: Env) -> None:
    r = env.client.post("/quotes", headers=env.headers(), json=_payload(env))
    assert r.status_code == 200, r.text
    data = r.json()["data"]
    assert re.fullmatch(r"TD-\d{8}-[A-Z2-9]{6}", data["reference"])
    assert "not a quote" in data["disclaimer"]
    stored = env.container.quote_repo.get("mock-alice", data["reference"])
    assert stored is not None
    assert stored.email_sent is True


def test_quote_emails_tdeez_inbox(make_env: Callable[..., Env]) -> None:
    env = make_env(quote_inbox="orders@tdeez.example")
    reference = env.client.post("/quotes", headers=env.headers(), json=_payload(env)).json()[
        "data"
    ]["reference"]
    assert len(env.mailer.outbox) == 1
    mail = env.mailer.outbox[0]
    assert mail.to == "orders@tdeez.example"
    assert reference in mail.subject
    assert "alice@example.com" not in mail.subject  # PII stays out of the subject
    assert "not a quote" in mail.body.lower() or "not a quote" in mail.body


def test_references_are_unique(env: Env) -> None:
    refs = {
        env.client.post("/quotes", headers=env.headers(), json=_payload(env)).json()["data"][
            "reference"
        ]
        for _ in range(5)
    }
    assert len(refs) == 5


def test_mail_failure_still_confirms_and_stores(make_env: Callable[..., Env]) -> None:
    env = make_env(mailer=FailingMailer())
    r = env.client.post("/quotes", headers=env.headers(), json=_payload(env))
    assert r.status_code == 200
    stored = env.container.quote_repo.get("mock-alice", r.json()["data"]["reference"])
    assert stored is not None
    assert stored.email_sent is False


def test_quote_validation(env: Env) -> None:
    bad_email = _payload(env)
    bad_email["customer"]["email"] = "not-an-email"
    assert env.client.post("/quotes", headers=env.headers(), json=bad_email).status_code == 422

    no_name = _payload(env)
    no_name["customer"]["name"] = ""
    assert env.client.post("/quotes", headers=env.headers(), json=no_name).status_code == 422

    bad_phone = _payload(env)
    bad_phone["customer"]["phone"] = "call me"
    assert env.client.post("/quotes", headers=env.headers(), json=bad_phone).status_code == 422

    both = _payload(env, slab={"slabId": "slab-001", "imageKey": ALICE + "slabs/x.png"})
    assert env.client.post("/quotes", headers=env.headers(), json=both).status_code == 422


def test_quote_rejects_other_customers_assets(env: Env) -> None:
    bobs = env.put_image("bob", "renders")
    r = env.client.post(
        "/quotes", headers=env.headers("alice"), json=_payload(env, visualizationImageKey=bobs)
    )
    assert r.status_code == 404
    assert env.mailer.outbox == []


def test_quote_unknown_slab(env: Env) -> None:
    r = env.client.post(
        "/quotes", headers=env.headers(), json=_payload(env, slab={"slabId": "nope"})
    )
    assert r.status_code == 404
