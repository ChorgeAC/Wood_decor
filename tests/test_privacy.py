"""Hard rule 4: logs carry only allowlisted fields; no PII anywhere in log output."""

import json
import logging

import pytest

from tdeez_ai.core.logging import ALLOWED_FIELDS, JsonFormatter, log_event
from tests.conftest import Env


def test_log_event_drops_non_allowlisted_fields() -> None:
    record_holder: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            record_holder.append(record)

    logger = logging.getLogger("tdeez_ai.test_privacy")
    logger.addHandler(Capture())
    logger.setLevel(logging.INFO)
    log_event(
        logger,
        "something",
        status=200,
        email="alice@example.com",
        image_url="https://x/y.png",
    )
    out = json.loads(JsonFormatter().format(record_holder[0]))
    assert out["status"] == 200
    assert "email" not in out and "image_url" not in out
    assert "alice@example.com" not in json.dumps(out)


def test_allowlist_contains_no_pii_fields() -> None:
    banned = {"name", "email", "phone", "image", "image_url", "url", "key", "design", "body", "sub"}
    assert not (ALLOWED_FIELDS & banned)


def test_requests_do_not_leak_customer_data_into_logs(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    key = env.put_image("alice", "renders")
    env.client.post(
        "/quotes",
        headers=env.headers("alice"),
        json={
            "customer": {
                "name": "Zed Uniquename",
                "email": "zed.unique@example.com",
                "phone": "+1 555 000 1111",
                "notes": "secret-notes-123",
            },
            "slab": {"slabId": "slab-001"},
            "design": {"furnitureType": "desk", "customization": {}},
            "visualizationImageKey": key,
        },
    )
    env.client.put("/profile", headers=env.headers("alice"), json={"email": "bad email"})
    env.client.get("/creations/unknown", headers=env.headers("alice"))

    logged = "\n".join(
        JsonFormatter().format(r) for r in caplog.records if r.name.startswith("tdeez_ai")
    )
    assert logged  # we did log something (request lines)
    for secret in (
        "Zed",
        "zed.unique",
        "555",
        "secret-notes",
        key,
        "users/",
        "mock-alice",
        "bad email",
    ):
        assert secret not in logged
