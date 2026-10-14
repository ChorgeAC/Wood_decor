"""Structured logging with a strict field allowlist.

Only fields named in ``ALLOWED_FIELDS`` are ever emitted. Customer data, image bytes/URLs and
design data have no allowlisted field, so they cannot be logged by accident.
"""

import json
import logging
import sys
from datetime import UTC, datetime

type LogValue = str | int | float | bool

ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "request_id",
        "method",
        "route",
        "status",
        "latency_ms",
        "error_code",
        "error_type",
        "task",
        "attempt",
        "count",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, LogValue] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update({k: v for k, v in fields.items() if k in ALLOWED_FIELDS})
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("tdeez_ai")
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def log_event(
    logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields: LogValue
) -> None:
    """Log ``event`` (a static string) with allowlisted scalar fields only."""
    safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS}
    logger.log(level, event, extra={"fields": safe})
