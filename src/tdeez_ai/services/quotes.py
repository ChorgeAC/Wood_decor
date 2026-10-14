"""Quote request (TBC-013): persist, email the TDeez inbox, return a reference."""

import logging
import secrets
from datetime import UTC, datetime

from tdeez_ai.core.config import Settings
from tdeez_ai.core.errors import AppError, NotFound
from tdeez_ai.core.keys import assert_owned, assert_readable
from tdeez_ai.core.logging import log_event
from tdeez_ai.models.quote import QuoteConfirmation, QuoteRecord, QuoteRequest
from tdeez_ai.repositories.base import CreationRepository, Mailer, QuoteRepository, SlabRepository

logger = logging.getLogger(__name__)

_REF_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no look-alike characters


def new_reference(now: datetime) -> str:
    suffix = "".join(secrets.choice(_REF_ALPHABET) for _ in range(6))
    return f"TD-{now:%Y%m%d}-{suffix}"


def _one_line(text: str) -> str:
    return " ".join(text.split())


class QuoteService:
    def __init__(
        self,
        repo: QuoteRepository,
        creations: CreationRepository,
        slabs: SlabRepository,
        mailer: Mailer,
        settings: Settings,
    ) -> None:
        self._repo = repo
        self._creations = creations
        self._slabs = slabs
        self._mailer = mailer
        self._settings = settings

    def submit(self, owner: str, req: QuoteRequest) -> QuoteConfirmation:
        if req.slab.slab_id is not None:
            if self._slabs.get(req.slab.slab_id) is None:
                raise NotFound("That slab could not be found.")
        elif req.slab.image_key is not None:
            assert_readable(owner, req.slab.image_key)
        if req.visualization_image_key is not None:
            assert_owned(owner, req.visualization_image_key)
        if req.creation_id is not None and self._creations.get(owner, req.creation_id) is None:
            raise NotFound("That creation could not be found.")

        now = datetime.now(UTC)
        reference = new_reference(now)
        email_sent = self._notify(reference, req)
        self._repo.put(
            owner,
            QuoteRecord(reference=reference, submitted_at=now, request=req, email_sent=email_sent),
        )
        return QuoteConfirmation(
            reference=reference,
            submitted_at=now,
            message=f"Thanks! Your request was received. Your reference is {reference}.",
        )

    def _notify(self, reference: str, req: QuoteRequest) -> bool:
        """The request is stored regardless; a mail failure must not lose the customer's request."""
        inbox = self._settings.quote_inbox or "orders@tdeez.invalid"
        c, d = req.customer, req.design
        body = "\n".join(
            [
                f"Quote request {reference}",
                f"Customer: {_one_line(c.name)} <{c.email}>",
                f"Phone: {c.phone or '-'}",
                f"Furniture: {d.furniture_type.value}",
                f"Slab: {req.slab.slab_id or 'customer upload'}",
                f"Customization: {d.customization.model_dump_json(by_alias=True)}",
                f"Notes: {_one_line(c.notes) if c.notes else '-'}",
                "",
                "AI visuals are drafts. This is a request, not a quote.",
            ]
        )
        try:
            self._mailer.send(inbox, f"New TDeez quote request {reference}", body)
        except AppError as exc:
            log_event(logger, "quote_email_failed", error_code=exc.code)
            return False
        return True
