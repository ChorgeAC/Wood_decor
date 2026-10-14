"""Per-request cost guardrails."""

from dataclasses import dataclass

from tdeez_ai.core.errors import BadRequest, PayloadTooLarge


@dataclass(frozen=True)
class CostGuard:
    max_images: int
    max_image_bytes: int

    def check_count(self, count: int) -> None:
        if count < 1 or count > self.max_images:
            raise BadRequest(f"Please request between 1 and {self.max_images} images.")

    def check_image(self, data: bytes) -> None:
        if len(data) > self.max_image_bytes:
            raise PayloadTooLarge("That image is too large. Please use a smaller image.")
