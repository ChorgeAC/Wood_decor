"""Image helpers: format sniffing and tiny PNG synthesis for mock mode."""

import struct
import zlib
from typing import Literal

type ImageFormat = Literal["png", "jpeg"]

ALLOWED_CONTENT_TYPES: dict[str, ImageFormat] = {"image/jpeg": "jpeg", "image/png": "png"}

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def detect_format(data: bytes) -> ImageFormat | None:
    if data.startswith(_PNG_MAGIC):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    return None


def content_type_for(fmt: ImageFormat) -> str:
    return "image/png" if fmt == "png" else "image/jpeg"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a small solid-colour PNG (used by mock generators and seed data)."""
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        _PNG_MAGIC
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
