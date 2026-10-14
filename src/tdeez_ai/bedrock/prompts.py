"""Prompt builders.

Image prompts stay under Nova Canvas' 1024-character limit and use positive phrasing for what
must stay fixed (the docs advise against negations); the unwanted traits go in negative text.
"""

from tdeez_ai.models.common import FurnitureType, Species
from tdeez_ai.models.customization import CustomizationValues, EdgeType, EpoxyType

CANVAS_PROMPT_LIMIT = 1024

GEOMETRY_CLAUSE = (
    "The reference live-edge slab outline is fixed and must remain unchanged: preserve its exact "
    "shape, contour and proportions. Never alter, reshape, crop, straighten or resize the slab outline."
)
GEOMETRY_NEGATIVE = (
    "altered slab outline, reshaped slab, cropped slab, straightened edge, different slab, "
    "text, watermark, people, distorted proportions"
)

FURNITURE_LABEL: dict[FurnitureType, str] = {
    FurnitureType.DINING_TABLE: "dining table",
    FurnitureType.COFFEE_TABLE: "coffee table",
    FurnitureType.DESK: "desk",
}

# (style key, title, descriptor). Distinct looks, same slab geometry.
CONCEPT_STYLES: tuple[tuple[str, str, str], ...] = (
    (
        "modern_minimal",
        "Modern Minimal",
        "Modern minimalist design, slim matte black steel legs, bright airy studio, soft daylight.",
    ),
    (
        "warm_rustic",
        "Warm Rustic",
        "Warm rustic farmhouse design, chunky trestle wood base, cozy interior, golden evening light.",
    ),
    (
        "organic_contemporary",
        "Organic Contemporary",
        "Organic contemporary design, sculptural pale wood pedestal legs, neutral gallery setting.",
    ),
)


def clip(text: str, limit: int = CANVAS_PROMPT_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "."


def concept_prompt(furniture: FurnitureType, descriptor: str) -> str:
    label = FURNITURE_LABEL[furniture]
    return clip(
        f"Photorealistic product photo of a {label} whose top is the reference live-edge wood "
        f"slab. {descriptor} {GEOMETRY_CLAUSE}"
    )


def customization_prompt(furniture: FurnitureType, values: CustomizationValues) -> str:
    label = FURNITURE_LABEL[furniture]
    parts = [f"Photorealistic product photo of a {label} with a live-edge wood slab top."]
    parts.append(f"Legs: {values.leg_style.value.replace('_', ' ')} style.")
    parts.append(f"Wood grain runs {values.wood_orientation.value} across the top.")
    if values.edge_type is not EdgeType.LIVE:
        parts.append(f"Edge profile: {values.edge_type.value}, applied to straight sides only.")
    if values.epoxy_type is not EpoxyType.NONE:
        color = f" in color {values.epoxy_color}" if values.epoxy_color else ""
        parts.append(f"{values.epoxy_type.value.capitalize()} epoxy resin finish{color}.")
    if values.dimensions is not None:
        d = values.dimensions
        parts.append(
            f"Overall proportions about {d.length_in / d.width_in:.1f} to 1 length to width."
        )
    parts.append(GEOMETRY_CLAUSE)
    return clip(" ".join(parts))


def room_prompt(furniture: FurnitureType, values: CustomizationValues | None) -> str:
    label = FURNITURE_LABEL[furniture]
    legs = f", {values.leg_style.value.replace('_', ' ')} legs" if values else ""
    return clip(
        f"A {label} with a live-edge wood slab top{legs}, placed naturally on the floor, realistic "
        "scale, matching room lighting and soft contact shadows."
    )


SUPPORTED_SPECIES = ", ".join(s.value for s in Species)

VISION_SYSTEM = (
    "You analyze photos for a custom live-edge slab furniture shop. Reply with a single JSON "
    "object only: no markdown, no commentary. Never state physical measurements, units, prices "
    "or structural/safety claims."
)


def slab_analysis_prompt() -> str:
    return (
        "Analyze the wood slab in the image. Return JSON with exactly these keys:\n"
        '"shape_description": short description of the slab outline and live edges;\n'
        '"length_to_width_ratio": number (long side divided by short side, estimated from the '
        "image) or null if unclear;\n"
        '"defects": list of {"kind": "knot"|"crack"|"bark_inclusion", "severity": '
        '"low"|"medium"|"high", "location_hint": short text}; only these three kinds;\n'
        f'"species": one of [{SUPPORTED_SPECIES}] or null if none fits;\n'
        '"species_confidence": number from 0 to 1, honest and calibrated;\n'
        '"summary": two sentences describing the slab for a customer, without any measurements.'
    )


def room_prompt_check() -> str:
    return (
        "Is this photo suitable for placing a table or desk into it? It needs visible floor "
        "space and a wall or room context. Return JSON with exactly these keys:\n"
        '"suitable": boolean; "confidence": number 0 to 1; "has_wall": boolean; '
        '"has_floor": boolean; "reason": short text.'
    )


def leg_prompt(furniture: FurnitureType, ratio: float | None, summary: str | None) -> str:
    ratio_text = f"{ratio:.1f}" if ratio is not None else "unknown"
    return (
        f"Recommend one leg style for a {FURNITURE_LABEL[furniture]} with a live-edge slab top. "
        f"Slab length-to-width ratio: {ratio_text}. Slab notes: {summary or 'none'}.\n"
        'Return JSON with exactly: "leg_style": one of [hairpin, trestle, u_shaped, sled, '
        'x_frame]; "rationale": one or two sentences, no measurements.'
    )
