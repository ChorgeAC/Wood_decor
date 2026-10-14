"""Sample slab inventory (mock data) + embedding computation. Used by the seed script and mock mode."""

from pathlib import Path

from tdeez_ai.bedrock.embeddings import ImageEmbedder
from tdeez_ai.core.images import content_type_for, detect_format, make_png
from tdeez_ai.models.common import Availability, Species
from tdeez_ai.models.slab import Slab, SlabDimensions
from tdeez_ai.repositories.base import ImageStore, SimilarityIndex, SlabRepository

_SPECIES_COLORS: dict[Species, tuple[int, int, int]] = {
    Species.BLACK_WALNUT: (74, 52, 40),
    Species.BIG_LEAF_MAPLE: (214, 182, 140),
    Species.WHITE_OAK: (196, 164, 118),
    Species.CHERRY: (150, 80, 56),
    Species.ACACIA: (176, 120, 70),
}


def _slab(
    n: int, species: Species, availability: Availability, dims: tuple[float, float, float] | None
) -> Slab:
    sid = f"slab-{n:03d}"
    return Slab(
        id=sid,
        name=f"{species.value.replace('_', ' ').title()} Slab {n}",
        species=species,
        availability=availability,
        dimensions=SlabDimensions(length_in=dims[0], width_in=dims[1], thickness_in=dims[2])
        if dims
        else None,
        image_key=f"inventory/{sid}.png",
        description="Sample inventory slab (mock data).",
    )


SAMPLE_SLABS: tuple[Slab, ...] = (
    _slab(1, Species.BLACK_WALNUT, Availability.AVAILABLE, (96, 34, 2.5)),
    _slab(2, Species.BLACK_WALNUT, Availability.AVAILABLE, (84, 30, 2.0)),
    _slab(3, Species.BIG_LEAF_MAPLE, Availability.AVAILABLE, (72, 28, 2.5)),
    _slab(4, Species.BIG_LEAF_MAPLE, Availability.RESERVED, (60, 26, 2.0)),
    _slab(5, Species.WHITE_OAK, Availability.AVAILABLE, (90, 32, 3.0)),
    _slab(6, Species.CHERRY, Availability.AVAILABLE, None),
    _slab(7, Species.ACACIA, Availability.AVAILABLE, (78, 24, 2.0)),
    _slab(8, Species.WHITE_OAK, Availability.SOLD, (66, 22, 2.5)),
)


def seed_inventory(
    slabs: SlabRepository,
    store: ImageStore,
    embedder: ImageEmbedder,
    index: SimilarityIndex,
    images_dir: Path | None = None,
) -> int:
    """Store slabs + images and index their embeddings.

    With ``images_dir``, ``<slab-id>.png|jpg`` files are used as the slab photos; otherwise
    placeholder PNGs are generated (their embeddings are only meaningful in mock mode).
    """
    for slab in SAMPLE_SLABS:
        image = _load_image(slab, images_dir) or make_png(64, 64, _SPECIES_COLORS[slab.species])
        fmt = detect_format(image) or "png"
        store.put_bytes(slab.image_key, image, content_type_for(fmt))
        slabs.put(slab)
        index.upsert(slab.id, embedder.embed_image(image))
    return len(SAMPLE_SLABS)


def _load_image(slab: Slab, images_dir: Path | None) -> bytes | None:
    if images_dir is None:
        return None
    for ext in ("png", "jpg", "jpeg"):
        candidate = images_dir / f"{slab.id}.{ext}"
        if candidate.is_file():
            return candidate.read_bytes()
    return None
