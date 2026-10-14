"""Load sample slab inventory and compute embeddings.

Mock mode (USE_MOCK=true): seeds an in-process store and prints a summary (useful as a smoke test;
the data does not persist beyond this process, but the API also auto-seeds on startup).

Real mode (USE_MOCK=false): writes to DynamoDB / S3 / OpenSearch Serverless and embeds each image
with Titan Multimodal Embeddings. Pass --images-dir with <slab-id>.png|jpg photos (slab-001 ...)
to use real photos; otherwise placeholder images are generated and the embeddings are NOT
meaningful for similarity.

    python scripts/seed_inventory.py [--images-dir ./sample_images]
"""

import argparse
from pathlib import Path

from tdeez_ai.container import build_container
from tdeez_ai.core.config import Settings
from tdeez_ai.seed import seed_inventory


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings().model_copy(update={"seed_on_startup": False})
    container = build_container(settings)
    count = seed_inventory(
        container.slabs, container.store, container.embedder, container.index, args.images_dir
    )
    mode = "mock" if settings.use_mock else "AWS"
    print(f"Seeded {count} slabs ({mode} mode) with embeddings.")


if __name__ == "__main__":
    main()
