"""Top-5 similarity hit-rate on a labelled set. Skeleton: supply evals/data/similarity.csv."""

import argparse
import csv
from pathlib import Path

from tdeez_ai.container import build_container
from tdeez_ai.core.config import Settings

TARGET = 0.80
TOP_K = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    settings = Settings()
    container = build_container(settings)
    hits = total = 0
    with args.dataset.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vector = container.embedder.embed_image(Path(row["query_image"]).read_bytes())
            top = [h.slab_id for h in container.index.query(vector, TOP_K)]
            hits += row["expected_slab_id"] in top
            total += 1
    if total == 0:
        raise SystemExit("Dataset is empty.")
    rate = hits / total
    print(f"Top-{TOP_K} hit rate: {rate:.1%} ({hits}/{total}); target {TARGET:.0%}")
    print("PASS" if rate >= TARGET else "FAIL")


if __name__ == "__main__":
    main()
