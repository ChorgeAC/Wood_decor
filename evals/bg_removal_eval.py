"""Background-removal quality (mask IoU) on a labelled set. Skeleton: supply evals/data/bg_removal.csv.

Requires the optional ``evals`` extra (Pillow, numpy).
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image

from tdeez_ai.bedrock.canvas import NovaCanvasGenerator
from tdeez_ai.bedrock.client import BedrockClient
from tdeez_ai.bedrock.guard import CostGuard
from tdeez_ai.core.config import Settings

TARGET = 0.90


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.90)
    args = parser.parse_args()

    settings = Settings()
    guard = CostGuard(settings.max_images_per_request, settings.max_image_bytes)
    canvas = NovaCanvasGenerator(
        BedrockClient.from_settings(settings), settings.canvas_model_id, guard
    )
    scores: list[float] = []
    with args.dataset.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            result = canvas.remove_background(Path(row["image"]).read_bytes())
            alpha = np.array(Image.open(io.BytesIO(result)).convert("RGBA"))[..., 3] > 127
            truth_img = Image.open(row["ground_truth_mask"]).convert("L").resize(alpha.shape[::-1])
            scores.append(iou(alpha, np.array(truth_img) > 127))
    if not scores:
        raise SystemExit("Dataset is empty.")
    passed = sum(s >= args.iou_threshold for s in scores) / len(scores)
    print(f"mean IoU {np.mean(scores):.3f}; pass rate @IoU>={args.iou_threshold}: {passed:.1%}")
    print("PASS" if passed >= TARGET else "FAIL")


if __name__ == "__main__":
    main()
