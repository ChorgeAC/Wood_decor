# Accuracy evals (skeletons)

The product targets — **≥ 80 % Top-5 similarity** and **≥ 90 % background removal** — can only be
verified on a curated, labelled image set run against **real Bedrock**. Unit tests use fakes and
say nothing about model quality. These scripts are skeletons: fill in the dataset, run, and record
results per model version.

```
pip install -e ".[evals]"
USE_MOCK=false python evals/similarity_eval.py  --dataset evals/data/similarity.csv
USE_MOCK=false python evals/bg_removal_eval.py  --dataset evals/data/bg_removal.csv
```

## Datasets (create under `evals/data/`, not committed)

* `similarity.csv`: `query_image,expected_slab_id` — a photo (different angle/lighting from the
  inventory shot) and the inventory slab it truly matches. Seed the inventory with the real
  inventory photos first (`scripts/seed_inventory.py --images-dir ...`).
* `bg_removal.csv`: `image,ground_truth_mask` — photo and a binary mask (white = slab).

## Metrics

* Similarity: **Top-5 hit rate** = fraction of queries whose expected slab is in the top 5.
* Background removal: **mean IoU** between the model's alpha mask (alpha > 127) and the ground
  truth; the pass rate counts images with IoU ≥ `--iou-threshold` (default 0.90, an assumption —
  agree the definition of "90 %" with the business before reporting).

Report the dataset size and its composition (species, lighting, backgrounds) with every result.
