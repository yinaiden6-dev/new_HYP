# Collections and training protocol

Current protocol: **package_holdout_v4_supplement**, 2026-10-10. Inventory **0.8.0-rc1** contains **1,117 photos, 130 packaging designs, 118 source groups and 5,405 reference rows (5,330 unique image hashes)**. CSV paths are relative to `dataset/`.

| File | Photos | Designs | Source groups | Use |
|---|---:|---:|---:|---|
| `train.csv` | 783 | 91 | 82 | Fit parameters and construct training negatives |
| `val.csv` | 173 | 21 | 18 | Select checkpoints and hyperparameters |
| `test.csv` | 161 | 18 | 18 | Evaluate the selected method |
| `review.csv` | 0 | 0 | 0 | Unresolved associations held for review |
| `photos.csv` | 1,117 | 130 | 118 | Complete photo inventory |
| `train_subsets/25.csv` | 236 | 25 | 21 | Nested smaller training budget |
| `train_subsets/50.csv` | 429 | 49 | 41 | Nested larger training budget |

All validation/test queries retrieve against **all 5,405 references**. `positives.csv` contains one correct reference for each of the 130 designs. `packages.csv` locates designs, XML and photograph folders; `groups.csv` defines each photographed package's source group and primary split.

## Supplement and reproducible allocation

The supplied ZIP adds package_114 through package_130 and 117 unchanged photographs. Its 17 reference images were already in the gallery; only their previously empty package associations change. The full gallery's IDs, order and image bytes are preserved.

Source components join identities, design SHA-256, original/current photo SHA-256, document IDs, Set IDs, product codes and legacy groups transitively. The new designs form 11 source groups. The existing 113 packages retain their earlier group IDs and roles. Newly discovered links joining distinct frozen groups would stop allocation for review.

Allocation uses seed **20261010**. Largest-remainder rounding sets total train/val/test group targets to **82/18/18** from 70/15/15 weights. Seven new groups enter train, two val and two test. Category-proportional quotas select groups by seeded SHA-256 order, allocating train first, then validation and test. No model scores or scene labels determine membership. Added photograph counts are **77/26/14**.

The training subsets retain their previous selected source groups and extend using one fixed new-training-group order to rounded 25%/50% group budgets, **21/41** groups. Every collection contains complete source groups, and the subsets are nested. The percentages describe source-group budgets, not exact photo counts.

`allocation.json` pins current input hashes and the [pre-import protocol](../review/supplements/20261010_114_130/before/lists/protocol.json). `allocation_report.json` records every added package's group and split. The snapshot also preserves the previous 706/147/147 lists and historical label-correction documentation.

## Training references and evaluation

The default strict training scope permits only the positive design images of packages in the selected training collection: **91 references** for the full training set. Training negatives and synthetic examples must follow that same scope. A setting using the entire reference catalog during training must be declared separately.

```python
from pathlib import Path
import sys
sys.path.insert(0, "dataset/tools")
from splits import training_reference_ids
allowed = set(training_reference_ids(Path("dataset"), "train"))
```

All 5,405 reference images may be indexed for validation/test retrieval using a fixed checkpoint. Rank the complete gallery, collapse exact image-byte duplicates by their highest-ranked row, and break score ties by CSV row order. Report photo/package R@1/5/10, MRR and source-group uncertainty. Select checkpoints on validation only, then evaluate test.

The original 1,000 photos have development/inspection history. Earlier use of the 117 supplement photos is unknown. Source isolation does not guarantee unseen manufacturers, ingredients or capture sessions. All 117 additions have completed human review of scene labels and foreground occlusion, with `reviewed: true` and no exclusions or missing labels. See [the completed review](../review/photos/manual_reviews/20261010T141601Z/summary.json); the model prelabel run remains a historical snapshot.

## Caches, review and historical results

Core OCR, six image-vector banks, four text-vector banks and 27 baseline/fusion score caches now cover **all 1,117 photos and 5,405 references** and have passed full verification. Original vectors and score rows are preserved. See [the cache completion](../../cache/completions/20261010T142424Z/verification.json). Experiment reports and protocol-specific method/foreground caches still cover **0.7.1-rc1 / Full-1k**; evaluation for the expanded inventory remains pending. Historical files retain their original provenance.

`active/` is the editor's filtered mirror. Exclusion edits affect this mirror while preserving formal source groups. Human confusion decisions are retained and their derived subset is updated; that curated subset remains distinct from `val.csv`. Model suggestions for newly added packages are pending.

```bash
python -B dataset/tools/splits.py check
python -B dataset/tools/check.py verify
python -B dataset/tools/photo_metadata.py check
python -B dataset/tools/allocate_splits.py --output /tmp/drugpack-v4-splits
```

The allocator writes to an empty output directory. Verification checks frozen input/output hashes, all current memberships, source leakage, nested budgets and byte-identical allocation reproduction from the local snapshot.
