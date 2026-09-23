# Finish index 70, then pause GPU collection

Latest user instruction: finish acquisition for index 70; use existing caches afterward, with no additional GPU work. This supersedes the earlier request to restore GPU fusion jobs after acquisition.

## Applied execution boundary

- New bounded dispatcher: `programs/run_rc_h593_finish70_then_pause_v1.py`.
- Authority: `registry/rc_h593_finish70_then_pause_authority_v1_20260923.json`.
- Live status: `results/rc_h593_finish70_then_pause_v1/status.json`.
- Resume job: `5159474_70`, dev_accelerated, original 15-minute / 8-CPU / 1-GPU worker.
- Existing query070 data: 83 of 128 candidate-pair capsules retained. The worker skips saved pairs.
- Only query70 can be submitted, including any necessary continuation chunks and CPU export. After all three original validators pass, the new dispatcher exits with `INDEX70_VALIDATED_COLLECTION_PAUSED`.
- The previous pending continuation `5159449_70` was cancelled before starting to prevent the unbounded dispatcher from expanding after query70. Accounting records zero runtime. Its old ledger is preserved: the old controller either waits for that cancelled array row or stops on cancellation, and cannot pass its first-query gate. It was not remotely killed.
- The local priority guardian that would automatically release GPU fusion jobs was stopped. Existing 50 pending fusion tasks remain held, as do the legacy GPU acquisition tasks. No running GPU computation was interrupted.
- Existing CPU pair-quality training continues. GPU fusion training has not been converted to CPU by this scheduling change.
- Original scientific worker sources, model/feature caches, candidate axes, precision, and validators are unchanged.

## What missing visual-intervention outputs means

The existing protocol modifies only the RoMa image input, separately on the query or reference side:

1. Gray: equal-channel mean, removing color differences.
2. Low-pass: 16x16 average pooling and bilinear resizing, suppressing fine details.
3. Shuffle: a fixed permutation of 4x4 blocks, changing spatial arrangement.

These yield six changed-input arms plus NATIVE. Each arm saves seven stages (COARSE, LR4/2/1, HR4/2/1), both directions, exact token visibility weights, 16x16 overlap grids, 16x16 warp grids, and native overlap dimensions. Overall quality and frozen-head decisions can be recomputed on CPU from those saved outputs and the fixed ColNomic evidence.

The final native COST1 statistics cannot reconstruct matcher outputs under changed images. Even a cache of original single-image features does not contain features computed from a blurred or shuffled image. Therefore extending these interventions to additional images requires new model execution; analyzing already completed interventions does not.

Current independently validated coverage before query70 completes: inside 41, visual 55, coordinate 70. Query70 adds one query to each family, yielding 42/56/71 respectively, not a common panel of 71 for every intervention. The existing common 41-query decision-path analysis remains valid. No text-specific or logo-specific intervention has been established by these gray/low-pass/shuffle controls.

The fixed single-image feature bank for the existing 593-query learning comparisons is already complete. That is a separate cache from visual-intervention outputs.
