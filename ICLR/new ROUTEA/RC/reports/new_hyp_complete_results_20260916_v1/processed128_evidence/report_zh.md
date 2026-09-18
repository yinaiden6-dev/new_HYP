# processed128：是否伤害原正确

冻结 full-H593 头；完整 5413 reference／5412 身份，自然 C128，阈值不变。

| 模型 | 正确 / 128 | 救回 | 改错 | 净增 |
|---|---:|---:|---:|---:|
| RAW | 121 | 0 | 0 | +0 |
| COST1 | 122 | 1 | 0 | +1 |
| CE | 123 | 2 | 0 | +2 |
| COST4 | 121 | 0 | 0 | +0 |
| GROUP_COST4 | 121 | 0 | 0 | +0 |
| RAW2_CE | 121 | 0 | 0 | +0 |

C128 内含 target：127/128。

这是合成处理图回归，不是独立外部确认。未观察到损失不等于总体永不损失。

## COST1

RAW 原正确中的改错：0/121；来源分组净增区间：[0.0, 0.0234375]。

| 变化 | query | 原图 |
|---|---|---|
| 救回 | PROC-Q-0099 | [50mLcarton](</hkfs/work/workspace/scratch/ap7811-benchmark/1/processed/50mLcarton__motion_blur+aged_05.jpg>) |

## CE

RAW 原正确中的改错：0/121；来源分组净增区间：[0.0, 0.0390625]。

| 变化 | query | 原图 |
|---|---|---|
| 救回 | PROC-Q-0091 | [cla04-0002-05](</hkfs/work/workspace/scratch/ap7811-benchmark/1/processed/cla04-0002-05__aged+motion_blur_02.jpg>) |
| 救回 | PROC-Q-0099 | [50mLcarton](</hkfs/work/workspace/scratch/ap7811-benchmark/1/processed/50mLcarton__motion_blur+aged_05.jpg>) |
