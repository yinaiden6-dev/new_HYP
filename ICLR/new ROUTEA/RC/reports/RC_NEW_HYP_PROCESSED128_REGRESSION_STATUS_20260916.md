# processed128 冻结模型回归检验：已提交，尚无本轮成绩

核验时间（UTC）：2026-09-15T22:56:14.240774+00:00

用户问题：ColNomic 在 `1/processed` 本来就很强，加入当前 new HYP 会不会改坏原正确？

本轮固定 128 个不同原图来源各一张处理图，按哈希选取，未按模型成绩挑选。合格来源 511 个；6 个重名来源共 24 张图在抽样前排除。128 个来源在现有 XML 映射下属于 128 个不同分组，其中 5 个身份在 H593 出现过、123 个未出现过，图片字节重叠为 0。

历史 ColNomic-7B 单独检索记录：完整 processed2000 为 93.8%，本轮冻结子集在历史记录中为 117/128。历史与当前使用的 gallery 缓存、身份修正及打分 batch 不完全相同，因此该数字仅作为背景，不作为当前 COST1/CE 的直接比较基线。当前 RAW 重新计算。

比较固定 full-H593 COST1 主头、CE 次头，以及同 bundle 的其他控制头。完整 5413 物理 reference / 5412 修正身份，自然 C128，不插入 target；复用原编码器、RoMa、FP64 评分与 HOLD/SWITCH 阈值，不训练或调参。

结果将直接列出正确数、救回、改错、净增、原正确中的损失率、C128 遗漏以及逐图清单；同时分开报告训练见过与未见过的身份。这是合成处理图回归，不是新的独立外部 HYP 确认。

| 任务 | Job ID | 分区 | 每分片时限 |
|---|---|---|---|
| RAW 首片 | 5147076 | dev_accelerated | 10 分钟 |
| RoMa 首片 | 5147077 | dev_accelerated | 15 分钟 |
| RAW 余下 15 片 | 5147078 | accelerated | 10 分钟 |
| RoMa 余下 15 片 | 5147079 | accelerated | 15 分钟 |
| 自动核算 | 5147080 | dev_accelerated,accelerated | 10 分钟 |

提交后 live scheduler 显示 RAW 首片 PENDING，其余按依赖等待；尚未声称运行成功或得到科学结果。后续分片须首片 RAW 和 RoMa 验证完成后开始。

8 项本地预检通过，包括原 RAW/RoMa 数值循环、原 head/action AST 一致、两套冻结运行时、独立哈希抽样、稳定排序、分组统计和实际启动命令。所有提交到 Slurm 的脚本字节均与冻结启动器一致。

- [冻结方案](</hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/plan/RC_NEW_HYP_PROCESSED128_REGRESSION_V1_20260916.md>)
- [抽样与排除清单](</hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_new_hyp_processed128_regression_v1/selection.json>)
- [预检](</hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_new_hyp_processed128_regression_v1/submission/preflight.json>)
- [提交回执](</hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_new_hyp_processed128_regression_v1/submission/jobs.json>)
- 结果输出目录：`/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_new_hyp_processed128_regression_v1`；完成后自动生成 `result.json`、`per_query.csv`、`report_zh.md` 和 `result_validation.json`。

## 提交后的运行更新

2026-09-15T22:57:04.374211+00:00：`5147076_0` 已在 `dev_accelerated / hkn0401` RUNNING，已运行 53 秒；stdout/stderr 尚为空，尚未观察到模型输出。其余任务按依赖等待。此状态不代表推理已通过或已有结果。
