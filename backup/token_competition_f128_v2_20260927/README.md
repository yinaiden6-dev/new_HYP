# F128 native token competition V2 / 原生 token 竞争实验

已完成 15 项 seed0 reader 拟合及独立回放。 原生 ColNomic token 关系分支与同一 F128、同种子的 B_CAL 对照。
This archive records the completed development experiment and its independently validated predictions.

数据 / Dataset: H593 原自然执行顺序 0–127 的 128 张开发面板，原 identity-grouped 五折交集；不是旧 EVAL128，也不是完整 H593 测试。
Model: frozen ColNomic backbone and RoMa support; native query tokens and full-reference content matching, no new backbone forward or update.
候选 / Candidates: unchanged natural ColNomic C128, all 128 candidates retained.
Head: frozen phase-specific same-F128 same-seed 18-dimensional B_CAL plus D_g − D_RAW; RAW anchor exactly zero.
Action: HOLD/SWITCH over 127 challengers, fixed threshold zero is primary; inherited B_CAL inner threshold is secondary, without new threshold optimization.
Training: three predeclared readers × five original grouped folds; 15 fits, seed0. Inner validation selects epochs, with epoch0 B_CAL fallback.

## 主结果 / Primary: fixed0

| Model | Seed | Correct / 128 | RAW rescues | RAW breaks | RAW net |
|---|---|---|---|---|---|
| B_CAL | 0 | 103 | 11 | 1 | 10 |
| TOKEN_QR | 0 | 108 | 19 | 4 | 15 |
| TOKEN_QRR_ANCHOR | 0 | 108 | 20 | 5 | 15 |
| TOKEN_QRR_MULTI | 0 | 106 | 18 | 5 | 13 |

## 次结果 / Secondary: inherited B_CAL threshold

| Model | Seed | Correct / 128 | RAW rescues | RAW breaks | RAW net |
|---|---|---|---|---|---|
| B_CAL | 0 | 93 | 11 | 11 | 0 |
| TOKEN_QR | 0 | 101 | 18 | 10 | 8 |
| TOKEN_QRR_ANCHOR | 0 | 102 | 19 | 10 | 9 |
| TOKEN_QRR_MULTI | 0 | 105 | 18 | 6 | 12 |

## 同面板配对变化 / Paired effects against same-F128 B_CAL

救回、误伤、净变化与基线正确样本损失率均完整报告，不仅看困难样本。
Rescues, breaks, net effect, baseline-correct loss rate and changed decisions include easy/baseline-correct queries.


| Model | Seed | Operating point | Rescues | Breaks | Net | Baseline-correct loss rate | Changed decisions |
|---|---|---|---|---|---|---|---|
| TOKEN_QR | 0 | zero | 9 | 4 | 5 | 0.038834951456310676 | 20 |
| TOKEN_QR | 0 | inherited_tau | 11 | 3 | 8 | 0.03225806451612903 | 23 |
| TOKEN_QRR_ANCHOR | 0 | zero | 9 | 4 | 5 | 0.038834951456310676 | 20 |
| TOKEN_QRR_ANCHOR | 0 | inherited_tau | 14 | 5 | 9 | 0.053763440860215055 | 28 |
| TOKEN_QRR_MULTI | 0 | zero | 7 | 4 | 3 | 0.038834951456310676 | 17 |
| TOKEN_QRR_MULTI | 0 | inherited_tau | 13 | 1 | 12 | 0.010752688172043012 | 21 |

## 结构对照 / MULTI versus ANCHOR

相同参数，仅改变 rival graph；固定零阈值为主，继承阈值为次。
Identical parameterization; only the rival graph changes. The paired reference below is TOKEN_QRR_ANCHOR.

| Model | Seed | Operating point | Rescues | Breaks | Net | Baseline-correct loss rate | Changed decisions |
|---|---|---|---|---|---|---|---|
| TOKEN_QRR_MULTI | 0 | zero | 0 | 2 | -2 | 0.018518518518518517 | 6 |
| TOKEN_QRR_MULTI | 0 | inherited_tau | 5 | 2 | 3 | 0.0196078431372549 | 10 |

## 证据边界 / Limits

F128 是已打开的开发面板。不能据此声称完整 H593 提升、空间因果归属、唯一性或独立外部确认；不能用 held 结果重新选择分支、epoch、threshold 或 seed。
This is an opened development panel with seed0. It does not establish full-H593 improvement, spatial causality, uniqueness, or external confirmation. Historical F71 and old F128 heads remain separate lineages.
The parameter-matched MULTI/ANCHOR comparison is the structural comparison. QR comparisons also change the reader structure. Confidence intervals and easy-query regression analysis are available in the linked report.
The frozen promotion gate is only a trigger for development replication. Passing it does not turn this opened panel into an untouched test or external confirmation. All declared arms and seeds remain reported; the best seed is never substituted for the full result.
预先冻结的晋级门槛只授权开发面板复验，不等于独立统计确认或外部确认；全部声明分支与种子均保留。

## 阅读与复核 / Evidence and validation

- [完整分析 / Full analysis](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/analysis/REPORT_TOKEN_COMPETITION_F128_V2_ANALYSIS.md)
- [Analysis summary](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/analysis/analysis_summary.json)
- [Final scientific validation](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/validation.json) · [Joined predictions](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/joined_predictions.json)
- [Frozen protocol](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/protocol.json) · [Final result report](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/REPORT_TOKEN_COMPETITION_V2.md)
- [Evidence manifest](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/evidence_manifest.json) · [Evidence validation](../../ICLR/new%20ROUTEA/RC/results/rc_token_competition_f128_v2/evidence_validation.json)
- [File SHA256 inventory](files.json) · [Export byte validation](validation.json)

科学验收证明原工作区预测及独立回放通过；export validation 只验证归档字节和范围，GitHub 推送确认另存工作区 publication receipt。
Scientific validation, export byte validation and remote publication are separate checks. A local commit alone does not prove remote publication.
源码、协议、轻量结果、JSON 参数和原基线逐种子预测保留原相对路径及 SHA。原图、大模型/优化器 checkpoint、token/embedding/trace 张量、重复候选缓存及运行日志均不上传。
No raw images, binary model/optimizer checkpoints, token/embedding/trace tensors, execution logs, or candidate caches are exported. Their original path/SHA bindings remain in provenance. Full tensor replay requires the original workspace; this Git archive alone is not a complete replay package.
