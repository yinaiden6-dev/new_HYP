# Full candidate identity loss：固定开发面板比较

证据级别：已打开的 EVAL32／EVAL128 开发结果；未产生新 593 准确率、外部确认或 HYP GO。

候选来源为冻结 RAW full-gallery C128。三个头均为相同 6 个特征加 bias；REAL／CBIND 均运行完整 127 challenger HOLD/SWITCH，最大 logit ≤ 0 时 HOLD，正值并列取首个 challenger。

仅 LISTWISE_UNIT1 新拟合：原 mixed96（PAIR64 + FULL32），原顺序、FP64、零初始化、seed 17、2000 AdamW 更新。PAIR 单位成本保持不变；FULL 改为 logsumexp([RAW=0, 127 logits]) − target logit。ORIGINAL7 与 TRAIN_UNIT_COST7 的旧参数和预测逐 bit 保留。

| 面板 | 头／路径 | 正确数 | Accuracy | MRR |
|---|---|---:|---:|---:|
| EVAL32 | RAW | 25/32 | 0.781250 | 0.828199405 |
| EVAL32 | ORIGINAL7 | 28/32 | 0.875000 | 0.901413690 |
| EVAL32 | ORIGINAL7_CBIND | 20/32 | 0.625000 | 0.745907738 |
| EVAL32 | TRAIN_UNIT_COST7 | 27/32 | 0.843750 | 0.885750205 |
| EVAL32 | TRAIN_UNIT_COST7_CBIND | 15/32 | 0.468750 | 0.665178571 |
| EVAL32 | LISTWISE_UNIT1 | 26/32 | 0.812500 | 0.870869253 |
| EVAL32 | LISTWISE_UNIT1_CBIND | 14/32 | 0.437500 | 0.649553571 |
| EVAL128 | RAW | 88/128 | 0.687500 | 0.770066940 |
| EVAL128 | ORIGINAL7 | 99/128 | 0.773438 | 0.824013211 |
| EVAL128 | ORIGINAL7_CBIND | 67/128 | 0.523438 | 0.677085789 |
| EVAL128 | TRAIN_UNIT_COST7 | 99/128 | 0.773438 | 0.823318766 |
| EVAL128 | TRAIN_UNIT_COST7_CBIND | 50/128 | 0.390625 | 0.609348733 |
| EVAL128 | LISTWISE_UNIT1 | 101/128 | 0.789062 | 0.831131266 |
| EVAL128 | LISTWISE_UNIT1_CBIND | 52/128 | 0.406250 | 0.616790030 |

| 面板 | REAL 比较 | Rescue | Loss | Net |
|---|---|---:|---:|---:|
| EVAL32 | TRAIN_UNIT_COST7 → LISTWISE_UNIT1 | 0 | 1 | -1 |
| EVAL32 | ORIGINAL7 → LISTWISE_UNIT1 | 0 | 2 | -2 |
| EVAL128 | TRAIN_UNIT_COST7 → LISTWISE_UNIT1 | 2 | 0 | +2 |
| EVAL128 | ORIGINAL7 → LISTWISE_UNIT1 | 7 | 5 | +2 |

TRAIN_UNIT_COST7 → LISTWISE_UNIT1 是 FULL loss 的单因素比较；ORIGINAL7 → LISTWISE_UNIT1 是相对原强基线的比较，同时包含成本与 FULL loss 差异。逐面板报告净增，不将零损失设置为成功前提。

MRR 由完整 gallery 实际 move-to-front 后的秩计算，并保存精确 Fraction。独立 NumPy 重放了 121920 个 logits，随后才打开评估标签及历史结果；训练与两面板的 query、图像 SHA、identity、group、component 均无交集。

结果：`results/rc_full_candidate_identity_loss_v1/result.json`。

独立验证：`results/rc_full_candidate_identity_loss_v1/independent_validation.json`。
