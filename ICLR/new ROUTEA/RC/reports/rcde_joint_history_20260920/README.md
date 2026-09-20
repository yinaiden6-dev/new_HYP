# 早期 query–reference 联合 RCDE：结果、停止依据与归档索引

核对日期：2026-09-20。此页整理已有原始文件，不重新训练、不重新推理，不改写历史结果。各 JSON 和历史文档保持源文件字节；新增的本页负责解释跨版本关系。

**早期 RCDE 有记录。此前 GitHub 主路线导出遗漏了这里的大部分原始结果，不能据此说实验未做。也不能把各分支都解释成同一种失败。**

这里的 RCDE 是早期 DINO reference-conditioned / candidate-relative evidence 分支及其 connected-superregion、H0、Track-R、GX 后续诊断。它与当前 RoMa 软可见性 × ColNomic full-reference MaxSim × 检索监督小头不同。这里的 594、582、567 和 32 均为各自历史合同中的人口，不能与当前 H593 五折、旧 ORIGINAL7 EVAL32/128 合并比较。

## 为什么暂停和停止

1. **BAG 最初暂停是主线切换，并非当时已经证明失败。** 四折各 2,048 updates 后完成训练/检查点验证，但当时仅有八查询工程 prejoin，没有 held-out label join 和科学归约。V22 把主线改成 candidate-conditioned connected-superregion，并把后续 V 初始化固定到 RCDE_CONTEXT。[V22 原文](../../registry/current_authority_v22_20260814.json)、[BAG 完成合同的历史说明](../../plan/DINO_RCDE_BAG_FULL594_COMPLETION_CONTRACT_V1_20260827.md)。合同中的“尚未评价”描述的是启动补评之前，不能当成如今状态。
2. **BAG 后来补完了 594-query 评价，当前确有经独立验证的 NO-GO。** 四折、49 个 supergroup，固定模型和既定检验下方向与候选绑定不足。其效力只涉及这一套冻结 BAG decoder，不能外推成“query/reference 联合建模不可能”。
3. **其余具体路径分别因绝对校准、完整候选竞争、空间对照或合法 donor 不足而停止。** 它们仍保留局部积极结果，例如 pairwise 方向提升、MRR 提升、共享偏置改善 loss；这些未变成各自要求的部署增益。
4. **V123 需要单独标注协议缺陷。** 原 NO-GO 数字保留，但后继 V124 authority 已明确将其因果结论降级；不能用旧 validator 的 PASS 覆盖后来发现的问题。

## 实验总表

| 分支 / 判定路径 | 人口与比较基准 | 原始结果 | 停止或保留理由 |
|---|---|---|---|
| RCDE_BAG 固定 decoder | 自然 C128 target-hit 594；四折、49 组；target vs 冻结 RAW 最强非目标 rival | 组均衡正方向 0.5709998；95% bootstrap 下界 0.4815517；单臂检验 p=0.0559；C_BIND margin-drop 下界 −0.1666799、p=0.1465 | 方向门要求 ≥0.60、下界 >0.50、p<0.05；绑定门也未通过。有效窄范围 NO-GO；这不是 R@1 |
| V114 exact-geometry donor | 594 查询，16 cross-fit scopes，2,376 scope nodes | metadata 合法边 225,792，但满足完整 exact geometry key 的边 0、最大匹配 0 | 绝对损失的 donor 支持为空；停止这条 donor 构造，`scientific_GO_or_NO_GO=null`，不代表 V 训练失败 |
| H0 U4 条件绝对验证 | 594 中可匹配 567；给定 target/donor；INIT vs Track-H | 主 loss 0.6904257062 → 0.6903228272，改善约 0.000102879；历史诊断 90 rescue / 56 break | loss 改善未达预定 0.01；pairwise rescue 不等于在完整 C128 中选中 target |
| H0 共享偏置修复 | 同一 U4 开发人口，留一折拟合偏置 | LOFO loss 改善相对 INIT 为 0.0166621；margin 最大变化约 2.22e−16 | 优化资格 PASS，但相同偏置在 target−donor 中抵消；不会改善候选排名，不能作为检索 GO |
| H0 C128 PJ2 | 594 查询、76,032 候选；INIT vs Track-H 的完整候选排名 | Top-1 1/594 → 2/594；2 rescue / 1 break；组均衡增益 0.0014172、p=0.2438；MRR 增益 0.0142339、p=0.0018 | MRR 有积极信号，但 Top-1 没过 ≥0.03 和显著性门；基准是 INIT，不能误写成 ColNomic RAW |
| Component Scale-4 | 排除 12 个开发查询后 582；INIT / Track-H / Scale-4 | 正确数 1 / 2 / 10；相对 INIT 为 10 rescue / 1 break；相对 Track-H 为 10 / 2 | 有有限提升，仍未过原冻结门，状态 `COMPONENT_SCALE4_INTERNAL_CONFIRMATION_NO_GO`；不能说所有量均无增益 |
| ColNomic + component residual | 同一 582、自然 RAW C128，固定 residual weight=1 | RAW 466/582 → FUSED 464/582；1 rescue / 3 break | 直接融合净损失 2；这条组合没有技术优势，不是当前 RoMa/ColNomic COST1 头 |
| Track-R V123 三臂 | 594、49 组；ALL_PATCH / query region × full reference / local component | 原 decision_A 为 NO_GO，decision_B 为 NO_INCREMENT | V124 发现完整 C128 的 C_DINO_V 控制未正确构造、FULL/LOCAL 混入 decoder 粒度差异、缺 pre-P C_COL_P；只保留诊断效力 |
| GX-CBNR R0 | 历史 balanced32，四折，16 RAW-correct / 16 RAW-wrong | 保留 16/16；1 rescue / 0 break；REAL−P_COORD=−0.821153，95% CI [−1.171043, −0.476994] | 错例正方向 1/16，低于 11/16；空间 null 反而更强，窄范围几何排他性 NO-GO |
| GX 三轮修复 | 同一历史诊断体系；R1Q、R1P 四折；R2A 仅 fold1 | 增加优化预算后错例正方向 6/16；直接空间 pair loss 后 7/16；live 4D consensus 在 16/32/72 updates 为 1/4、0/4、0/4 | 均未达固定门，属于修复不足诊断；不能将单折预算曲线视为独立泛化结论 |

这些结果支持停止重复推进对应的固定实现，但没有构成对所有联合模型、所有 DINO 表征或所有 reference-conditioned 机制的否定。这里也不声称已经从历史数据识别出唯一统一根因。

## 原始证据入口

每组依次给出原结果与独立验证；这些“独立验证”是历史已生成的工件，本次只核验归档及关键字段，没有重跑模型或重新计算所有统计检验。

- BAG：[result.json](../../results/dino_rcde_bag_full594_completion_v1/result.json) · [independent_validation.json](../../results/dino_rcde_bag_full594_completion_v1/independent_validation.json) · [594 查询 join 行](../../results/dino_rcde_bag_full594_completion_v1/label_join.json) · [prejoin 清单](../../results/dino_rcde_bag_full594_completion_v1/prejoin_manifest.json)。50 份 shard validation JSON 一并保存。
- BAG 训练完成证据：[foldset_manifest](../../results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_manifest.json) · [foldset_validation](../../results/dino_rcde_r1_main_bag_foldset_v1_0/foldset_validation.json)。其 `heldout_forward_count=0` 对应当时训练归档阶段，不覆盖后续补评。
- donor 空图：[result](../../results/dino_rcde_sr0_mt_v2_donor_matching_v1/result.json) · [validation](../../results/dino_rcde_sr0_mt_v2_donor_matching_validation_v1/result.json) · [结项说明](../REPORT_V114_EXACT_GEOMETRY_DONOR_EMPTY_GRAPH_JOB5094214_20260821.md)。
- U4：[result](../../results/dino_rcde_h0_u4_absolute_gate_v1/result.json) · [validation](../../results/dino_rcde_h0_u4_absolute_gate_validation_v1/result.json) · [原始汇总行](../../results/dino_rcde_h0_u4_raw_aggregate_v2/result.json)。
- 偏置：[result](../../results/dino_rcde_h0_bias_calibrator_optimization_v1/result.json) · [validation](../../results/dino_rcde_h0_bias_calibrator_optimization_validation_v1/result.json) · [历史解释](../REPORT_H0_U4_NO_GO_LITERATURE_AND_BIAS_REPAIR_DECISION_20260824.md)。
- 完整 C128 PJ2：[原始完整 result，约 37.4 MB](../../results/dino_rcde_h0_c128_target_free_postjoin_pj2_v1/result.json) · [validation](../../results/dino_rcde_h0_c128_target_free_postjoin_pj2_validation_v1/result.json)。保留全部 `rows`，未改写成只含摘要的 JSON。
- Scale-4：[result](../../results/dino_rcde_h0_component_scale4_confirmation_v1/result.json) · [validation](../../results/dino_rcde_h0_component_scale4_confirmation_validation_v1/result.json)。
- RAW 残差融合：[result](../../results/dino_rcde_colnomic_component_residual_confirmation_v1/result.json) · [validation](../../results/dino_rcde_colnomic_component_residual_confirmation_validation_v1/result.json)。
- V123：[历史 result](../../results/dino_rcde_track_r_scientific_result_v1/result.json) · [历史 validation](../../results/dino_rcde_track_r_scientific_validation_v1/result.json)；必须同时阅读 [V124 authority 的 `v123_forensic_disposition`](../../registry/dino_rcde_track_r_v124_core_e0_authority_v1_20260824.json) 和 [协议修正说明](../../plan/DINO_RCDE_TRACK_R_V124_CORRECTED_MECHANISM_AUTHORITY_PLAN_DRAFT_V1_20260824.md)。
- GX R0：[result](../../results/dino_rcde_gx_formal_r0_science_v1/result.json) · [validation](../../results/dino_rcde_gx_formal_r0_science_validation_v1/result.json) · [三轮修复结项](../REPORT_DINO_RCDE_GX_R0_THREE_REPAIR_ROUNDS_CLOSURE_20260826.md)。
- R1Q：[result](../../results/dino_rcde_gx_r1q_optimization_reduction_v1/result.json) · [validation](../../results/dino_rcde_gx_r1q_optimization_validation_v1/result.json)。
- R1P：[result](../../results/dino_rcde_gx_r1p_direct_pair_reduction_v1/result.json) · [validation](../../results/dino_rcde_gx_r1p_direct_pair_validation_v1/result.json)。
- R2A：[72-update result](../../results/dino_rcde_gx_r2a_live_consensus_fold1_72_v1/result.json) · [validation](../../results/dino_rcde_gx_r2a_live_consensus_fold1_72_validation_v1/result.json) · [预算曲线原始 JSON](../../results/dino_rcde_gx_r2a_live_consensus_fold1_budget_curve_v1/result.json)。

## 如何使用这份历史

写论文可以报告：旧方案能产生部分候选相对证据，但所测试的固定 decoder、绝对校准、完整 C128 排名和空间对照没有共同形成稳定部署收益；转向新机制有明确的实验依据。不要写成“联合 query/reference 已被证明无效”，也不要把一次工程暂停改写为科学失败。

旧方案中的 joint 有具体含义：在 query/reference tokens、candidate-relative comparator 或空间 relational head 上建模；当前 new HYP 联合的是匹配质量、可见性与内容证据并训练共享 action。两者不是只换了名称的同一实验。

## 保存范围

[evidence_index.json](evidence_index.json) 提供 12 个主要结果入口、原始状态、当前解释等级与 SHA256。[source_manifest.json](source_manifest.json) 列出本次保存的原始结果、合同、authority、核心实现和 reducer/validator 源码。

大型 `.pt`/`.npz` token 或中间特征缓存、旧检查点、原图、err/out/log 不随这次历史补档上传；历史路径和哈希仍留在原始工件里。特别是 PJ2 上游约 95.9 MB 的 PJ1 中间结果不在本次增量中。因此该增量足以查阅原始结论与逐查询结果，不宣称包含重新运行全部历史训练和前向所需的完整依赖。

任何旧文档中的“当前/待完成”均以该文档日期为准；本页对 BAG 完成状态和 V123 的协议处置给出了明确的后续更新。
