# RoMa-RGH reference-first S0：最终结果与假设数量偏差审计

日期：2026-09-10。范围：已打开的 TRAIN32、12 supergroups、每条自然 RAW-C128，冻结零参数 RoMa reference-first max-tree + RAW-ColNomic V。最终作业：5139368。

## 1. 正式结论

`ROMA_RGH_REFERENCE_FIRST_NATURAL_S0_NO_GO`。

Slurm 为 `COMPLETED 0:0`，耗时 37 秒。独立 postjoin validator 为 `ROMA_RGH_REFERENCE_FIRST_NATURAL_S0_POSTJOIN_INDEPENDENT_VALIDATION_PASS`。本轮是固定机制的内部科学筛查失败；工程计算闭环已完成。

正式产物均保留：

- `results/roma_rgh_reference_first_natural_s0_v1/aggregate/postjoin_result.json`
- `results/roma_rgh_reference_first_natural_s0_v1/aggregate/postjoin_independent_validation.json`
- 对应匿名 aggregate、8 个 shard 的独立 replay、32 个 query checkpoint。

审计另行核对了 13/13 postjoin authority 文件哈希，并直接从已封存 aggregate 独立复算 32 × 17 = 544 组 target-key join、排名、own strongest-wrong 和 margin，均一致。所有 transition 分类覆盖完整 32 条。未发现本次结果由 target 位置错接或排序方向写反造成。

## 2. 同一数据、同一候选轴上的结果

以下四项均属于本次 TRAIN32 S0 的冻结 readout；这些数字不得与旧 EVAL32 FROZEN_C/NATIVE7 或 difficult90 的 action 成绩混用。

| 本次 readout | 正确数 | MRR | 地位 |
| --- | ---: | ---: | --- |
| RAW-ColNomic all-patch × full-reference | 27/32 | 0.88916 | 同源内容对照 |
| candidate-independent query-only family × full-reference | 26/32 | 0.86874 | 预注册比较基线 |
| RoMa-RGH query-region × full-reference | 7/32 | 0.31660 | 主臂 |
| RoMa-RGH paired query/reference region | 5/32 | 0.23718 | 诊断臂 |

主臂相对 query-only：0 rescue / 19 break，净增 -19，另有 5 条 wrong-to-different-wrong。相对 all-patch：0 rescue / 20 break，净增 -20。这里未执行既有 HOLD/SWITCH action，也未替换既有成功检索路径。

| 冻结门 | 观测 | 阈值 | 结果 |
| --- | --- | --- | --- |
| target legal-H1 | 31/32 | ≥26/32 | PASS |
| primary strict wins | 7/32 | ≥21/32 | FAIL |
| primary 相对 query-only 净增 | -19 | ≥+4 | FAIL |
| C_P_BIND 正向 margin-drop | 15/32；等组均值 +0.03603 | ≥21/32 且等组均值 >0 | FAIL |
| P_QUERY_COORD 正向 margin-drop | 7/32；等组均值 -0.09956 | ≥21/32 且等组均值 >0 | FAIL |
| P_REFERENCE_COORD 正向 margin-drop | 29/32；等组均值 +0.17929 | ≥21/32 且等组均值 >0 | PASS |

31/32 target 有合法连通区域，仅证明几何可用性。每条 query 有 109–128 个候选可形成合法 H1，因此合法性本身没有足够的身份选择性。没有人工 target mask，本轮没有测量完整药盒分割覆盖率或 IoU。

## 3. 已定位的主要数学问题

固定 core 的 `h0_marginal` 使用温度 0.1、H0 prior 0.5、总 slot 容量 768。所有无效/H0 slot 分数为 0，因此原公式可精确改写为：

\[
S_g=0.1\log\left[1+\frac{N_gQ_g}{1536}\right],\quad
Q_g=\frac{1}{N_g}\sum_{h\in H1_g}\left(e^{v_h/0.1}-1\right).
\]

`N_g` 是合法 H1 数量，`v_h` 是该区域的 RAW-ColNomic mean MaxSim。`Q_g` 是每个合法 H1 的平均指数超额分数，不能直接解释成校准概率或普通 patch 质量。

固定 768 只统一总容量，并未消除不同候选合法 H1 数量的影响。当区域分数为正时，增加同分 H1 就提高候选分数。max-tree 中的嵌套/重叠节点分别分配权重；相同或相近的局部内容可能通过多个节点累计。query-only 的同一 family 被所有候选共用，候选间没有这一 `N_g` 差异。

只用已封存分数与 H1 数量即可逆算 `Q_g = 1536 × expm1(S_g/0.1) / N_g`。本次事后诊断未生成新的 C128 排名，未改原分数，未再次调用模型：

- target 合法 H1 数中位数 36，当前 strongest-wrong 中位数 376.5（标准中位数）。
- 30/32 条 strongest-wrong 的 H1 数更多；25/25 个主臂错例均如此。
- 在双方有 H1 的 31 条中，target 的 `Q_g` 更高：24/31。
- 25 个主臂错误可按当前固定 strongest-wrong 分解：17 个 target 的 `Q_g` 更高却因数量较少落败；7 个 target 的 `Q_g` 也较差；1 个 target 无合法 H1。

这证明候选间假设数量项足以解释 17 个已观察到的 pair 排序反转。它不证明修改归一化后能救回 17 条：改变评分后可能出现第三个错误候选。`24/31` 也不是新的全 C128 准确率。

可重算的逐 query 数值保存在同目录 `roma_rgh_s0_job5139368_h1_multiplicity_audit_20260910.json`。

## 4. 控制结果的解释限制

`C_P_BIND` 保留目的候选 reference 内容，只错绑 proposal。它同样为 7/32，REAL 相比它有 5 个正向纠正和 5 个反向破坏；15/32 margin-drop 为正。故本轮没有证明正确 reference 所生成的区域能稳定优于错绑区域。

`P_QUERY_COORD` 导致 4094/4096 个候选没有 H1、30/32 条 query 的整个 C128 都为 H0；target 32/32 全部 H0。控制正确数为 0/32。其 margin 接近零，在算术上可能大于负的 REAL margin，但不能说“打乱空间更会找 target”。这项失败不能独立推出“RoMa 中没有空间信息”。

`P_REFERENCE_COORD` 的 margin-drop 门通过，同时合法 H1 可用性/数量也改变。它提供扰动敏感性证据，却不能单独证明 target 定位或空间 ownership。

H0=0 是当前分数约定；cosine MaxSim 没有经过自然匹配/不匹配似然比校准，因此不能把当前 H0 混合分数解释为已验证的 no-match posterior。

## 5. 机械裁决与后续修正边界

遵循原冻结 S0 合同，关闭本次 `reference-first max-tree + fixed-H0 marginal + RAW V` 家族；`next_authorized_stage=null`，不签 P0，不打开 EVAL32，不在本 panel 上扫描 pooling、阈值或温度。

若继续研究 successor，最先需要解决的是假设证据的计权定义：

1. 同一物理证据不应仅因树节点枚举更密而增加候选优势；需先建立对等价区域重复/枚举粒度变化的评分约束。
2. 把区域可用性、区域内部内容与合法假设数量分别报告，不能再把三者的变化全归因于 candidate binding。
3. 即使解决计数项，仍须面对本轮 7 个内容项较差及 1 个 target-H0 的错例。需要自然 query/reference/near-rival 监督或新的表示证据，并单独验证其作用，不能承诺只改归一化即可通过。
4. 新评分必须重新进行完整 target-free C128 竞争，比较各自 strongest-wrong 与 query-only/all-patch；不能用固定旧 rival 的事后分解作为新模型的 GO。

上述是下一方案的设计要求，尚未实现、训练或提交。本次封存的负证据与已有成功的 RAW+RoMa action 证据分别保留。
