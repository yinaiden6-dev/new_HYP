# new HYP：重训练证据通道充分性结果（5138847）

作业 `5138847` 完成，五个头均已在独立进程中重新训练并逐值复现。预定的简化主张 **未成立**：只保留 RAW、质量 M、内容 L 的 JOINT4 为 26/32，没有保留原 NATIVE7 的 28/32，也没有同时胜过两种重训单通道头。

同时，预定次比较得到应保留的正向内部证据：完整 NATIVE7 分别胜过重训 RAW+M 和 RAW+L，各为 **2 救 / 0 损**，等权 supergroup 准确率差都为正。不能把“JOINT4 简化失败”写成“完整联合证据没有价值”。本轮没有改进原最好模型，也没有宣布 population GO。

## 同一数据、候选和动作路径

所有结果使用原 RAW ColNomic 自然 C128、冻结 RoMa soft visibility × ColNomic full-reference image-token MaxSim 标量，固定全部 127 challenger 与零阈值 HOLD/SWITCH。FULL TRAIN32 的 RAW 基线为 27/32；已反复打开的内部 EVAL32 基线为 25/32，含 11 个 supergroup。PAIR64 是带标签优化池，其成绩不是额外验证集性能。旧 FROZEN_C 的 27/32、69/90 属于另一 head/split lineage，不在本表合并。

五头仅逐 bit 抽取原六维特征的不同列，使用原 PAIR_SIGN + FULL sign 损失、相同 FP64 全零初始化、seed 17、AdamW 和 2000 次更新。输入维度及参数数不同，已明确记录；没有更换 optimizer、损失、tokens、控制位移、candidate 或阈值。任务监督仅为 retrieval identity / pair labels，没有任务空间标注，也没有新 encoder / RoMa forward。

| 头（含 bias 参数数） | 原特征列 | TRAIN32 | EVAL32 | EVAL 相对 RAW 救/损 | EVAL MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| RAW2（2） | RAW gap | 27 | 25 | 0/0 | 0.828199 |
| RAW_PLUS_M3（3） | RAW gap、M | 28 | 26 | 2/1 | 0.856101 |
| RAW_PLUS_L3（3） | RAW gap、L | 27 | 26 | 1/0 | 0.849033 |
| JOINT4（4） | RAW gap、M、L | 28 | 26 | 2/1 | 0.856845 |
| ORIGINAL7（7，原 NATIVE7） | RAW gap、S、M、L、Q/R 响应 | 28 | 28 | 3/0 | 0.901414 |

这里的 M 头仍有 RAW ColNomic 内容先验；L 也是 visibility-conditioned 内容，不能分别命名为“纯几何”和“完全无几何”。S=M×L 是候选标量层面的关系；各头输入为原 symmetric 候选差，因此不能把 symmetric(S) 简写成 symmetric(M)×symmetric(L)。

## 预定比较与逐例变化

| EVAL 比较（前者相对后者） | 救/损/净增 | 等权 group 准确率差 | 改善/受损 group 数 |
| --- | ---: | ---: | ---: |
| JOINT4 − RAW_PLUS_M3（主比较） | 0/0/0 | 0.000000 | 0/0 |
| JOINT4 − RAW_PLUS_L3（主比较） | 1/1/0 | -0.022727 | 1/1 |
| JOINT4 − ORIGINAL7（保留原效果门） | 0/2/-2 | -0.068182 | 0/2 |
| ORIGINAL7 − RAW_PLUS_M3（预定次比较） | 2/0/+2 | +0.068182 | 2/0 |
| ORIGINAL7 − RAW_PLUS_L3（预定次比较） | 2/0/+2 | +0.045455 | 1/0 |

- JOINT4 与 RAW_PLUS_M3 的正确集合完全相同；这不意味着全部错误预测或 logit 相同。
- ORIGINAL7 相对 RAW_PLUS_M3 多正确 `OUTCOME-0212`、`OUTCOME-0618`；相对 RAW_PLUS_L3 多正确 `OUTCOME-0212`、`OUTCOME-0220`。
- JOINT4 相对 RAW_PLUS_L3 救回 `OUTCOME-0220`，但损坏 `OUTCOME-0618`，因此总数相同并不代表同一决策集合。
- ORIGINAL7 相对 RAW 保留全部 25 个正确，救回 `DIFFICULT-0044`、`OUTCOME-0212`、`OUTCOME-0220`。原参数及 EVAL REAL / C_BIND 全动作与冻结原结果精确一致。

原完整 evidence C_BIND 下，ORIGINAL7 为 20/32，原 3 个 rescue 保留 0 个。该对照扰乱的是完整 candidate evidence 绑定，不能单独解释成某一 Q/R 坐标通道的必要性，也不能转为 ownership 结论。

## 本轮定位到哪里

本轮排除了一个具体的简化解释：在固定原训练过程下，单用加性的 RAW+M+L 三个输入，无法重现完整六特征头的当前内部效果。完整 NATIVE7 胜过两个重训边际头，是正向的内部证据，但尚未区分收益来自 S 的乘积信息、Q/R 响应、它们与重训参数的组合，还是这些输入带来的有效容量差异。

因此下一步已有明确、有限的变量空间：JOINT4 保留原列 `[0,2,3]`，缺少的是 **列 1 的 symmetric(S)、列 4 的 query response、列 5 的 reference response**。可以固定其它全部训练和评价定义，对这三个原输入做完整 2³ 子集比较；两端分别必须精确回放 JOINT4 与 ORIGINAL7。本报告仅提出这个有边界的下一步，不代表下一实验已经提交或通过。

之前“在固定完整头中去掉某通道会失效”与本轮“去掉通道后允许其它参数重新训练”回答不同问题。前者检验当前计算的依赖；后者检验受限特征集合在该训练过程下的可替代性。两者不矛盾，不能把固定头失效直接升级为任何重训模型都无法替代的结论。

这些结果支持继续研究 **new HYP 的完整 reference-conditioned 联合证据校准**。它们没有证明严格空间 HYP、superregion 或 ownership，也没有因主比较失败而否定已经验证的检索增益。反复打开 EVAL32 及 11 个 group 的限制仍在，不能称作未触碰的总体可靠性确认。

## 独立完成核对与证据

作业内新进程验证状态：`RETRAINED_EVIDENCE_SUFFICIENCY_INDEPENDENT_REEXECUTION_PASS`，包括重新构造全部输入、重新训练五头、原参数/动作回放，以及完整比较复现。

本次完成核对额外用原特征列进行字面切片，从已封存参数独立计算 **81,280 个 FP64 logit、640 份完整动作字典、320 个 PAIR 决策和 16 项按角色配对比较**；直接重建 tie break、HOLD/SWITCH、目标排名、MRR、救/损和 group 计数，全部一致。没有调用 producer 的 predict/actions/summary/paired 计算函数，也没有再次训练。权威文件、预检及所有结果/封存文件哈希均闭合。完成核对没有查询 scheduler。

根对话提供的实时 accounting：COMPLETED，ExitCode 0:0，8 CPU，MaxRSS 390420K，Berlin 2026-09-09 20:51:21–20:54:04，运行 2 分 43 秒。该 scheduler 信息与独立产物核验分开记录。

- [冻结计划](../plan/RC_RETRAINED_EVIDENCE_SUFFICIENCY_V1_20260909.md)
- [执行授权](../registry/rc_retrained_evidence_sufficiency_authority_v1_20260909.json)，SHA `012542d9d25d0641187583a4e51f3f5193d8f4d1ef968342a65ebc734de60fe5`
- [原始结果](../results/rc_retrained_evidence_sufficiency_v1/result.json)，SHA `20102901242c734c2b5c0404e0e5571999ab1dc25ceefa5cee2f37f884a12601`
- [独立重训验证](../results/rc_retrained_evidence_sufficiency_v1/independent_validation.json)，SHA `ebac4a7d704e121d972f8999dbcbe767e588681e3f9a7352974f78d9e809e195`
- [参数](../results/rc_retrained_evidence_sufficiency_v1/parameters.json)，SHA `8d1781e0abaec6bcd2cdacb6f0637140398764ca1f533452f9bb1cf956aee20b`
- [全部 join 前预测](../results/rc_retrained_evidence_sufficiency_v1/eval_prejoin.json)，SHA `6cbd8aadf3df59ffe9aedc41179119298b18bf06e12ac107fa109377aa82e8bb`
- [完成收据](../registry/rc_retrained_evidence_sufficiency_completion_receipt_v1_20260909.json)
