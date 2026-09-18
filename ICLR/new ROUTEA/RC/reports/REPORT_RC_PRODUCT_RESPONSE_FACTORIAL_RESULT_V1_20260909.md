# new HYP：乘积评分 × Q/R 响应的重训练析因结果（5138852）

同一 RAW C128、已打开内部 EVAL32 上，四个重训头形成清楚的顺序：基础 JOINT4 为 **26/32**，补入原 S 特征的 PRODUCT5 为 **27/32**，补入 Q/R 响应但不含 S 的 RESPONSE6 为 **26/32**，完整 ORIGINAL7 为 **28/32**。新增两种简化头都未保留原 28 个正确，简化候选门未通过；原完整模型仍被保留。

这次不能仅概括为“又一个 NO-GO”。预定析因比较得到两个具体正向变化：在不含响应的背景中加入 S 后，重训模型救回 `OUTCOME-0212`；在包含 S 的背景中加入 Q/R 响应后，重训模型保护了 RAW 原本正确的 `OUTCOME-0618`。两条变化均为 **1 个改善、0 个受损**。这为完整证据组合为什么优于所测简化头提供了内部解释，不等于新模型超过原成绩或普遍规律已获确认。

## 固定条件与四个模型

原 RAW ColNomic 自然 C128、RoMa soft visibility × ColNomic full-reference image-token MaxSim、全部 127 challenger、max-logit challenger 和零阈值 HOLD/SWITCH 全部固定。训练只使用 PAIR64 配对标签与 FULL TRAIN32 retrieval 身份标签；FULL TRAIN32 的 RAW 基线为 27/32，EVAL32 的 RAW 基线为 25/32。EVAL32 已反复打开，含 11 个 supergroup；其结果属于内部 development。旧 FROZEN_C 27/32、69/90 属于另一 head/split lineage。

各头从原 NATIVE7 六维特征逐 bit 抽取指定列，保留原 PAIR V1 roll-one、V2/FULL half 控制定义，使用原 PAIR_SIGN + FULL sign 损失、FP64 全零初始化、seed 17、AdamW、2000 更新。未引入新 tokens、标量值、matcher、阈值、训练标签或优化目标。没有任务空间标注，没有新增 encoder / RoMa forward。各头参数数不同，不能称为容量完全相等。

| 模型 | 原特征列（含 bias） | TRAIN32 | EVAL32 | EVAL 相对 RAW 救/损 | EVAL MRR |
| --- | --- | ---: | ---: | ---: | ---: |
| JOINT4 | RAW、M、L（4 参数） | 28 | 26 | 2/1 | 0.856845 |
| PRODUCT5 | RAW、S、M、L（5 参数） | 28 | 27 | 3/1 | 0.885789 |
| RESPONSE6 | RAW、M、L、Q/R 响应（6 参数） | 28 | 26 | 2/1 | 0.856845 |
| ORIGINAL7 | RAW、S、M、L、Q/R 响应（7 参数） | 28 | 28 | 3/0 | 0.901414 |

S 是原 joint local score，候选标量层面满足 S=M×L；实际头输入是原 symmetric 对比量。这里“加入 S”指加入已有原特征列 1，不是计算新的乘积特征，也不是增加额外底层图像信息。M 与 L 已参与同一个 scorer，输入和重新训练结果不能被简化为独立的“纯几何/纯内容”系统。

## 预定 2×2 比较完整报告

两个因素为“是否加入原 S 特征”和“是否加入原 Q/R 响应特征组”。Q 与 R 在本轮作为一组同时出现，尚未分别归因。

| EVAL32 | 不含 Q/R 响应 | 含 Q/R 响应 |
| --- | ---: | ---: |
| 不含 S | JOINT4：26 | RESPONSE6：26 |
| 含 S | PRODUCT5：27 | ORIGINAL7：28 |

| 预定条件增量比较 | 改善/受损/净增 | 等权 group 准确率差 | 变化样本 |
| --- | ---: | ---: | --- |
| PRODUCT5 − JOINT4：无响应背景加入 S | 1/0/+1 | +0.022727 | 0212 改善 |
| ORIGINAL7 − RESPONSE6：有响应背景加入 S | 2/0/+2 | +0.068182 | 0212、0618 改善 |
| RESPONSE6 − JOINT4：无 S 背景加入响应 | 0/0/0 | 0.000000 | 正确集合不变 |
| ORIGINAL7 − PRODUCT5：有 S 背景加入响应 | 1/0/+1 | +0.045455 | 0618 改善 |

所有 TRAIN32 配对比较的正确集合相同，均为 28/32。PAIR64 正确决策数为 JOINT4 50、PRODUCT5 50、RESPONSE6 51、ORIGINAL7 51，仅描述优化池，不当作独立泛化结果。两种简化头对 ORIGINAL7 的“保留全部正确并达到同样数量”检查均为 false；超过原模型的改进候选也均为 false。

该表展示了本固定训练协议下的条件预测增量。因为各头重新拟合全部参数、参数数也不同，不能把这些差值直接解释为固定网络中某个输入的纯直接因果效应，也不能仅凭 26/27/26/28 宣布统计学交互显著或普遍协同定律。

## 两个具体决策解释

`OUTCOME-0212` 的 RAW 正确 reference 位于 C128 第 20 名，RAW winner 为 physical row 2715。JOINT4 选择错误 row 814，RESPONSE6 选择错误 row 3965；PRODUCT5 与 ORIGINAL7 都选择正确 row 1024，并 SWITCH 到第 1 名。此处差异是 **challenger 身份选择**，不是所有头选同一 candidate 后的阈值差。

`OUTCOME-0618` 的 RAW winner 本来就是正确 row 3234。四头最强 challenger 均为错误 row 4659，最大 switch logit 如下：

| 头 | 最大 switch logit | 决策 |
| --- | ---: | --- |
| JOINT4 | +0.2278743343 | SWITCH，损坏 RAW 正确答案 |
| RESPONSE6 | +0.1422310321 | SWITCH，损坏 RAW 正确答案 |
| PRODUCT5 | +0.0531134848 | SWITCH，损坏 RAW 正确答案 |
| ORIGINAL7 | -0.0109729457 | HOLD，保留 RAW 正确答案 |

因此 ORIGINAL7 相对 PRODUCT5 的 +1 是 **减少一次错误切换**，不是相对 RAW 新救回一个原本错误的 query。原阈值始终是零，未调整阈值来制造此结果。这个内部样本靠近零边界，不能据此保证新数据上的稳定性。

四头的原完整 evidence C_BIND 均为 EVAL20/32，各自 REAL rescue 保留 0 个。这个对照扰乱完整 candidate evidence 绑定；不能把它直接当作 Q 或 R 单一坐标通道的效应，更不构成 ownership 证明。

## 当前可保留的解释及下一步边界

完整 reference-conditioned 证据中的 S 特征与 Q/R 响应组，在该固定训练协议和当前内部样本上承担了不同可观察的决策作用：S 可用的模型恢复了一个正确 challenger；响应组在 S 可用的背景中帮助完整头避免了一次错误 SWITCH。仅“RAW+M+L 加性读出”或“RAW+M+L 加响应而无 S”的简化均未复现完整效果。

这使 new HYP 的解释比“匹配质量与内容联合有用”更具体，但尚不能唯一归因 Q、R，也不能把重训后的整头变化等同于直接特征贡献。后续若继续拆分，变量已经限定为原 Q/R 响应各自的条件作用，或对上述整头变化进行明确区分的直接计算分解；不需要另换匹配器、重新定义 HYP 或启动空间 ownership。

本轮没有发布新部署模型，没有改变原 NATIVE7 参数，没有消费未授权 formal392 或新数据集，也没有把简化候选未过门升级为“所有 new HYP 理论失败”。

## 验证与完成收据

作业内独立进程状态为 `PRODUCT_RESPONSE_FACTORIAL_INDEPENDENT_REEXECUTION_PASS`，四头重新训练、所有输入和完整预测/比较全部重现。JOINT4 与 ORIGINAL7 两端参数、TRAIN/EVAL REAL/C_BIND 动作与上轮精确一致。

本次完成核对另行字面抽取原特征列，使用封存参数独立重算 **65,024 个 FP64 logit、512 份完整动作、256 个 PAIR 决策、14 项 TRAIN/EVAL 配对与 group 比较**，全部一致；候选绑定 rescue retention 和两个端点回归亦一致。没有调用 producer 的 predict/actions/summary/paired 函数，没有额外重训，没有 scheduler 调用。权威、预检、前序源、参数、join 前预测和结果文件的哈希闭合。

根对话提供实时 accounting：COMPLETED，ExitCode 0:0，8 CPU，MaxRSS 393464K，Berlin 2026-09-09 21:06:21–21:08:39，运行 2 分 18 秒。此 scheduler 来源与本次独立产物核验分别记录。

- [冻结计划](../plan/RC_PRODUCT_RESPONSE_FACTORIAL_V1_20260909.md)，SHA `1220b46a48c1f9e83d5b0d3bf440e2bd7da95081cb389f5843a3360450c78a0e`
- [执行授权](../registry/rc_product_response_factorial_authority_v1_20260909.json)，SHA `30109e4a00633970a3e2278edd1a0451c73d6fa08285dc0a328447bacca44b90`
- [原始结果](../results/rc_product_response_factorial_v1/result.json)，SHA `e9dae1e5837e72d3f38ba3e4473d0ff555b6ee90fadfbe3d3dbf64376759c0bf`
- [独立重训验证](../results/rc_product_response_factorial_v1/independent_validation.json)，SHA `f91c01caac1824ceb8cde358a9c22f95b94fe0d846965b35815c66cc778ed3bb`
- [参数](../results/rc_product_response_factorial_v1/parameters.json)，SHA `019cfeac14575b97371f66f81e5ad6ec55ec321af502a0b9110e4cc207c79b43`
- [全部 join 前预测](../results/rc_product_response_factorial_v1/eval_prejoin.json)，SHA `3ed9dd108c5e50c8ba9d84fdef1f10a06589301f19b3a07d950cb4f76afe4011`
- [完成收据](../registry/rc_product_response_factorial_completion_receipt_v1_20260909.json)
