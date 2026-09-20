# H593 六项证据消融 V1

2026-09-20 用户授权补齐。新增历史开发诊断，独立于旧 authority 的执行期限；不修改旧代码、头、预测、候选集或原结论。

- 人口：原 H593、68 identities、64 source/identity components、原 grouped OOF5；自然 RAW C128，不插入目标。所有 593 留出图进入分母，包含 23 个 C128 target-absent；训练沿用原规则排除无定义正项的 absent 行。
- 特征固定依次为 RAW、S、M、L、Q、R。S/M/L 及 Q/R 有计算耦合，删输入列不等于从照片中删除所有几何或内容信息。
- COST1 为主，CE 为对照；不按消融结果更换部署头。每模型包括 FULL、六个 ZERO（冻结完整头，仅将对应权重置零，bias 和其它参数不变）、六个 REFIT（实际删掉一列，五权重加 bias，从零重新训练）。不扫描其它子集、阈值或步数。
- 所有重训使用原训练行顺序、FP64、seed17、AdamW lr=.03/weight_decay=.001、2000 updates，8 CPU threads，末点参数；直接调用原训练函数。完整头每折重放一次，参数和全部 held-out logits 必须与封存 COST1/ALL_CE 逐位一致。
- 五折 × 六删项 × 两损失 = 60 个删项头，另 10 个完整头重放；每折在新进程重复训练并逐位复核，NumPy 独立重算参数读出和动作。
- 严格保持完整 127 challenger，最大 logit >0 才 SWITCH，否则 RAW/HOLD=0；同分沿用 canonical challenger physical order。
- 训练进程只读取当前折训练身份；held-out identities/curator 在五折预测全部封存验证后才能由 joiner 读取。禁用 D1-MI、formal392、外部集、原 EVAL 面板及新编码器/RoMa前向。
- 汇总报告正确数、对 RAW 及对应 FULL 的 rescue/break、动作改变数、MRR、分折值、64组等权 FULL−ablation 差和 bootstrap95%（10000，seed20260920）、单侧组 sign-flip（9999）和 Holm 修正。每模型×干预方式的六个删项为一个预定多重比较族。H593 已开发，区间/p 值仅作补充开发描述，不当新外部确认。
- 冻结置零衡量现成函数依赖；重训删项衡量该固定学习配方下的替代能力。两者均不证明普遍不可替代性、最小充分集或像素因果/ownership。
- 执行：accelerated，五折并行，每折 8 CPU/16GB/1GPU（实际 CPU 计算）、10分钟上限；汇总依赖全部折成功。复用已有约束，不重复计算 tokens/匹配。
- 产物：每折原始127分数与参数 JSON、fresh-replay/NumPy验证、预标签总封存、逐查询结果、Markdown报告、独立汇总核算。最后更新 GitHub 和当前完整结果文档/ZIP，保留旧内容。
