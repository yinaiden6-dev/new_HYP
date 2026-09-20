# Route A 项目阶段性研究汇报（教师版）

日期：2026-08-26  
项目：Reference-Defined Exact-Instance Recognition / Route A RC  
当前结论等级：阶段性机制与工程进展；尚无自然场景、目标未知、未见身份、全图库检索的最终科学 GO

## 一、建议先向老师这样概括

本项目研究一个比普通分类更严格的问题：给定一张盲人自然拍摄、可能有遮挡、背景干扰或多个药盒的 query，以及完整 gallery 中每个药盒的 reference，模型要识别具体是哪一个药盒。训练时允许使用 query-reference exact identity pair，但不允许人工目标框、mask、点、oracle 候选插入或 identity-specific 参数；测试身份在训练中未见，但测试时有 reference。

现阶段最重要的结论不是“ownership 已经成功”，也不是“局部信息不存在”，而是：

1. 照片到 reference 的域差是第一层主要瓶颈，D1 域对齐和 rank-only 候选排序确实能稳定改善检索；
2. reference-conditioned 的候选绑定和局部证据确实存在，但旧的 top-k、固定 readout、局部均值、坐标归一化和 selector 无法把它稳定转成 target-free 决策；
3. 真正未解决的是：在正确 reference 自然进入候选集后，如何由该 reference 生成 query 中的目标假设，并让它在另外 127 个自然强错候选的最大值竞争中胜出，同时不破坏原本正确的结果；
4. 最新 RGH 路线正针对这一缺口：让 reference 生成带完整来源链的候选区域，以自然 exact-pair 检索监督学习软匹配，再由独立 verifier 验证。该路线已经通过工程门、两查询自然可学习性门和四折训练首段，但尚未打开 32-query P0 panel，因此没有新的 Top-1、MRR、rescue/break 或科学 GO。

一句最稳妥的话是：

> 我们已经把问题从“是否存在 ownership”收窄为“如何在 target-free 的强候选竞争中形成可验证的 reference→target 假设并进行 no-regret 决策”；旧机制的失败边界已厘清，新机制已经进入正式训练，但最终科学结论尚未产生。

## 二、研究目标和论文问题

输入为自然 query 图像 \(x\) 与完整 reference gallery \(\{g_j\}\)。模型应在不知道正确身份的情况下，对完整 gallery 打分，并识别训练阶段未见过、但推理阶段有 reference 的 exact instance。

核心假设是：完成照片到 reference 的域对齐后，模型还能通过以下能力获得额外检索增益：

- reference-supported foregroundness：哪些 query patch 能被某个 reference 解释；
- candidate-relative exact-instance evidence：同一局部更支持候选 \(g\) 还是近邻竞争者 \(c\)；
- part-to-whole correspondence：自然照片中的局部可见部分如何对应完整 reference；
- counterfactual attribution：打乱 candidate binding 或空间关系后，新增收益应消失；
- no-regret action：证据不足时 HOLD，证据充分时才 SWITCH。

这里的 ownership 不再定义成“先生成一张主体分割图”。更准确的顺序是：

```text
全图库检索先验
→ 自然候选集
→ 每个 reference 生成 candidate-conditioned target hypothesis
→ 独立证据验证 hypothesis
→ 候选间竞争确定 target
→ ownership 解释获胜所依赖的区域
→ 仅在 no-regret 门通过时修改检索结果
```

因此，ownership 是 reference 与 query 之间的候选相对解释关系，而不是 query-only 前景分割。

## 三、当前完整实现路径

### 1. 基础检索与域对齐

ColNomic 作为冻结的原始基线，对自然 query 和完整 gallery 打分。D1 学习 fold-local 的 photo→reference 非对称域对齐；A10/rank-only 在冻结的 D1 表征上继续学习候选排序。它们提供 candidate prior，但不宣称空间 ownership。

在 difficult224 的固定开发结果中：

| 模型 | R@1 | 含义 |
|---|---:|---|
| RAW ColNomic | 143/224 | 原始全图检索 |
| D1 | 155/224 | 域对齐有效 |
| A10 / matched rank-only | 157/224 | 候选相对排序再增 2 |
| A20 local-star | 157/224 | 显式局部路径未超过 A10 |

这说明 D1 和 rank-only 不是 ownership 的对照之外对象，而是已经学到部分“全局/候选级 ownership”的基础；新的空间机制必须超过相同容量的 A10，而不能只超过 RAW 或 D1。

### 2. reference-conditioned proposal

对每个候选 reference 独立提出假设：“如果我是正确 reference，query 中哪些连通区域能由我解释？”单 patch 只能作为 seed，不能独立成为目标证据。有效 hypothesis 必须由多个 patch 支持，并保持 query/reference 双端连通、几何相容和 candidate provenance。

### 3. 独立 verifier

Proposal 通道只负责生成候选区域；Verifier 必须读取未参与 proposal 的证据，比较同一 query 上不同 reference 对其假设的解释能力。当前 V124 固定三臂：

1. `ALL_PATCH_SAME_MODEL`；
2. `QUERY_MULTITILE × FULL_REFERENCE`；
3. `QUERY_MULTITILE × LOCAL_COMPONENT_SET`。

同时分离三种 family：REAL、完整 C128 candidate-binding destruction、pre-P ColNomic binding destruction。只有正确 binding 和正确空间都优于 matched null，才可宣称 candidate-specific spatial evidence。

### 4. target-free C128 competition

所有候选在 role join 之前匿名计算；target 不插入，若 target 自然不在 C128，则记录为 absent。训练标签只进入 listwise retrieval loss，不能进入模型输入或 patch 选择。评价时必须使用每个模型自己的 strongest wrong candidate，不能用给定 target-vs-donor pair 替代 127-way 最大错误竞争。

### 5. no-regret HOLD/SWITCH

只有 proposal、verification、candidate-binding 和 spatial-null 均通过后，才训练小型 action layer。它必须优先保护 RAW/D1 已正确样本，并显式输出 HOLD；不能用全局加分或固定 scale 强制修正全部 query。

## 四、实验历程与目前证据

| 阶段 | 真实问题 | 关键结果 | 当前效力 |
|---|---|---|---|
| Domain-first / D1 | photo-reference 域差是否主导 | 域差诊断约造成 0.333 R@1 成本；D1 在 difficult224 从 143 提至 155 | 有效开发证据 |
| A10/rank-only | 候选相对排序是否可学 | difficult224 达 157/224 | 有效 matched 基线 |
| C6d A20 | 显式 local-star 是否增加新空间证据 | A20=157/224；相对 A10 为 0 rescue/0 break | 该实现 NO increment |
| C6d-E/F1 | patch 选择和 selector 是否可部署 | real 158、patch permutation 161、candidate shuffle 154；Oracle 可救 44 个但实际只救 4；F1C 五臂全部 abstain | 候选绑定存在，空间选择/行动失败 |
| R4/R5 | 正确 reference 能否进入候选集 | R4 46/62=74.19%；R5 194/235=82.55%，均低于 90% | candidate-recall stop，不是 ownership 否定 |
| D32/M0/S8 | 固定 DINO/H0/readout 能否选对 target | D32 18/32→14/32；M0 正方向 54/190、Top-1 0/190；S8 65/92→66/92，6 rescue/5 break，binding null 几乎不变 | 固定 readout/代理似然 NO-GO |
| H0 pairwise | 给定 target-rival 是否有相对证据 | U4 为 90 rescue/56 break；但完整 C128 PJ2 仅 2 rescue/1 break，Top-1 增益 +0.001417，p=0.2438 | 证明 signal≠deployable decision |
| V123→V124 | 三臂和控制是否语义正确 | V123 有完整 C128 C_DINO、FULL/LOCAL 混杂和缺 pre-P C_COL_P 三项协议缺陷；V124 Core/Phase-A/Phase-B 工程独验 PASS | 修正实验协议，尚无新科学结果 |
| GX-CBNR R0 | 连通 DINO relational energy 是否形成几何排他性 | 16/16 RAW-correct 保留，1 rescue/0 break；但 wrong 仅 1/16 转正，REAL−P_COORD=-0.821，CI 全负 | 正式窄范围科学 NO-GO |
| RGH | reference 生成区域并自然训练 assignment 是否可行 | E0、自然 S0、P0 manifest 和四折 Stage-0 独验 PASS；尚未读取 panel | 当前主攻路线，只有工程/训练进展 |

## 五、已经成立的积极结论

### 1. 项目不是“没有信号”

D1 和 A10 的增益、candidate shuffle 后性能下降、C5/C6 中少量真实 rescue，以及 target-known Oracle 的较高可恢复上限，都说明 query-reference 候选关系存在。C6d-E 中 Oracle 在 59 个 A10 错例里识别出 44 个可恢复案例，而实际 readout 只救 4 个，证明主要问题不是容量绝对为零，而是 target-free 选择和 readout 不可靠。

### 2. 关键瓶颈已经被实验而非直觉定位

旧实验反复显示：

- 给定 target/rival 的 pairwise 方向可以改善；
- 一旦进入完整 C128，其他 127 个错误候选的最大值会吞掉增益；
- candidate binding 被破坏时信号常下降，说明 reference identity 进入了表示；
- spatial permutation 有时不降反升，说明旧 patch map 并非可靠空间因果解释；
- selector 要么到处出手产生 break，要么退化为全部 abstain。

因此瓶颈已经从笼统的“图不够好”收窄为“target-free strongest-rival decision + spatial causal evidence + safe action”。

### 3. 评价协议比早期可靠得多

当前链路已经形成以下硬约束：完整自然 C128、零 target insertion、prejoin 匿名封存、identity/supergroup-disjoint folds、candidate-binding 与 spatial destruction 分开、matched same-capacity control、rescue/break 分报、独立 validator、hash/byte replay、fresh/resume 校验。这些约束防止把 target-known、synthetic、pairwise 或工程 PASS 误报成论文级成功。

### 4. 最新 RGH 是针对根因的机制变化，不是又换一个 pooling

RGH 不新增 backbone 或 selector head，只训练继承的 4,357 参数 assignment：

```text
specific reference atom
→ query/reference 双向 soft assignment + reliability + dustbin
→ reference endpoint cross-fit affine
→ reference cell area 投影到 query
→ REAL 与 matched coordinate null 的有符号差
→ 仅保留双方向为正的 visible atoms
→ query/reference 双端接触且 affine 相容后合并
→ frozen selected superregion
```

它直接修复两个历史缺口：reference 以前不能生成新的 query geometry；relative-only loss 没有可识别的绝对零点。RGH 用 matched coordinate null 把零点固定为 0，并让没有稳定对应的 reference atom 进入 dustbin/H0。

当前 RGH 的硬进展为：

- E0：`REAL>P_TRAIN` 8/8、`REAL>P_EVAL` 8/8，4,357/4,357 梯度有限非零；
- 自然 S0：两个完整 C128 查询的 loss 有限，fresh/resume、optimizer、RNG 和 full-C128 VJP 独立重放闭合；
- P0 manifest：600 queries，其中 594 target-present、6 absent；32-query panel 含 32 个独立 identity/supergroup，四折各 8；
- V4 Stage-0：四折各完成 64 updates，共 256/1782；四份 producer 和四份独立 validator 全 PASS；panel read=0。

这些只证明机制可算、可学、可恢复，尚不证明 target discovery 或 retrieval improvement。

## 六、必须主动向老师说明的消极结果

### 1. 尚无最终科学 GO

目前不能宣称：

- 自然未见身份 full-gallery ownership 已建立；
- 模型已稳定找到 query 中的真实 target；
- DINO/GX 或 RGH 已提升完整图库 Top-1；
- HOLD/SWITCH 已可部署；
- opened/sealed 或外部数据已完成论文级确认。

### 2. GX 正式 R0 是有效科学 NO-GO

Balanced32 上，GX 保留了 16/16 个 RAW-correct，并有 1 rescue/0 break，这是安全性的微弱正面信号；但 16 个 RAW-wrong 仅 1 个转正。correctness increment 的 group-balanced mean 为 +0.0370，95% CI `[0, 0.1111]`；target-vs-rival 为 +0.196，CI 跨 0；REAL 相对 P_COORD 为 -0.821，95% CI `[-1.171, -0.477]`。因此该 connected DINO relational-energy 形式没有形成可部署的几何排他性。

这否定的是该固定 GX-CBNR R0 形式，不是否定所有 reference hypothesis 或 ownership 理论。

### 3. candidate recall 是独立上限

若正确 reference 不在 C128，后面的 proposal/verifier 无从纠错。历史 R4/R5 均未达到 90% 门；RGH manifest 中也有 6/600 target absent。短期可以先在 target-present 条件下验证 hypothesis/verifier，但最终 full-gallery 系统仍必须补独立候选生成能力。

### 4. 真实照片上的区域仍可能不完整或被 generic 局部吸引

旧 ColNomic region 往往能找到“像这个药盒的一小块”，但也会吃背景、文字块或边缘，错误 candidate 同样能形成合法区域。早期 all-legal merge 又会通过传递连接膨胀为接近整张 reference 的大区域。RGH 的 visible-atom matched-null 与双端合并正为解决此问题，但尚无 P0 数字证明已经解决。

### 5. 工程链过长，曾多次遮蔽科学问题

历史中存在 cache、authority、hash、完整候选轴、GPU/CPU 迁移、resume、control 语义等工程错误。V123 的旧 NO-GO 也因三项协议缺陷被降级为 diagnostic。现在的独立验证和 append-only authority 必要，但后续应减少重复 freezer/launcher 版本，把科学漏斗保持为：E0 → 小规模自然 S0 → P0 → V → action，不再在同一 opened endpoint 扫描 readout。

## 七、核心创新点

### 创新 1：从“预测 ownership map”改为“reference 生成 target hypothesis”

每个候选 reference 都提出一个关于 query 的可检验假设；target 是候选假设竞争的结果，ownership 是胜出后的解释。这比 query-only foreground 或先验 mask 更符合 reference-defined recognition。

### 创新 2：无人工 mask 的 latent superregion 学习

以 exact identity retrieval loss 监督整个 candidate bag，区域/atom 保持 latent；使用 reliability、dustbin 和 H0 允许背景或无法对应部分不表态。理论上属于 multiple-instance learning 与 partial matching 的结合。

### 创新 3：双端连通和完整 provenance

区域不仅在 query 侧连通，也必须在 specific reference 侧连通；每个 query cell 都能追溯到 candidate physical row、reference atom、reference cell、assignment、affine 和投影。它防止把彼此无关的高分 patch 拼成“目标”。

### 创新 4：matched-null 定义可识别零点

旧 pairwise loss 只能学相对排序，无法判断某个 hypothesis 是否真正高于“无证据”。RGH 比较 REAL 和保留统计量但破坏坐标对应的 `P_TRAIN`，以差值 0 作为冻结可见性边界。

### 创新 5：Proposal/Verification/Action 三层因果解耦

P 负责提出区域，V 用独立证据验证，Action 只在 C_BIND、P_COORD 和 no-regret 门通过后才决定 HOLD/SWITCH。候选重排等变性和 candidate-binding destruction 被明确区分。

### 创新 6：面向论文可信度的 target-free 实验设计

完整 C128 先算后 join、零 oracle insertion、identity/supergroup OOF、strongest-wrong 竞争、same-capacity baseline、严格 ABSTAIN、独立重放和 untouched endpoint，使最终结论能回答真实部署问题，而非只回答“已知 target 时能不能找 patch”。

## 八、当前瓶颈，按优先级排序

### 瓶颈 A：正确 reference 在强错竞争中不一定胜出

这是当前首要瓶颈。pairwise target-vs-donor 信号不能替代 C128 的 extreme-value competition。必须在匿名完整候选轴上证明 target rank、margin 和每折 rescue>break。

### 瓶颈 B：新证据是否真的依赖 specific reference 和空间关系

如果 C_BIND 或 P_COORD 后得分不降，模型可能仍在读 candidate-common query statistics、颜色/文字强度或分数分布，而不是 reference-specific geometry。

### 瓶颈 C：区域完整性与模拟到真实照片的断层

合成仿射、同图 warp 或 target-known assay 容易成功；自然照片存在透视、遮挡、反光、背景、多药盒和完整 reference 与局部可见 query 的域差。训练必须依赖自然 exact pairs 与自然 hard negatives，而不能让 synthetic geometry 承担自然 likelihood。

### 瓶颈 D：安全动作

即便平均 margin 增加，也可能破坏大量原本正确样本。需要显式 H0/HOLD、正确样本保护、wrong-to-wrong 统计和 no-regret loss；不能只优化 full-gallery CE 或统一 residual scale。

### 瓶颈 E：candidate recall

目标 absent 是结构性不可恢复错误。它与 target selection 分开评估：先证明 target-present 条件下能选对，再引入独立全图库候选生成器补 recall，不能用 oracle 插入掩盖。

### 瓶颈 F：统计量和数据规模

当前自然 identity 数量和 hard wrong episode 有限，部分 panel 已经 opened。开发结果只能用于机制筛查；论文级结论还需要未参与方案选择的外部或 sealed identity-disjoint endpoint。

## 九、当前状态与下一步决策树

### 当前状态

- V124：Core E0、C_COL_P Phase-A、Phase-B ALL_PATCH 已独立工程 PASS；正式 E1 natural target-free producer 尚未完成。排队中的 Job 5110283 只是不可晋升 smoke，即使完成也不是科学结果。
- GX：正式 R0 已产生窄范围科学 NO-GO；R1Q/R1P 修复仍不足。后续 R2A 只是开发可行性诊断，不应覆盖 R0 结论。
- RGH：E0、S0、P0 manifest、V4 四折 Stage-0 已独立 PASS；V5 continuation 目前只是工程设计/graph ready，尚无 execution authority、launcher 或已提交 RGH job；`UNRESOLVED_PARENT_COMPARATOR` 也仍阻止最终 P0 reducer authority。

### 推荐的唯一主线

1. 不重跑已完成的 V4 updates 0–63；建立 V5 execution authority，从四个已验证 checkpoint 精确续训剩余 1,526 updates；
2. 完成四折后先冻结 checkpoint，解决并冻结 parent comparator/reducer authority，再生成 32-query、完整 C128、target-free prejoin ledger；
3. 只做预注册 P0：coverage、target-vs-strongest-rival、C_BIND、P_COORD、rank improvement/degradation；
4. P0 任一核心门失败，停止 RGH，不在同一 panel 改 top-k、阈值、temperature、atom cap 或 pooling；
5. P0 通过后，才接入已修正的 V124 三臂 verifier，比较 all-patch、query-region×full-reference、paired-region；
6. P/V 因果门通过后，才训练 HOLD/SWITCH，并与冻结 ColNomic/D1/A10 比 rescue、break、wrong-to-wrong；
7. 内部 opened GO 后，再使用一次 untouched identity-disjoint endpoint；最后补独立外部数据验证。

### 成败含义

- 若 RGH P0 通过：首次证明自然 exact-pair 监督能够让 reference 生成 candidate-bound、coordinate-sensitive、可在 C128 中排序的连通 target hypothesis；这是接入独立 verifier 的充分理由。
- 若 P0 coverage 失败：reference→query assignment/表示仍不能形成真实目标区域，应更换表示或增加自然数据，而不是调 readout。
- 若 coverage 通过但 rank/C_BIND/P_COORD 失败：区域可生成，但不是 exact-reference 判别证据；停止空间 ownership 主张。
- 若 P/V 通过而 action 失败：瓶颈转为决策校准与数据覆盖，而不是 target hypothesis 本身。

## 十、建议的 3 分钟口头汇报稿

老师，我的项目研究的是 reference-defined exact-instance recognition：输入不是固定类别，而是一张自然拍摄的 query 和完整 reference gallery，要求识别训练中未见过、但测试时有 reference 的具体药盒。我们不使用人工框、mask、点，也不把正确候选 oracle 插入。

前期实验首先确认照片到 reference 的域差是主要瓶颈。D1 域对齐把 difficult224 的 R@1 从 143 提到 155，rank-only 又到 157。但显式局部 ownership 路径没有超过 matched rank-only。进一步的因果审计发现，candidate binding 的确有信息，但空间 patch 打乱后性能不降，有时反而更高；说明模型知道“哪个候选整体更像”，却不知道“哪块空间证据足以推翻当前 winner”。

之后我们把问题重新定义为 reference-conditioned target hypothesis：每个 reference 独立提出 query 中可能由它解释的连通区域，再用独立 verifier 竞争，而不是先预测一张 ownership map。历史 fixed DINO/H0 和 GX 机制已经给出正式负结果：它们能保留正确样本，偶尔 rescue，但不能稳定胜过 candidate-binding 和空间 null。这个负结果把瓶颈定位到了 target-free strongest-rival decision，而不是“完全没有局部信息”。

目前的新主线 RGH 不再增加复杂 head，而是训练 4,357 参数的共享 query-reference assignment。reference atom 经双向软匹配、cross-fit 几何、matched coordinate null 和双端连通，生成带完整 provenance 的 superregion。工程 E0、两查询自然 S0、600-query manifest 和四折训练首段都已通过独立验证，但 32-query P0 panel 尚未打开，所以我不会把它说成检索成功。

下一步是完成四折训练并一次性做完整 C128 P0。如果 target 在 strongest wrong 竞争、C_BIND 和 P_COORD 控制下都通过，再接独立 verifier 和 HOLD/SWITCH；如果失败，就停止这条具体机制，转向表示或自然数据，而不继续在同一 panel 调阈值。当前贡献是：明确了问题、建立了严格 target-free 评价体系、排除了多类伪 ownership，并把最终瓶颈收敛到一个可证伪的新机制上。

## 十一、老师可能追问的问题

### “你现在到底成功了吗？”

域对齐、rank-only、工程链和若干候选绑定机制已成功；GX 已得到有效窄范围科学 NO-GO；最终自然 target-free full-gallery retrieval 尚未成功。RGH 已进入四折正式训练流程并完成首段，但续训尚未取得执行 authority，也尚未产生 P0 科学结果。

### “做了这么多为什么一直没有 GO？”

因为早期很多方法只在已知 target、给定 pair、synthetic warp 或小候选上有效。完整部署需要同时满足 target 自然进入候选集、在 127 个强错候选中胜出、空间/绑定控制下降、且不破坏原正确样本。以前的机制通常只满足其中一两项。现在这些条件已经拆开，并用 stop rule 防止继续堆模型。

### “这些负结果有价值吗？”

有。它们证明单纯 top-k patch、query-only structure、固定 DINO matcher、共同 bias、pairwise rescue 和强制 residual 不能替代 target-free exact retrieval；并证明 candidate binding 有信号而 spatial actionability 不足。这直接决定了新机制必须让 reference 生成 hypothesis、使用 matched null，并接受完整 C128 strongest-rival 检验。

### “创新到底在哪里？”

创新不在再换一个 backbone，而在任务分解和学习/验证闭环：reference 生成 latent target hypothesis；无 mask 的 assignment+dustbin+MIL；query/reference 双端连通与 provenance；proposal/verifier/action 解耦；candidate-binding/spatial matched null；以及 full-C128 target-free、no-regret 的评价标准。

### “下一步多久能知道方向对不对？”

不应按排队时间承诺。科学上只剩清晰的两级判断：先完成 RGH 四折与 32-query P0；P0 通过才接 V124。P0 会直接回答 reference→target hypothesis 是否突破了历史瓶颈，不需要先完成整套 full-gallery 论文实验。

## 十二、证据入口

- C6d：`ICLR/new ROUTEA/results/route_a_c6d_local_target_star_selective_router_v1/formal_job5038839/result.json`
- C6d-E：`ICLR/new ROUTEA/results/route_a_c6de_zero_training_residual_readout_closure_v1_1/formal_job5039134/result.json`
- F1C selector：`ICLR/new ROUTEA/results/route_a_c6df1c_target_free_crossfit_selector_v1/formal_job5041811/result.json`
- GX formal R0：`ICLR/new ROUTEA/RC/results/dino_rcde_gx_formal_r0_science_v1/result.json`
- GX independent validation：`ICLR/new ROUTEA/RC/results/dino_rcde_gx_formal_r0_science_validation_v1/result.json`
- V124 Core E0：`ICLR/new ROUTEA/RC/results/dino_rcde_track_r_v124_core_e0_v1/result.json`
- V124 Phase-B：`ICLR/new ROUTEA/RC/results/dino_rcde_track_r_v124_c_col_p_phase_b_all_patch_e0_validation_v1/result.json`
- RGH E0：`ICLR/new ROUTEA/RC/results/cw0_rgh_xf_v2_frozen_selected_superregion_e0_v1/result.json`
- RGH natural S0：`ICLR/new ROUTEA/RC/results/cw0_rgh_xf_v2_natural_s0_exec343_38_v1/result.json`
- RGH P0 manifest：`ICLR/new ROUTEA/RC/results/cw0_rgh_xf_v2_p0_a0_manifest_v2/result.json`
- RGH V4 Stage-0：`ICLR/new ROUTEA/RC/results/cw0_rgh_xf_v2_p0_fold_training_v4/fold{1..4}/segment00/`
- RGH V4→V5 disposition：`ICLR/new ROUTEA/RC/results/cw0_rgh_xf_v4_to_v5_cross_resource_disposition_v2/result.json`
- RGH P0 contract：`ICLR/new ROUTEA/RC/plan/CW0_RGH_XF_V2_FROZEN_SELECTED_SUPERREGION_NATURAL_P0_CONTRACT_V1_20260825.md`

## 十三、最终边界

本项目当前最强、最诚实且积极的表述是：

> 已经证明域对齐和候选绑定能够改善或影响 exact-instance retrieval，也完成了 reference-conditioned target hypothesis 的严格实现与多层反事实评价框架；同时，正式实验表明旧的局部 readout、selector 和 GX 几何能量尚不能形成可部署的 target-free 决策。最新 RGH 针对 reference→target 生成和可识别零点进行机制重构，已通过工程与训练首段，但能否在自然 C128 中胜过 strongest wrong 仍待 P0 验证。

这既不是最终成功，也不是原地踏步；它是一个已经把核心科学问题、负证据和下一次决定性实验都明确化的阶段。
