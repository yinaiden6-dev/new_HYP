# REBUT 下一阶段完整研究计划：Mqr 与 Mqrr 的公平比较

版本：v1.0｜日期：2026-09-27｜状态：研究设计，未执行实验、未审计当前代码

## 研究目标与范围

主线不变：自然实拍 query → 冻结检索器产生自然候选 → specific-reference evidence → exact-identity 检索纠错。

本阶段保留当前已验证有效的 M0 和完整 REBUT 系统，检验三个不同问题：

1. 已有证据是否主要缺少合适的决策校准？
2. 单个 query–reference 内，先压成一个数是否过早丢掉可用判别信息？
3. 在局部证据压缩前联合比较两个竞争 reference，是否比“分别编码，再比较”更有效？

Mqr 与 Mqrr 是同一阶段的兄弟实验，不把 Mqr 失败作为 Mqrr 开始的前提。

本计划的核心比较为：

\[
\text{现有系统}\quad\rightarrow\quad
\{\text{同证据重新校准},\ M_{qr},\ M_{qr\text{-}vec},\ M_{qrr}\}.
\]

其中 Mqr-vec 是必要对照：它先独立编码两份 query–reference 证据，保留低维向量，再联合比较。没有这个对照，Mqrr 的增益可能仅来自没有过早压成标量，而不是局部联合比较。

### 证据边界

“粗 M 有效、进一步细化收益有限”按用户已报告的现象作为研究起点，不在本计划中重新声称完成核验。当前代码、训练参数、loss、最新基线数字必须在步骤 0 确认。

不预设 M 是充分统计量，不预设残余错误主要来自 reference 差异，不预设三元输入会学到剂量或规格。低容量读出失败不能证明整个 representation 没有信息；三元模型涨点不能证明具体机制。

已有 few-shot 工作 RaPSPNet 已研究 query 与两个不同类别 support 的联合关系，故“三张图一起比较”不作为首创主张。[5]

---

## 步骤 0：确认当前计算图，封存有效系统

### 目的

确保新实验没有因更换基线、冻结边界、loss 或候选集合而得到无法解释的增益。

### 操作

在实际代码中逐项确认：

- 原始检索分数 b、ColNomic 内容分数 L0、当前可能使用的加权内容分数 Lcur 分别是什么；不把它们默认视为同一变量。
- u、v 是 RoMa 原始 certainty、投影后的支持，还是经可训练模块处理的权重。
- M 的实际公式、有效区域、归一化、padding mask、双向分母。
- 哪些参数 requires_grad=true，哪些进入 optimizer，实际 loss 在哪里计算。
- external 与 internal 分别训练什么，不能把两条路径都笼统写成“只训练 head”。

暂以讨论中的定义记：

\[
M_0(q,R_c)=\sqrt{\bar u_c\bar v_c}.
\]

它只能按已核实的输出解释为支持响应汇总；没有额外验证时，不能称为真实覆盖率、双射一致性、正确身份概率或几何闭合证明。RoMa 官方输出为 warp 与 certainty，这本身不等于 identity supervision。[1]

冻结当前选定的完整系统 B_old：checkpoint、输入预处理、图像分辨率、匹配设置、候选轴、fold、决策规则和完整预测。使用当前已确认的合法最优版本，不因新方法容易获胜而选较弱旧版本。

ColNomic 和 RoMa 的参数保持冻结。第一轮不改变原 Lcur，不用新权重反向改变 ColNomic 内容评分；否则同时变化了证据和内容路径。已有可训练 support 模块先随基线 checkpoint 固定；新模块另建。

### 产出

`baseline_audit.md`：真实计算图、冻结/训练参数表、实际 loss、基线来源及复算结果。

`baseline_predictions.parquet`：每个 query 和全部自然候选的旧预测、已有分数及最终决策。

### 完成条件

复现既定基线预测；数值容差沿用已有工程规范。若不一致，先定位工程差异，不比较科学结果。本步骤不是重新做一轮大规模研究，也不要求新增复杂的哈希治理体系。

---

## 步骤 1：区分残余错误发生在哪一层

### 目的

区分“候选找不到”“匹配器没提取到”“表示压缩/比较方式不合适”“最终决策没用好”，而不是把所有错误都归因于 M。

### 操作

对当前残余错误逐例记录，并加入一组按预先固定规则随机抽取的正确案例作为对照。检查原始 query、target reference、实际错误 winner，以及完整候选排名。

使用以下诊断标签，允许一个案例有多个原因，也允许不确定：

| 类型 | 观察 | 含义 |
|---|---|---|
| 候选缺失 | 正确 identity 不在自然 C128 | 固定候选的重排不能救回 |
| 可用信号未被决策利用 | 某些证据有利于 target，但最终仍选错 | 校准或多候选竞争值得检查；仅 Mtarget > Mrival 不足以定性 |
| 两候选总体支持接近 | M 接近，但可见内容疑似能区分身份 | 有进一步读出/比较空间的假设 |
| 匹配提取失败 | 可见内容未被正确对齐、响应不足或错配明显 | 聚合器不一定能恢复上游未提取的信息 |
| 可见信息不足或不确定 | 遮挡、模糊、视角问题，无法可靠判读区别 | 不把它宣称为已证明的信息论不可识别 |

人工判读时隐藏模型分数和 target 标识，随机展示候选左右顺序；判读完成再与标签核对。可用两人独立判断；意见不一致记为不确定。人工标注只用于分析，不进入模型训练或候选选择。

### 产出

`failure_audit.csv`：案例、可见性、提取状态、候选是否包含 target、各系统预测、审阅一致性。

### 完成条件

得到各类数量和不确定数量。此诊断不作为 Mqrr 的硬准入门；人工没看出差异不等于模型必然没有空间。

---

## 步骤 2：建立相同的输入信息预算

### 目的

防止 Mqr 只看两个平均值，而 Mqrr 看完整图像，然后错误宣称“三元比较优于二元比较”。

### 冻结输入包

对每个自然候选 c 建立共享证据包：

\[
E_c=E(q,R_c).
\]

第一轮只使用已选定层级的冻结 RoMa 特征与对应关系，不新增 backbone，不提高分辨率。证据包包括双向支持响应、warp、有效性 mask、同一层级的 query/reference 局部描述子。若当前缓存缺少局部描述子，对所有实验臂统一补充一次，并注明这是新增输入层级。

在 query 的共同有效网格上，可构造：

\[
x^q_{c,i}=\big[F_q(i),\ F_{R_c}(W_c(i)),\ u_i^c,\ \tilde v_i^c,\ \mathrm{valid}_{c,i},\ \mathrm{coord}_i\big].
\]

其中 reference 特征通过固定 warp 采样到 query 网格；反方向构造 reference 侧证据包。具体 u/v 映射以步骤 0 的审计定义为准，不凭名字替换变量。

只排除 padding 和几何无效位置；低置信响应保留，并用 mask 区分“不可见/无有效映射”和“可见但不支持”。不以 target label、人工框或 M 排名筛选 token。不重新开发 superregion bank 或 P-lock 选择器。

不同 reference 的原生网格不能按相同数组下标直接相减。第一版 Mqrr 的局部比较统一在 query 网格完成。

### 两层输入的解释

S 层：仅支持值/统计量，检验置信响应的读出。

F 层：支持、对应与冻结局部描述子，检验包含局部内容的读出。

主比较的 Mqr、Mqr-vec、Mqrr 统一使用 F 层。只用 S 层不能默认识别“100”和“300”；F 层成功也不能自动等同于原始 certainty 中存在这些文字信息。

### 产出

`evidence_schema.md` 与共享只读缓存，记录真实 shape、坐标变换、mask 语义、特征来源。

### 完成条件

三条新分支读取同一批 E；新证据模块不读取 identity ID、target 标签、文件名语义或检索排名。原始检索分数只在共同的最终决策层使用。

---

## 步骤 3：并行建立实验臂

### 目的

把“更好校准”“更好单对读出”“更少压缩”“更早跨候选交互”拆开。

| 实验臂 | 新增内容 | 回答的问题 |
|---|---|---|
| B_old | 无，封存的完整现有系统 | 新方法是否真的超过现有结果 |
| B_cal | 用原有证据、统一新训练协议重训决策层 | 是否只是 loss/校准改变带来提升 |
| QR | 单个 E 独立编码并压成标量 | 单对内部可学习读出是否有效 |
| QR-vec | 单个 E 独立编码成向量，再比较两个向量 | 是否只需保留更多单对信息 |
| QRR | 在局部证据聚合前联合比较 Eg、Ew | 早期跨候选交互是否提供额外价值 |

### QR：可学习单对证据

采用共享低容量 token MLP，分别编码 query 侧和 reference 侧证据，使用 mean pooling 与 learned attention pooling，再输出一个实数：

\[
a_c=f_\theta(E_c),\qquad D_{qr}(q,g,w)=a_g-a_w.
\]

同一 E_c 在面对不同 rival 时，a_c 必须不变。这是 QR 的定义边界。

新 a_c 是 learned identity-evidence score，不再沿用 M0 的 coverage/支持比例语义。

### QR-vec：压缩容量对照

用同样的独立编码器保留 16 维向量：

\[
z_c=f_\theta^{vec}(E_c),
\]

再比较：

\[
D_{late}(g,w)=\tfrac12\{h_\psi(z_g,z_w)-h_\psi(z_w,z_g)\}.
\]

该臂已经能建模候选之间的非线性关系，只是交互发生在各自空间聚合之后。它不是“完全没有比较”的弱对照。

### QRR：局部联合比较

先分别得到共同 query 网格上的局部编码 hgi、hwi，在聚合前组合：

\[
d_{gw,i}=\psi_\theta([h_{g,i},h_{w,i},h_{g,i}-h_{w,i},h_{g,i}\odot h_{w,i}]).
\]

对这些联合局部描述子池化，并结合两侧 reference 汇总，得到 k(Eg,Ew)。最终定义：

\[
M_{qrr}(q,g,w)=\tfrac12\{k(E_g,E_w)-k(E_w,E_g)\}.
\]

因而结构上满足：

\[
M_{qrr}(q,g,w)=-M_{qrr}(q,w,g),\quad M_{qrr}(q,g,g)=0.
\]

这是相对证据的接口约束，不是“学会了语义差异”的证据。完整 HOLD/SWITCH 决策层仍可保留原 winner 的保护偏置。

第一版不额外运行 Rg↔Rw RoMa，不构造强制 difference mask，不把 reference 外观不一致直接认定为身份差异。QRR-v1 比较的是：query 同一局部在两个竞争 reference 下呈现怎样的支持和内容关系。

### 容量约束

建议首版公共 hidden width 为 64、QR-vec 输出维度为 16；QR、QR-vec、QRR 总可训练参数均控制在 25 万以内，尽量控制在 ±10% 内并报告差异。这些是起始实验配置，不是已验证最优值。B_cal 保持简单，不为凑参数而人为加大。

所有新臂保留原始 M0 和原有内容分数供共同决策层使用，不先删除已有效证据。替换 M0 仅作为后续消融。

### 产出

四个可训练臂 B_cal、QR、QR-vec、QRR 及参数/输入/计算量对照表，外加 B_old 只读回放。

---

## 步骤 4：统一标签、训练目标和自然候选决策

### 目的

让 label 训练证据提取方式，同时学习避免错误改判；不只证明在人工提供的 target–rival 二选一里有效。

### 候选规则

沿用冻结检索器产生的自然 C128 和原始 winner w。所有新臂使用相同候选。训练、验证都不把缺失 target 塞入候选，不按真实标签挑 rival。

沿用当前 reference→identity 聚合规则。以下公式以一个 identity 对应一个决策动作书写；多个物理 reference 属于同一 identity 时，不把它们当互相排斥的错误类别，可先按既有规则聚合或采用正确 identity 的概率和。不得静默改变已有候选轴。

### 第一版统一 loss

共同决策头读取原有证据 x0 与新关系项：

\[
z_w=0,\qquad z_g=h_\phi(x^0_{qgw})+D_\theta(q,g,w),\quad g\ne w.
\]

B_cal 设 D=0；QR 用 a_g−a_w；QR-vec 与 QRR 使用各自反对称关系项。各新臂采用相同决策头类型、相同 loss 和优化预算。

若正确 identity y 在候选内，使用候选级交叉熵：

\[
p(c|q)=\frac{e^{z_c}}{\sum_{j\in C(q)}e^{z_j}},\qquad
\ell_{hit}(q)=-\log p(y|q).
\]

label 是离散候选索引，不需要对 label 求导。交叉熵对连续 logits 的梯度为 p(c|q)−1[c=y]，它可以继续传到可训练证据读出层。PyTorch CrossEntropyLoss 正式支持离散 class indices。[4]

这自然包括：winner 正确→压低错误 challenger；winner 错但 target 在候选→提高正确 challenger；其他错误 challenger→不能因略胜原 winner 就被奖励。

若 y 不在候选内，不能伪造“winner 是正确身份”的标签。本版规定不鼓励无证据改判，使用：

\[
\ell_{miss}(q)=\frac{1}{|C(q)|-1}\sum_{g\ne w}\operatorname{softplus}(z_g).
\]

这是“拒绝现有错误 challenger”的保守动作监督，不是正确身份监督；此类 query 在准确率上仍计为错误。整体按 query 等权平均对应的 hit/miss loss。

不另加只包含 target–rival 的辅助 loss，避免增加第一轮混杂。后续确需辅助 pairwise loss 时，只在两者中恰有一个正确的样本上使用；两者都错时不能同时要求正反方向都输出负的反对称分数。

### 推理规则

固定原始 winner w，一次性计算所有 challenger 相对 w 的分数。选择最高 z_g；仅当它严格超过共同阈值 τ 才替换，平局保持原 winner。不得逐个更新 winner 形成顺序依赖。

τ 在每个外层训练 fold 内部选择，遵守预设 break 成本；没有既有成本时首版按净正确数 R−B 选取，并在平局时选择更保守阈值。不在外层测试 fold 上调阈值。

新增关系模块初始化为零输出时，应能回到同 fold 的 B_cal 初始预测。

### 产出

共享 loss、候选构造、推理器及训练日志。实际旧 loss 由步骤 0 记录；这里的 loss 是新实验统一协议，不冒充旧实现。

---

## 步骤 5：完成工程检查和分组训练

### 目的

验证梯度路径、候选选择、对称性和数据隔离，避免把工程错误或身份记忆当成方法结论。

### 工程检查

小规模训练子集检查数值有限、loss 能下降、读出层在若干更新后得到梯度、冻结 backbone 参数不变。QR 输出不随 rival 改变；QRR 交换顺序严格反号；相同 reference 的关系分数为零；候选顺序置换仅置换输出，不改变最终 identity；零残差回到 B_cal。

这些只说明工程可用，不构成科学 GO。

### 数据划分

沿用 H593 当前已确认的 grouped folds，不挑新的有利 split。先分组，再构造 pair/triple；同一 query 的全部候选及增强不能跨 fold。GroupKFold 的基本保证是组在各测试 fold 间不重叠。[2]

审计“identity-disjoint”究竟覆盖哪些角色：测试 query identity 是否曾以训练 negative/reference 身份进入可训练模块？若出现过，则不能写“所有 reference identities 对模块都未见”。纯冻结 gallery 编码不等于 supervised exposure，但不能用测试 gallery/标签拟合归一化或阈值。

主比较继承并准确描述现有协议。若既有合同要求严格 identity-disjoint，却发现负 reference 泄漏，先修复协议并在同一修正规则下重训所有新臂和 B_cal；历史 B_old 单列，不跨协议直接归因。严格版本可在训练期将候选限制到允许的训练 identities，按冻结分数重新确定训练 anchor；推理仍保留正式自然 C128，并报告训练/推理候选差异。

H593 是已经参与开发的数据，继续称 development grouped OOF，不重新包装成未触碰测试集。这不否定既有药品域阶段性验证。

### 调参与预算

外层 fold 只评估；标准化、早停、阈值和配置选择只用外层训练部分的内层分组数据。重复用同一外层结果挑配置会造成选择偏差，嵌套划分用于隔离这种选择。[3]

首轮每臂一套预先冻结主配置，固定 seeds={0,1,2}；各臂沿用相同优化预算。若无可复用配方，建议统一 AdamW，lr=3e-4，weight_decay=1e-4，最多100 epochs，按内层验证 loss 早停、patience=10，batch 按 query 计数。全部只作为预注册起点，不称最优参数。

5 folds×3 seeds×4 可训练主臂 = 60 次小模块拟合，B_old 只回放。缓存先检查现有可复用内容，不重新运行不必要的 backbone forward。此处是计算任务预算，不是工期承诺。

### 产出

`engineering_checks.json`、`fold_exposure_audit.csv`、全部 inner-validation 和 OOF 预测、参数量及资源消耗。

---

## 步骤 6：评估真实纠错，而非只评估二选一

### 目的

确定新证据在全部候选竞争中是否有净收益，且不是以大量 break 换取表面 rescue。

### 指标

主指标为全 query、自然 C128 下的 exact-identity top-1。对每个基线分别计算：

\[
\Delta\mathrm{Acc}=\frac{R-B}{N},
\]

R 为基线错、新方法对；B 为基线对、新方法错。基线至少包括 RAW、B_old、B_cal。QRR 的结构性比较另以 QR-vec 为基线。

同时报告 target-in-C128 比例及条件准确率、错误→错误切换、switch 总量、break rate、逐 fold/seed 结果、延迟和缓存成本。target–rival margin、二选一胜率和人工可见性分组仅作诊断，不代替 C128 主指标。

置信区间采用按 identity 分组的 paired resampling：同一 identity 的 queries 一起抽样，建议2000次。不同 seed 的结果分别报告，种子不是新增独立测试样本；不能把 N×3 当成三个独立测试集。区间不消除已发生的开发选择偏差，也不替代外部验证。

### 首轮预设判定

建议把“值得进入主线”的最小工程收益设为1个百分点；在593个query上约对应6个净正确案例。这是研发门槛，不是预测。

- 工程通过：步骤5检查完成。
- 效用 GO：相对 B_old 和 B_cal 都有预设实质增益，3个 seed 方向一致，配对区间支持正向，且满足预先确定的 break 操作约束。
- 交互机制支持：QRR 还须超过 QR-vec；只超过标量 QR 不足以支持局部联合比较的必要性。
- 不确定：点估计正向但区间跨零、种子不稳定或效应过小。明确写证据不足，不升级为 GO，也不宣称无信息。
- 本版本 NO-GO：无净收益或 break 恶化，停止该实现的无边界调参，保留有效旧系统。

本轮预先指定 QRR vs QR-vec 为结构性主对比，QR vs B_cal 为单对读出对比；不从许多事后子集中挑最好的结果替代主比较。

### 产出

`main_results.csv`、`paired_comparisons.csv`、全部逐query预测与不确定性说明。

---

## 步骤 7：只对有效分支做机制检验

### 目的

区分“用到了新的结构化证据”“依赖 query”“依赖候选对应关系”与“真的利用身份差异”。

### 预设关键对照

1. 同证据 refit：B_cal 是第一层对照。它追回增益时，不把提升归因于新 M。
2. QR-vec 对照：判断 QRR 的收益能否被晚期向量比较替代。
3. 去局部内容只留 support：在有效臂中重训，判定提升依赖 confidence 形状还是冻结局部描述子。
4. 候选错绑/对齐破坏：保持候选轴与内容基线不变，破坏 query–reference 证据关联。固定模型干预测已训练模型依赖；受限输入下重新训练测替代方案能否恢复。二者分别报告。
5. query-free 对照：真正去掉 query 及所有 query-conditioned warp/support，只使用两个 references 建模。仅删显式 query token、却保留 E(q,R)，不能称 query-free。
6. 在少量预先固定的“差异可见”诊断案例中，对疑似差异区域和等面积共享区域分别做遮挡/模糊。检查决定是否更受差异区域影响；同时承认干预引入的分布偏移，不凭热图或单个案例宣布因果。

反对称性和相同 reference 输出零是单元测试，candidate binding 是支持性控制，均不单独包装为创新。

### 产出

`mechanism_report.md`：每项干预、对应假设、可得结论、不可得结论及失败案例。

### 完成条件

只写实际支持的解释。模型未显示差异区域依赖时，可称联合局部证据模型有效，不能称“学会了剂量差异”。

---

## 步骤 8：验证迁移，最后再考虑 internal 接入

### 目的

判断新模块是否超出当前药品开发集，同时不让本阶段无限扩张。

### 迁移顺序

先复用现有 grocery transfer 协议作为回归检查。源域模型、归一化、阈值全部固定；已有看过的 GroZi 结果不能改称 untouched，但仍是有意义的 source-frozen 迁移检查。

再按现有 ColQwen 协议检查跨检索器适配。必须区分：完全冻结源域/源检索器训练参数的迁移，与在 ColQwen 训练 folds 上重拟合后的可移植性；二者不是同一主张。

新增 Products-6K 等外部数据不是本轮启动前提。若后续增加，应先核实可下载版本、gallery/query 规模、标签与许可，并在看结果前固定协议。

首轮只接 external 决策层。只有证据层出现稳定增量，才单列新实验把它接入 internal query-representation 路径；不同时改变 M、query 表示、retrieval projection 和 loss。

### 产出

`transfer_report.md`，明确 frozen/refit、已使用/新增数据边界，报告完整 failures 与成本。

---

## 步骤 9：按结果收口

### 目的

让实验决定下一步，而不是每个失败都触发一个更复杂的新模块。

| 结果 | 可支持解释 | 下一步 |
|---|---|---|
| B_cal 已追回主要增量 | 现有证据的使用方式重要 | 保留更简单决策层 |
| QR 优于 B_cal，QRR 无额外收益 | 本版单对可学习读出有用 | 优先 QR，不强行三元化 |
| QR-vec 优于 QR，QRR≈QR-vec | 过早标量瓶颈值得关注，晚期比较已足够 | 保留向量方案 |
| QRR 稳定超过 QR-vec | 本设置支持聚合前跨候选交互增量 | 继续机制与迁移验证 |
| 三条新分支均无稳定增量 | 本轮未读出可泛化增益 | 停止本实现族，保留 M0 和旧成果 |
| 二选一涨、C128不涨 | 相对信号未转化为多候选决策收益 | 先定位校准、第三候选与阈值，不宣布成功 |

不允许从 QR≈M0 推出 M0 是充分统计量；不允许从 QRR>QR 推出 reference-relative 信息是唯一缺口；不允许把尚未通过控制的局部热图解释当成机制结论。

当前论文的冻结成果与下一阶段探索保持版本隔离。新分支不成功不取消旧结果，也不要求推迟已有论文/毕业交付。

---

## 每一步目的速查

| 步骤 | 最主要的问题 | 最小交付物 |
|---|---|---|
| 0 | 当前到底在算什么、训练什么？ | 基线与计算图审计 |
| 1 | 错误出在哪一层？ | 失败类型与不确定案例表 |
| 2 | 比较是否基于同样的输入信息？ | 共享证据缓存定义 |
| 3 | 压缩、容量和交互如何拆开？ | QR/QR-vec/QRR与校准对照 |
| 4 | 离散 label 如何训练并避免乱换？ | 共享 loss 与自然候选决策器 |
| 5 | 是否工程正确、没有泄漏？ | 测试及 grouped OOF 预测 |
| 6 | 是否在完整任务中净增益？ | 全候选结果与paired对比 |
| 7 | 增益到底依赖什么？ | 受控干预报告 |
| 8 | 是否能迁移？ | source-frozen/refit 分开的迁移结果 |
| 9 | 什么保留、什么停止？ | 方法选择及声明边界 |

## 最小执行交付集

`baseline_audit.md`、`failure_audit.csv`、`evidence_schema.md`、统一训练/推理入口、全部 OOF 预测、`main_results.csv`、`mechanism_report.md`、`transfer_report.md`。

不得只交一个平均准确率或几张漂亮热图；不得将目标已知的诊断结果混入目标未知的正式结果。

## 参考资料与用途

[1] Edstedt et al. RoMa: Robust Dense Feature Matching. CVPR 2024. 官方实现输出 warp/certainty，并列明版本复现差异。用于界定冻结证据的来源，不用于证明本项目 M 的身份判别性。官方来源：`https://github.com/Parskatt/RoMa`

[2] scikit-learn. GroupKFold documentation. 用于组不重叠划分的基本定义；本项目的身份/负 reference 暴露仍需额外审计。官方来源：`https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html`

[3] scikit-learn. Nested versus non-nested cross-validation. 用于说明模型选择与外层评估隔离及选择偏差。官方来源：`https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html`

[4] PyTorch. CrossEntropyLoss documentation. 用于离散 class index 与连续 logits 的监督接口。官方来源：`https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html`

[5] Zhang et al. Re-abstraction and perturbing support pair network for few-shot fine-grained image classification. Pattern Recognition 148, 2024. 出版社摘要明确涉及 query 与两个类别 support 的联合关系。用于限制“首次三元比较”的主张，不表示与本计划完整实现等价。出版来源：`https://www.sciencedirect.com/science/article/pii/S0031320323008555`
