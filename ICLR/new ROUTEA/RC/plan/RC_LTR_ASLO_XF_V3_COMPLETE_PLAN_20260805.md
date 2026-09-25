# Route A RC-LTR–ASLO-XF V3 完整方案

## Reference-Conditioned Target Hypothesis, Cross-Fitted Local-Odds Verification, and Exact-HOLD Retrieval

**版本：** `rc_ltr_aslo_xf_v3_20260805`  
**日期：** 2026-08-05  
**当前状态：** `MECHANISM_PLAN_AND_CONTRACT_FROZEN / E0_ONLY / NO_EXPERIMENT_RUN`  
**本轮授权：** 只完成方案与合同；不运行 smoke、不提交 Slurm、不访问 opened/sealed  
**唯一实验根目录：** `/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC`

---

# 0. 最终决策

本路线不再把 ownership 定义为一张自由学习的 patch 概率图，也不再让
PPR、top-k、selector 或 score-slot 全量重排承担“发现 target”的责任。

唯一主机制为：

\[
\boxed{
\text{full-gallery base}
\rightarrow
\text{每个 reference 生成 candidate-conditioned target hypothesis}
\rightarrow
\text{held-out cross-fit verification}
\rightarrow
\text{conditional match-vs-no-match evidence}
\rightarrow
\text{query-level empirical CAL max-exceedance SWITCH/HOLD}
}
\]

统一命名：

```text
RC-LTR：科学框架，规定每个动态 reference 独立提出 target hypothesis；
ASLO-XF：项目内新命名的 `Atlas-conditioned Signed Local Odds with Cross-Fitting`
求解器，用高分辨局部证据、几何假设、cross-fit和no-match验证；它不是既有文献方法，
也不表示pixel synthesis、3D rendering或inverse-graphics保证；
D1：当前987-query主线的冻结完整图库全局/domain base；
A10：仅作为已打开历史224-query回归证据，不进入本路线训练或选择；
ownership：候选 hypothesis 相对 challenger 的逐区域 signed conditional evidence。
```

这不是两套模型。RC-LTR 定义问题，ASLO-XF 实现 hypothesis-and-verify。

---

# 1. 研究目标与不可违反合同

## 1.1 输入与目标

模型接收：

1. 自然 query 图像或协议内冻结的三视图；
2. 完整图库中的动态 visual references；
3. 训练时 exact query-reference identity pairs；
4. 冻结/版本化 retriever 自然产生的竞争 references。

推理必须：

```text
target-free；
完整扫描 5,413 physical rows / 5,404 exact labels；
不插入真实 target；
新 identity 只增加 reference atlas，不重新训练 identity 参数；
unseen identity 可由运行时 reference 定义；
最终输出完整图库 exact-label ranking。
```

## 1.2 禁止信息

禁止：

```text
人工 target masks / boxes / points / polygons；
target-derived pseudo mask；
oracle candidate insertion；
identity-specific trainable parameters；
推理读取 target label；
local hypothesis generator 读取 base rank、slot、winner flag 或 correctness；
opened/sealed 参与特征、参数、阈值、candidate width 或模型选择。
```

自动、冻结、候选对称的局部特征/文字/几何提取允许，但其来源、版本和 hash
必须进入 receipt。

## 1.3 成功命题

成功不是“画出看起来像目标的图”，而是同时证明：

1. reference 在不知道 target 时生成 candidate-specific query hypothesis；
2. correct-reference hypothesis 的 held-out evidence 胜过 no-match 与自然 rival；
3. evidence 含有 base/rank 之外的 identity 条件信息；
4. 该证据产生正的 full-gallery net rescue；
5. candidate binding 和空间/几何销毁分别消除增量；
6. 无证据时完整 base ranking byte-identical HOLD；
7. signed ownership attribution以预注册Aumann--Shapley积分在冻结数值容差内重构
   被部署的局部增量，而非用不闭合的热图冒充证据。

---

# 2. 历史实验如何进入 V3

历史结果永久保留，不覆盖、不重写。

| 历史阶段 | 可继承结论 | V3 组件 | 明确淘汰 |
|---|---|---|---|
| D1/domain | photo-to-reference domain alignment 是主底座 | 冻结 full-gallery base | 让未成熟 local path 替代 D1 |
| C0/A10 rank-only | candidate support/rank 是有效 global ownership | 仅作历史回归证据 | 把已打开的224-query A10用于当前训练/选择，或把rank gain冒充空间ownership |
| E5 growth | seed retention/collision 审计有用 | proposal coverage receipt | free region growth；稳定时欠扩张、充分时不稳定 |
| C1 | binding/reference correspondence 有信号 | candidate/reference binding 合同 | soft correspondence 自动改善 retrieval；C1 低于 rank-only，spatial permutation 不降 |
| C2 | C64 symmetry/dustbin/full tensor 可实现 | exact-label candidate context、H0 | candidate symmetry 自动阻止 prior confirmation |
| C6b-3h | target-star/unary 约占 99.95% | candidate-specific unary proposal | 默认 graph/PPR 是增益源；graph 约 0.05% |
| C6d | A10 在历史224-query上是更强base；signed same-pair思想有效 | signed g-vs-c 合同 | 将A10直接移植为987-query部署base；A20/readout；A10=A20=157/224，0 rescue/0 break |
| C6d-E/F0/F1 | candidate binding 存在；必须允许不动作 | explicit H0、query-level empirical gate与HOLD | top-8、positive clamp、错位 unused-mask、always-on gate、learned all-abstain selector |
| V3.1 direct mean | strongest-rival ledger与prejoin可复用 | one-action policy、完整 receipt | direct local 全接管；历史出现大量 breaks |
| RC-CF V1 | C64 外 byte-identical、zero-sum、direct/total/relock 分账 | overlay与因果 ledger | student self-target、同一 branch 自己提案又验证；正式 P0 仅净 +1 且核心门失败 |
| RC-CF V4 | legal target-known capacity 非零 | headroom gate | 把 oracle 当 learnability；149 wrong 中仅 123 在 C64，strict oracle 只救 21，26 在 C64 外 |
| Patch Utility | 正/负 patch 与 cancellation 必须分账 | 离线 utility/causal validator | 把 target-known utility 输入推理；现有 P0L 仍为 NO-GO |
| LTR v2 | seed/verify 解耦、null 意识正确 | proposal/verification split、whole-query null | PPR 主线、unit-mass seed、candidate-wise MAD-Z、LOCK-only 全量 score-slot |

权威证据路径包括：

```text
ICLR/audits/AUDIT_ROUTEA_ALL_EXPERIMENTS_RESULTS_RECORDS_PROTOCOLS_20260803.md
ICLR/new ROUTEA/reports/REPORT_ROUTEA_C6B3H_STAR_GRAPH_FACTORIZATION_20260731.md
ICLR/new ROUTEA/reports/REPORT_ROUTEA_C6D_JOB5038839_ROOT_CAUSE_AND_C6DE_DECISION_20260801.md
ICLR/new ROUTEA/results/route_a_v2_rc_cf_v1/stage_P0/fold0_seed17_v1/result.json
ICLR/new ROUTEA/results/route_a_v2_rc_cf_v4/stage_A0/a0_headroom_seed17_v1/result.json
ICLR/new ROUTEA/reports/REPORT_ROUTEA_V2_P0L_E32_JOBS5044342_5044343_5044345_DECISION_20260804.md
```

---

# 3. 正确的因果顺序

每个 natural candidate reference 独立提出 hypothesis：

\[
R_g,x
\longrightarrow
\mathcal H_g(x)
=
\{h_{g,0},h_{g,1},\ldots\}.
\]

target 不预先存在于 forward 中。所有 hypotheses 在 target-free 条件下竞争：

\[
\hat y
=
\operatorname*{argmax}_{g\in\mathcal C(x)}
\operatorname{Evidence}(h_g\mid x,R_g,B).
\]

ownership 在 hypothesis 之后定义。由于候选证据还会经过 hypothesis/row
log-marginal，最终逐区域 ownership 必须采用第11节的 Aumann--Shapley
路径积分；不能把单个已选 hypothesis 的 raw \(\ell_i\) 差直接冒充部署分数分解。

因此：

```text
错误顺序：先猜 ownership map，再希望 map 找到 target；
正确顺序：reference 先提出 target hypothesis，验证后确定 target，ownership 再解释竞争证据。
```

---

# 4. Base 与 natural candidate 合同

## 4.1 Full-gallery base

当前987-query主线冻结V3.1 D1机制与训练合同。任何OOF endpoint都必须使用只在该
evaluation fold之外训练的fold-local D1：

\[
B_r(x),\qquad r=1,\ldots,5413.
\]

权威fold-0 checkpoint、manifest、gallery与split的路径和哈希冻结在
`RC/registry/upstream_inputs.json`；该checkpoint只对其未训练的fold-0 heldout 212
提供合法base，不能给自身775个训练query产生科学endpoint。其它fold必须按同一冻结recipe
重新训练D1并登记hash，eval identity不得进入对应D1。历史 A10 已在 opened difficult224 上使用，
只允许作为回归/反例证据，禁止参与当前参数、阈值、candidate width和模型选择。

exact-label reducer冻结为historical deployment的physical-row max，tie按canonical row ID；
源文件和hash登记在`upstream_inputs.json`。它对label内统一offset具translation
equivariance，是第10.3节score projection闭合的前提。多个physical rows不得在candidate
population中重复占位。

## 4.2 Candidate width

A0 固定 natural exact-label `C128`：

```text
先完整图库评分；
稳定 score-descending / exact-label-ascending tie；
不插 target；
C128 外 rows/labels 始终保持 base；
所有query执行同一target-free action rule；
target-outside-C128只能在postjoin evaluator中记为proposal miss，绝不能改变forward或强制HOLD。
```

资格门：optimization/base-wrong 中 natural target coverage 必须至少 90%。不满足则
`A0_CANDIDATE_RECALL_STOP`，不得在同一结果上扫描 K。

C128 是基于旧 C64 仅覆盖 123/149 base-wrong 的预注册修正，不是 eval 调参。

---

# 5. Reference atlas 与新增可观测证据

## 5.1 为什么必须新增表示

C6d 中 A10/A20 residual cosine 约 0.9671，说明在相同 contextual token 上只换 pooling、
gate、OT、slot、graph 或 PPR 大概率继续重复 rank-only。

V3 主路径必须使用 rank-only 未直接观察到的证据：

```text
native/high-resolution bounded-receptive-field visual descriptors；
多尺度细粒度局部外观；
query/reference coordinates；
点、线、角点与包装平面几何；
dense raw-glyph/text appearance及其布局；
candidate-independent no-match/outlier model。
```

旧 3584-D local cache 只能作为历史 matched control，不能单独充当“新信息已经产生”的
证据。

## 5.2 Dynamic reference atlas

每个 reference row 物化：

```text
visual pyramid与stride/receptive-field receipt；
每个local descriptor的reference坐标；
line/corner primitives；
dense glyph/text visual descriptor与textness；
reference image hash；
encoder/model/processor hash；
physical-row exact-label binding。
```

atlas 由共享冻结 encoder 产生，无 identity-specific parameter。新 reference 只需生成
atlas，不重训模型。

## 5.3 Glyph/text 边界

主证据使用 dense glyph 外观、字形、间距和布局，不依赖人工文字框或产品名称标签。

OCR string 只能作为独立辅助消融，不能：

```text
生成target mask；
成为唯一candidate score；
使用人工品牌/药名标签；
绕过same-string/different-package hard negatives。
```

## 5.4 Feature implementation manifest gate

本文件冻结的是科学机制，不假装尚未物化的encoder/checkpoint已有hash。E0只允许测试与
具体backbone无关的数学core。进入E1前必须另行得到用户授权，并在任何自然query结果前
冻结唯一的`feature_implementation_manifest.json`，至少写明：

```text
唯一local encoder/checkpoint及SHA；
query/reference输入分辨率、stride和真实receptive field；
visual/glyph/line/corner各通道维度与归一化；
共享/冻结/可训练参数边界；
训练identity与增强来源；
reference atlas cache格式和版本；
显存/streaming方案；
禁止替代backbone与结果后fallback。
```

manifest只能依据可用性、合成fixture与资源上限冻结，不能依据自然retrieval结果挑模型。
若唯一实现不可用，状态是工程停止，不得悄悄换backbone后继续同一版本。

---

# 6. Candidate-conditioned target hypothesis

## 6.1 Hypothesis 状态

对 candidate exact label \(g\)、physical row \(r\)：

\[
h_{g,r,k}
=
(m_k,H_{g,r,k},\widehat\Omega^Q_{g,r,k},\widehat V^R_{g,r,k},
 \widehat e^{H0}_{g,r,k}).
\]

其中：

```text
m_k：模型类别；
H：reference→query几何；
widehat Omega^Q：该candidate的估计query support weight，不称真实posterior；
widehat V^R：估计的可见reference部分；
widehat e_H0：估计的candidate absent/no-match evidence，由固定marginal导出，不是自由gate。
```

这不是人工mask。\(\widehat\Omega^Q\)由匹配、几何和outlier mixture估计产生；其概率
校准必须由identity-disjoint OOF门验证，不能因变量名而声称posterior正确。

## 6.2 A0 模型族

A0 只允许：

\[
H_0=\text{candidate absent},\qquad
H_1=\text{one visible plane}.
\]

自由多 homography 被禁止。`H2` 只有 H1 路线正式显示 headroom 后才可新版本启用，且
必须是两个预定义相邻panel、共享边并满足物理兼容约束；不得独立拟合两个自由 H。

## 6.3 Proposal

对 proposal bank \(A\)：

\[
\widehat a^{A}_{ij,g,r}
=
\log
\frac{\widehat p_+(s^{A}_{ij}\mid \text{scale,channel})}
     {\widehat p_0(s^{A}_{ij}\mid \text{scale,channel})}.
\]

只从正的 absolute match-vs-no-match ratio、mutual/cycle 合同与非退化点集产生
**固定恰好 \(K_H=8\) 个 H1槽位**。不足8个时，剩余槽位逐位写成H0。退化H1只有
在读取held-out evidence之前，由result-blind proposal结构/数值合同判定时才严格等价于
H0。合法H1一旦进入verification，其held-out \(Z\)无论正、零、负都必须原样进入9-slot
marginal；禁止按verification结果回退H0或形成hypothesis-level positive clamp。
proposal score不是最终candidate score，不能因有效hypothesis更多而增加候选质量。

pre-verification validity只允许读取proposal matches、finite、非共线/rank、conditioning、
orientation、projected finite/area等target-free量；具体数值阈值必须在E1前的feature
implementation manifest中冻结，且`h1_validity_reads_heldout_evidence=false`。

几何 proposer 可使用 MAGSAC++ 或等价冻结 robust estimator，但最终 evidence 不能使用
其训练/拟合点重新自证。

---

# 7. Cross-fitted hypothesis verification

## 7.1 Split 合同

只把query crops按query hash、坐标棋盘与feature-bank版本确定性分成A/B；reference是
完整条件对象，不做空间A/B切分。每个reference token必须预先物化两套冻结描述通道：
proposal通道\(d^P_R\)与verification通道\(d^V_R\)。query也有对应\(d^P_Q,d^V_Q\)。

```text
A-query + P-reference proposes H; B-query + V-reference verifies H；
B-query + P-reference proposes H; A-query + V-reference verifies H；
任何 evidence patch 不能参与产生给它打分的 hypothesis；
相邻/重叠 receptive fields必须进入exclusion halo；
split和halo在target join前封存。
```

verification公式中的\(J_{g,r}\)始终指完整V-reference token set；proposal只读取完整
P-reference token set。P/V channel的checkpoint、训练数据与hash必须在feature manifest
冻结。该设计只防止query同点拟合-验证自证，不声称query patches或P/V channels统计独立。

## 7.2 Appearance 与 geometry density ratio

held-out query bank与V-reference channel的visual/raw-glyph evidence先输入一个共享、
identity-disjoint OOF
冻结的低容量 joint density-ratio calibrator；二者相关，禁止把两个单独 LLR 直接相加。
visual/glyph联合 appearance ratio：

\[
\widehat a_{ij}
=
\log\frac{\widehat p_+(s_{ij})}{\widehat p_0(s_{ij})}.
\]

几何 ratio：

\[
\widehat G_{ij}(H)
=
\log
\frac{K_\sigma(x_i-\pi(Hy_j))}{U_Q(x_i)}.
\]

其中 \(K_\sigma\) 是注册的归一化二维误差模型，\(U_Q\) 是query坐标null；帽号强调
它们是模型内估计量，不是真实数据生成密度。

appearance与geometry在proposal selection后也不保证独立，因此最终pair ratio不是简单
\(\widehat a_{ij}+\widehat G_{ij}\)。固定的joint verifier在identity-disjoint OOF上把
\((\widehat a_{ij},\widehat G_{ij},\text{scale},\text{channel})\)校准成单一估计
log-density ratio \(\widehat q_{ij}(H)\)，并必须胜过打乱geometry的同容量control。
NCE/classifier ratio只有在negative覆盖、sampling-prior correction和calibration成立时才
近似density ratio；本方案不把帽号省略，也不宣称真实likelihood已知。

reference-length-normalized ratio：

\[
\widehat R_i(g,r,H)
=
\frac1{J_{g,r}}
\sum_j\exp[\widehat q_{ij}(H)].
\]

显式outlier mixture；\(\pi_{match}\)明确表示match prior，不使用语义相反的
`pi_outlier`命名：

\[
\widehat\ell_i(g,r,H)
=
\log\left[(1-\widehat\pi_{match})+\widehat\pi_{match}\widehat R_i(g,r,H)\right].
\]

在注册的部署no-match估计模型下，若校准满足\(\mathbb E[\widehat R_i]=1\)，则
\(\widehat\ell_i\)以0为自然基线；它可以为正或负。不得positive clamp。

## 7.3 Cross-fit candidate evidence

先把每个held-out patch写成只含空间证据的可加坐标：

\[
z_i^{A\rightarrow B}(g,r,k)
=
\frac{\widehat\ell_i(g,r,H^A_{g,r,k})}{|B|}
.
\]

于是固定槽位总证据为：

\[
Z^{A\rightarrow B}_{g,r,k}
=
\sum_{i\in B}z_i^{A\rightarrow B}(g,r,k)
=
\frac1{|B|}\sum_{i\in B}\widehat\ell_i(g,r,H^A_{g,r,k})
,
\]

\[
Z^{B\rightarrow A}_{g,r,k}
=
\frac1{|A|}\sum_{i\in A}\widehat\ell_i(g,r,H^B_{g,r,k})
.
\]

非空间复杂度惩罚单独记为
\(P_{d,k}=df(H_k)\log n_d/(2n_d)\)，不得摊到patch或画进ownership图。

每个固定槽位保存两方向 evidence，并要求 action-level agreement。任一方向无合法 H1
时该槽位使用H0=0，不用可调fallback。两方向先分别完成第8节固定槽位、row marginal，
最后才取对称均值；禁止在marginal前混合proposal与verification证据。

---

# 8. Physical-row 与 hypothesis multiplicity

禁止：

```text
patchwise跨row max后拼成不存在的Frankenstein hypothesis；
每row单独Z后再max；
reference rows更多的label获得更多极值机会；
不同hypothesis数量不经校准直接比较。
```

## 8.1 固定 hypothesis marginal

每个 unique reference component 都必须包含1个H0和固定8个H1槽位。对槽位
\(k=1,\ldots,8\) 的held-out总局部证据记为
\(Z_{u,k}=\sum_i z_{i,u,k}\)，非空间复杂度项为\(P_{u,k}\)。定义：

\[
E_u
=
\log\left[
\frac{1}{K_H+1}
\left(1+\sum_{k=1}^{K_H}
\exp\{Z_{u,k}-P_{u,k}\}\right)
\right],\qquad K_H=8.
\]

H0的log-evidence固定为0；每个invalid槽位也按0计。因而所有槽位均为H0时
\(E_u=0\)，不同candidate不会因生成了更多有效proposal获得额外质量。

## 8.2 duplicate component collapse 与 row marginal

先按reference atlas canonical bytes/content hash把完全相同的physical rows合并为
unique component \(u\)。复制同一reference不得增加prior mass。只有hash与shape、dtype、
canonical byte length全部一致才允许合并；hash碰撞直接终止，感知近重复不得自动合并。
component base分数只进入base reducer与receipt，取其重复rows的确定性最大值：

\[
B_u=\max_{r\in\mathcal D(u)} B_r.
\]

local evidence禁止读取query-dependent base。对同一exact label的unique components使用
base-independent均匀prior：

\[
\pi_{u\mid g}
=
\frac{1}{|\mathcal U(g)|}.
\]

label evidence：

\[
E_g
=
\log
\sum_{u\in\mathcal U(g)}
\pi_{u\mid g}\exp E_u.
\]

若所有components的evidence为0，则 \(E_g=0\)。复制完全相同的physical row不改变
结果。`K_H=8`和MDL定义在任何结果前冻结；不允许按candidate改变。

上述marginal对两个cross-fit方向分别计算为\(E_{g,A\to B}\)与
\(E_{g,B\to A}\)，最终\(E_g\)是二者算术均值。MDL项固定为
\(df(H_1)\log n_d/(2n_d)\)，其中homography \(df(H_1)=8\)、\(n_d\)为该方向
held-out patch数。这是result-blind复杂度约束，不宣称有限样本BIC保证。\(B_u\)
绝不进入\(H\)、raw \(Z\)、\(E_u\)或\(E_g\)。

经验CAL reference仍必须对完整row/hypothesis/reducer管线重算，以量化残余selection
extremeness；不由此宣称统计显著性。

---

# 9. D1 与 local 的OOF新增预测信息，不允许双计

## 9.1 不能直接声称 prior + independent likelihood

D1 与 local 使用同一 query，可能共享 backbone、训练数据和外观信息。故：

\[
B_g/T_B+E_g/T_E
\]

不能未经验证就宣称为 Bayes posterior；这可能把相同视觉证据计算两次。

下面的Bayes恒等式只提供“给定base后还需新增信息”的概率动机：

\[
\log
\frac{p(Y=c\mid B,E)}{p(Y=w\mid B,E)}
=
\log
\frac{p(Y=c\mid B)}{p(Y=w\mid B)}
+
\log
\frac{p(E\mid Y=c,B)}{p(E\mid Y=w,B)}.
\]

本方案没有估出右侧真实conditional likelihood。第9.2节的单标量融合只能称
`OOF-calibrated incremental decision score`；只有它在未参与拟合的identity groups上
改善proper loss，才可称\(E\)含有给定base后的新增**预测信息**。

## 9.2 冻结 base calibration，再学习一个local增量尺度

local representation/hypothesis generator仍禁止读取rank/slot/winner。最终action calibration允许读取
冻结base gap，因为必须判断local evidence能否跨越该gap。

先在identity/supergroup-disjoint OOF上单独冻结base temperature\(T_B>0\)，定义严格
单调、保号且过原点的base margin calibration \(f_0(z)=z/T_B\)。输入\(z\)是冻结
exact-label reducer输出的raw score difference；之后不可随local训练改变。令base winner为
\(w\)、challenger为\(c\)：

\[
m_0(c,w)=f_0(B_c-B_w),
\qquad
m_1(c,w)=m_0(c,w)+\lambda\,[E_c-E_w],
\]

约束：

```text
f0必须在local路径训练前冻结；
f0严格单调、sign-preserving且f0(0)=0；
lambda是全模型唯一、共享、非负标量；
lambda=0是合法回退；
同一参数适用于所有identity/candidate；
不输入rank、slot、identity、winner correctness；
raw local hypothesis与E不得读取B/rank/slot；
prior flatten/swap不得改变H或raw E；
与lambda=0的同一个f0比较OOF proper log-loss。
```

\(T_B\)只用训练fold内natural C128 exact-label pairs的pairwise logistic NLL拟合；
\(\lambda\)随后只用同一训练fold的target-joined ledger，在base-correct break约束下最小化
pairwise logistic NLL。两者不看validation fold。base score ties按exact-label升序稳定打破。
在任何action前必须验证\(\lambda=0\)时全部pair margin符号与base order一致。

只有 \(\lambda>0\) 且 full model 在group-OOF中严格改善冻结的\(\lambda=0\) base，
才可称 local 包含条件新增信息。若所有local证据为0，必须逐位精确恢复base行为。

---

# 10. Query-level empirical atlas-derangement exceedance 与 exact HOLD

## 10.1 CAL reference operator必须完整重跑

CAL reference operator冻结为**query内、静态结构分层的candidate-to-reference-atlas整体错绑**。candidate
exact-label slots、base scores/gaps、query descriptors与C128顺序固定；reference atlas以整个
label为原子移动，内部component、token、descriptor、coordinate与glyph multiset不得拆散。

静态stratum key固定为：

```text
(unique_component_count,
 floor(log2(total_reference_tokens)),
 clamp(floor(4*median_component(log2(width/height))),-8,8))
```

每个size≥2的stratum独立抽取uniform derangement；size=1保持原绑定并记
`NULL_CONSERVATIVE_SINGLETON`。为跨环境byte replay，不依赖numpy/torch PRNG：先按
exact-label排序slots；对每个atlas id计算
`SHA256(RFC8785_JCS_UTF8([protocol_file_sha256,"CAL_NULL_V1",query_resource_sha256,
draw_index,stratum_key,attempt,atlas_id]))`，再按`(digest,atlas_id)`排序atlases并赋给已排序
slots；任何digest collision直接停止。若存在fixed point则attempt加1重算。1024次仍失败，
该query强制HOLD。`protocol_file_sha256`是冻结后协议文件原始bytes的SHA-256并写入design receipt。
一个query必须存在至少20个不同的whole-query permutation且至少一半C128 slots可移动，
否则`NULL_ORBIT_INSUFFICIENT_HOLD`。CAL null与正式控制分别使用
`CAL_NULL_V1`和`EVAL_CONTROL_{C,PQ,PR,H,T,M}_V1` namespace，禁止复用draw。

每个CAL draw从错绑后的原始candidate/reference/coordinate binding开始重跑：

```text
local match
→ proposal
→ H选择
→ held-out verification
→ physical-row marginal
→ candidate evidence
→ strongest challenger/max action。
```

只打乱最终 heatmap、patch contributions或E向量不是合法selection-bias校准。
validator逐draw只能检查：base slot/gap逐位不变；atlas静态stratum不变；完整atlas移动；
无target join；full pipeline hash改变；orbit与组合不变量通过。它不能从单query证明统计
exchangeability。这里采用的工作假设是“匹配静态atlas统计后，candidate-reference binding
在CAL orbit内可交换”；该假设未被证明，因此下节只产生经验exceedance score。

## 10.2 Max statistic 与经验exceedance score

令base winner为 \(w\)：

\[
T_{real}(x)=\max_{c\ne w}m_1(c,w;x).
\]

每个CAL draw同样计算：

\[
T_b^{CAL}(x)=\max_{c\ne w}m_1^{CAL,b}(c,w;x).
\]

Monte-Carlo plus-one经验exceedance score：

\[
e_x^{CAL}
=
\frac{1+\#\{b:T_b^{CAL}\ge T_{real}\}}
     {B_{CAL}+1}.
\]

A0固定`B_CAL=255`，正式确认固定`999`。`e_x^{CAL}`只是在注册atlas错绑下的经验安全门，
不是统计p-value，不提供type-I、FWER、conformal或显著性保证。条件新增信息的科学结论
只由group-OOF proper loss、false-action与独立C/P controls承担。

## 10.3 SWITCH/HOLD

只在全部成立时执行一次 challenger switch：

```text
T_real > 0；
e_CAL <= 0.05（预注册经验保守门，不解释为显著性水平）；
A→B与B→A各自margin都大于0且支持同一challenger；
获胜challenger在两个方向都至少有一个target-free合法H1，且各自candidate-vs-H0 evidence>0；
prejoin、CAL和candidate ledger均通过独立validator。
```

否则：

```text
完整5,413-row base tensor byte-identical HOLD。
```

SWITCH先生成exact-label顺序 `[c*, w, 其余labels按原base相对顺序且排除c*,w]`；
不是把winner扔到challenger原来的低rank位置。随后把原base从rank 1到原rank(c*)的
**已有exact-label score slots**按该新顺序稳定赋值；其它label score逐位不变。每个受影响
label \(g\) 的所有physical rows统一加offset
\(S_g^{final}-B_g\)，从而保持label内row差值和顺序；C128外rows byte-identical。
HOLD仍返回原5413-row tensor对象本身。独立validator重做exact-label reducer，确认新tensor
精确产生上述ranking。它与旧“按local Z重排全部C64”本质不同。

---

# 11. Ownership 与因果记账

hypothesis和component reducer是log-sum-exp，故简单的raw patch和通常不等于\(E_g\)。
对每个方向同时把空间坐标\(z_i\)与独立非空间坐标\(-P_{u,k}\)从0缩放到真实值；
在\(t=0\)时全部9个槽位logit均为0，因此label evidence严格为0。定义双方向平均的
Aumann--Shapley空间贡献：

\[
u_i(g)=\frac12\sum_{d\in\{A\to B,B\to A\}}
\int_0^1
\sum_{u,k}q_{u,k,d}^{(g)}(t)\,z_{i,u,k,d}^{(g)}\,dt,
\]

其中\(q_{u,k,d}^{(g)}(t)\)是沿缩放路径由固定hypothesis prior与component prior得到的
归一化责任。非空间复杂度贡献单列：

\[
u_{MDL}(g)=\frac12\sum_d\int_0^1
\sum_{u,k}q_{u,k,d}^{(g)}(t)\,[-P_{u,k,d}]\,dt.
\]

积分固定使用32点FP64 Gauss--Legendre节点/权重，不允许按结果改变。注册闭合容差为
`abs<=1e-10 OR rel<=1e-8`；超过即停止，不能宣称数学exact。必须满足：

\[
\sum_i u_i(g)+u_{MDL}(g)=E_g,
\qquad
\omega_i(g,c)=\lambda[u_i(g)-u_i(c)],
\qquad
\sum_i\omega_i(g,c)+\omega_{MDL}(g,c)=\lambda(E_g-E_c),
\]

其中\(\omega_{MDL}=\lambda[u_{MDL}(g)-u_{MDL}(c)]\)只进数值ledger，禁止涂到
任何query patch。

pixel intervention先在clean target-free forward中固定一对\((g,c)\)（通常为获准challenger
与base winner）和方向，所有连续量都使用“\(g\)-vs-\(c\) margin从clean到corrupt的下降”
这一单位。另分四账，不能用\(\sum\omega_i\)冒充真实删除响应：

1. **\(D_B\)：** \(f_0(B_g-B_c)_{clean}-f_0(B_g-B_c)_{corrupt}\)；
2. **\(D_L^{fixed}\)：** 固定clean pair、components、H与assignment后，
   \((E_g-E_c)_{clean}-(E_g-E_c)_{corrupt,fixed}\)；
3. **\(D_{total}^{margin}\)：** corrupt后完整重编码、完整图库base、自然C128、重新提案与
   验证，但仍读取同一固定pair的连续\(m_1\) margin下降；若pair成员退出corrupt C128，
   其local evidence按部署合同置0，不做oracle reinsertion；
4. **\(D_{rehyp}\)：**
   \(D_{total}^{margin}-D_B-\lambda D_L^{fixed}\)。

candidate重新选择、SWITCH/HOLD变化和rank变化单列为离散\(D_{action}\)，绝不进入上述
连续closure。

ownership因果资格要求正区域删除与负区域删除都有足量样本，预测符号与真实pixel
干预方向一致，并严格胜同面积random、candidate-binding C与spatial P。该干预只用于
离线验证，不进入训练target。

---

# 12. 数据与训练监督

## 12.1 数据角色

```text
internal-development 987：过去775/212均已参与历史开发，统一降级为optimization；
五折只用于identity/supergroup-disjoint OOF内部资格，不能称fresh；
fresh：当前未注册；只有A0/内部复现全GO后才可另行授权；
opened：当前零访问；
sealed：当前零访问；
历史废弃实验数据：只有明确改为optimization role、完成identity/supergroup leakage审计后可复用；
external animal/medical：Route A fresh GO后才允许，不能参与主线选择。
```

所有角色写入 `RC/registry/data_roles.json`，路径、identity、supergroup、source acquisition和
hash不可缺失。

## 12.2 无需新增拍摄的合法增强

现有约130个药盒identity可用于：

```text
known perspective/affine warp；
随机crop/occlusion；
blur、glare、color和print/scan退化；
result-blind背景合成；
尺度和局部可见性变化。
```

这些增强只用于几何等变、local density ratio和outlier训练；不能冒充自然query检索证据。

## 12.3 Negative 分布

NCE/no-match negatives必须覆盖：

```text
natural retrieved hard rivals；
same-string/different-package；
相似logo/layout；
candidate-binding derangement；
reference-patch derangement；
target-absent/background patches；
随机gallery negatives（仅作为一部分）。
```

只用随机负例会学成“药盒 vs 无关图”，不能解决exact-instance。

---

# 13. 训练阶段

## G0：Known-warp geometry/local pretraining

监督：已知合成warp、共享query/reference局部encoder、无人工mask。

目标：

```text
local descriptor equivariance；
known correspondence；
line/point geometry；
match/no-match NCE；
outlier calibration；
glyph布局一致性。
```

## G1：Natural exact-pair hypothesis training

先完整产生natural candidate context与prejoin ledger，再join训练target。

exact pair只进入loss；不向部署candidate set插入target。训练correct reference hypothesis胜过
natural hard rivals，并训练candidate-level H0。

正式C/P/H销毁实例不进入训练；训练derangement使用独立hash族。

## G2：No-regret retrieval calibration

base-correct样本约束final margin不低于base安全预算；base-wrong样本推动target跨越当前
winner。优先使用约束优化而不是扫描loss权重：

```text
minimize rescue/rank loss
subject to base-correct break budget <= frozen contract；
dual只在optimization identities更新；
标签只进入loss，不进入forward hypothesis/CAL。
```

若需要可学习表示，只允许共享、identity-free、query/reference对称或明确asymmetric的低秩
local adapter；不增加mask head、selector、PPR或多层policy。

---

# 14. 固定 matched arms

所有arms共享candidate population、data、fold、seed、encoder budget和reducer。

| Arm | 作用 |
|---|---|
| B0 | 当前987-query冻结D1 base；A10仅作历史回归 |
| HR-A | 新高分辨appearance/glyph all-patch，无geometry |
| HYP-SEED | 只用proposal/inlier score，不用held-out verifier |
| HYP-SAME | proposal与verify同bank，量化自证 |
| HYP-XF | cross-fit geometry，无显式H0/no-match |
| HYP-NM | cross-fit+no-match，无query CAL exceedance HOLD |
| HYP-REAL | 完整OOF incremental calibration+CAL exceedance+HOLD |
| HYP-DERANGE | 同容量、同监督数量、固定错配candidate/correspondence |
| LOCK-ONLY | 纯local排序，仅诊断，不作为晋级主endpoint |

HYP-REAL必须胜过B0、HR-A、HYP-SEED、HYP-NM和HYP-DERANGE，才能把增量归因于
完整机制。HYP-SAME是故意允许拟合-验证自证的泄漏诊断arm，不要求HYP-REAL在raw
retrieval上胜它；要求HYP-SAME不得同时通过OOF、CAL-null与C/P因果资格，否则说明控制
失去区分力，整版停止。

---

# 15. 因果控制与CAL reference必须分开

## 15.1 正式因果控制

```text
C：candidate binding shuffle；
Pq：query coordinates permutation；
Pr：reference coordinates permutation；
H：geometry hypothesis换candidate；
T：glyph/text binding shuffle；
M：保持evidence magnitude，破坏candidate/endpoint方向；
Random：相同面积/数量的随机区域。
```

C、Pq、Pr分别过门，不能写“C或P”。

## 15.2 经验CAL reference distribution

经验CAL distribution严格使用第10.1节`CAL_NULL_V1`整体atlas错绑operator。它保持：

```text
candidate slots与base gaps逐位不变；
unique-component count与reference-token/aspect stratum；
完整descriptor/coordinate/glyph multiset；
固定9个hypothesis slots；
full max-selection pipeline。
```

普通空间permutation是重要因果销毁，但不自动形成有效统计null。CAL reference与正式C/P
控制必须使用第10.1节不同namespace、hash seeds和置换实例；stratum balance、orbit size或
完整atlas移动任一审计失败，action只能HOLD。即使组合审计通过，也不升级为exchangeability
或显著性声明。

---

# 16. 一次性资格漏斗

## E0：工程fixture（无科学结论）

必须检查：

```text
known H1 recoverability；
H0/no-match返回0 evidence；
固定8个invalid H1 slots返回0，all-H0 marginal精确为0；
完整reference-token multiset等倍复制不变性；任意单token复制不宣称不变；
byte-identical reference component collapse与physical-row duplicate invariance；
canonical hash collision fail-closed与near-duplicate不合并；
candidate reorder equivariance；
query-only A/B+halo exclusion；P-reference只进proposal、V-reference只进verification；
Aumann--Shapley spatial+MDL在冻结容差内closure，且MDL不出现在patch图；
同一固定pair连续四账与离散D_action分离；
base-independent E在任意B/rank flatten/swap下不变；
f0严格单调、保号、过原点，lambda=0逐pair恢复base order；
CAL_NULL_V1整atlas derangement、orbit、namespace和seed replay；
SWITCH move-to-front的5413-row projection与label reducer重放；
joint visual+glyph calibrator不允许独立LLR重复相加；
whole-query CAL max-exceedance plus-one score；
exact HOLD对象/bytes不变；
C和P是不同operator；
target label变化不改变prejoin hash；
CPU FP64 independent replay。
```

## E1：16-query result-blind runtime smoke

只输出吞吐、峰值显存、finite、resume和hash，不输出target/rescue/break科学指标。

若资源失败，只允许优化streaming、batching、cache I/O；禁止更改科学公式。

## A0：一次987-query、五折OOF、optimization-only冻结资格

不得扫描candidate K、H数量、sigma、text权重、threshold、CAL exceedance cutoff、scale或gate。
固定seed 17。每个fold都从其train identities训练fold-local D1、local representation、\(T_B\)与
\(\lambda\)，只在该fold heldout identities产生endpoint。五折拼接后再算下面所有门；
任何in-sample D1/local score禁止进入科学门。
当前registry只有fold-0 D1 checkpoint；A0授权前必须按冻结recipe为fold1--4物化fresh
D1 checkpoint与heldout full-gallery scores并独立验证。缺任一fold时只能做工程smoke或
fold-0 212-query内部screen，绝不能标五折A0。

硬门：

1. C128对base-wrong自然target coverage ≥90%；
2. legal target-known H0/H1 headroom至少30个base-wrong、覆盖≥8 identities；
3. joint verifier在identity-OOF match/no-match上NLL严格胜prior-only与appearance-only；
   sampling-prior correction有receipt；10个equal-mass bins的ECE≤0.05，校准slope在[0.8,1.2]；
4. frozen target-free target-vs-deployed-rival正确率≥65%；
5. A→B/B→A hypothesis/winner agreement≥80%；
6. wrong-reference/target-absent false action≤1%，receipt明确该状态只由postjoin evaluator定义；
7. HYP-REAL相对B0至少8 rescues、最多2 breaks、net≥6；
8. 正向覆盖≥5 identities和5 supergroups，单一identity不超过40% net rescue；
9. HYP-REAL identity-macro margin严格胜HR-A与HYP-SEED；
10. full incremental score OOF log-loss严格胜冻结base-only；
11. C、Pq、Pr分别移除≥50% local increment；
12. Aumann ownership closure通过；正、负region各覆盖≥8 queries和≥4 identities；
13. pixel intervention sign agreement≥60%，且严格胜同面积random/C/P；
14. \(D_B,D_L^{fixed},D_{total}^{margin},D_{rehyp}\)连续四账闭合，\(D_{action}\)单列；
15. opened/sealed/A10-historical运行时访问数均为0。

任一失败，停止本版本，不补丁、不扫参数。

## Phase B / Fresh P0

只有A0全部GO才允许新授权：

```text
固定同一五折、再跑两个预注册seed 29与43（不改公式/阈值）；
三个seed均至少4/5 folds同方向；
base-correct break率≤1%；
group-cluster CI下界>0；
随后才允许一次fresh P0。
```

当前合同不授权Phase B、fresh、opened或sealed。

---

# 17. Fresh授权前的证据条件

当前没有统计预测模型或独立重复足以给出“成功率超过50%”的数值；任何55--65%都只能是
未经校准的项目主观判断，禁止写入论文claim。只有A0实际证明以下三个事实，才允许重新
评估是否值得授权后续formal/fresh，而不是自动宣称成功：

1. 新高分辨appearance/geometry/glyph在给定base后仍有条件信息；
2. legal target hypothesis具有足够headroom且target-free可捕获；
3. exact HOLD把wrong-reference false action与base breaks压到硬门内。

在A0之前，文献和历史结果只能支持“值得进行一次严格证伪”，不能支持数值成功概率或
成功保证。

---

# 18. RC目录与artifact合同

本路线全部新增内容必须位于：

```text
RC/plan
RC/protocols
RC/registry
RC/src
RC/programs
RC/tests
RC/slurm
RC/logs
RC/results
RC/receipts
RC/cache
RC/artifacts
RC/reports
RC/tmp
```

Slurm脚本必须显式：

```text
#SBATCH --output=<RC>/logs/<job>-%j.out
#SBATCH --error=<RC>/logs/<job>-%j.err
```

结果只写 `RC/results/<immutable_stage_id>/`。历史外部cache只读使用时，registry必须保存：

```text
absolute path；
logical/physical SHA；
shape/dtype/count；
producer/version；
role；
禁止写标志。
```

不得修改home文件、shell profile、Codex配置或历史Route A结果目录。
未来runner必须以`registry/upstream_inputs.json`中的role生成stage-specific read allowlist；
`historical_regression_only`（尤其A10 opened224）、opened与sealed路径在训练、candidate生成、
阈值和模型选择进程中硬拒绝。仅写registry文字不算隔离，preflight和validator必须分别
记录实际opened/sealed/A10 read count为0。

---

# 19. 每query prejoin ledger

target join前必须封存：

```text
query/resource hash；
full-gallery base hash；
natural C128 exact labels与physical rows；
reference atlas hashes；
A/B split与exclusion halo；
proposal matches；
H0/H1 hypotheses、退化原因和hash；
held-out sufficient statistics；
row/label marginal receipt；
255 CAL draw keys与operator ids；
candidate action statistics；
opened/sealed access receipt。
```

prejoin API不得接受target identity。标签只在sealed ledger生成后由独立evaluator join。

---

# 20. Stop rules

以下任一发生，立即停止当前版本：

```text
C128 coverage不足；
新features不胜base-conditioned OOF与注册controls；
target/rival geometry不可区分；
H0 false action超门；
cross-fit两方向不一致；
target-known headroom不足；
rescue≤break或net低于门；
C/P任一不移除增量；
local gain只来自same-string语义；
Aumann--Shapley closure失败；
total response被rehyp/reallocation完全支配；
任何opened/sealed提前访问；
任何target-before-prejoin泄漏。
```

停止后不得在同一population上调整：

```text
K、top-M、H数量、sigma、visible prior、text权重、CAL exceedance cutoff、residual scale、gate。
```

任何科学变化必须新版本、新合同、新result-blind pre-registration。

---

# 21. 文献依据与严格边界

## 21.1 组件先例

- [OnePose, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Sun_OnePose_One-Shot_Object_Pose_Estimation_Without_CAD_Models_CVPR_2022_paper.html)：reference-defined unseen-object matching先例；但需要reference scan/SfM，不能证明本任务。
- [DELG, ECCV 2020](https://research.google/pubs/unifying-deep-local-and-global-features-for-image-search/)：global retrieval加local instance evidence的两级范式。
- [RoMa, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html)：foundation语义特征之外需要细粒度高分辨local特征。
- [SuperGlue, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html)：correspondence与point-level unmatched建模先例；dustbin不等于candidate-level H0。
- [GlueStick, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Pautrat_GlueStick_Robust_Image_Matching_by_Sticking_Points_and_Lines_Together_ICCV_2023_paper.html)：点线联合匹配，适合包装边线与低纹理平面。
- [SuperPoint, CVPRW 2018](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html)：known homography/homographic adaptation提供无人工点监督。
- [MAGSAC++, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html)：robust H1 proposal；内部model score不是部署likelihood。
- [Multiple Homography Ill-Solved, CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/html/Szpak_Robust_Multiple_Homography_2015_CVPR_paper.html)：自由multi-H兼容性风险，支持A0只做H0/H1。
- [Noise-Contrastive Estimation, AISTATS 2010](https://proceedings.mlr.press/v9/gutmann10a)：通过data-vs-noise分类估计未归一化density ratio；不保证部署分布校准。
- [Density Ratio/Class Probability, ICML 2016](https://proceedings.mlr.press/v48/menon16.html)：分类概率与density-ratio的条件联系。
- [Permutation p-values, Phipson & Smyth 2010](https://pubmed.ncbi.nlm.nih.gov/21044043/)：支持随机permutation的plus-one约定；由于本方案atlas orbit的exchangeability未证实，只借用数值约定，不借用p-value保证。
- [MaxT permutation control, Rempala & Yang 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3873102/)：multiple-testing控制依赖permutation validity与额外条件。
- [Integrated Gradients, ICML 2017](https://proceedings.mlr.press/v70/sundararajan17a.html)：支持路径积分与completeness思想；分数闭合不等于pixel因果性，因此本方案另做四账干预。
- [SeeTek, WACV 2022](https://www.amazon.science/publications/seetek-very-large-scale-open-set-logo-recognition-with-text-aware-metric-learning)：text-aware open-set logo检索先例；不证明OCR语义等于ownership。
- [RefPose, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kim_RefPose_Leveraging_Reference_Geometric_Correspondences_for_Accurate_6D_Pose_Estimation_CVPR_2025_paper.html)：reference hypothesis/render-and-compare范式启发；本方案无3D pose保证。

## 21.2 不得夸大的claim

文献只支持各组件是合理算法先例。没有任何一手工作证明：

```text
单张完整dieline reference；
无框自然照片；
5,404 unseen exact identities；
无人工空间监督；
还能够target-free纠正强base。
```

因此本方案的每个组件都必须作为可证伪机制，而不是最终增益必然存在的证明。
`ASLO-XF`、candidate-level H0、固定9-slot marginal、atlas empirical max-exceedance、
exact HOLD和Aumann--Shapley ownership是本项目提出的组合，不是上述任一论文已经验证的
方法或保证。

推荐论文边界：

> Prior work establishes algorithmic precedents for reference-conditioned inference,
> global-to-local instance retrieval, match rejection, robust planar hypothesis
> fitting, classifier-based density-ratio estimation, and permutation-based
> multiple-testing procedures under explicit validity assumptions. Our atlas CAL
> only borrows the plus-one convention and supplies no testing guarantee. None
> establishes the full Route A contract. RC-LTR–ASLO-XF
> therefore tests, rather than assumes, whether these components yield new
> target-free exact-instance evidence beyond the frozen full-gallery base.

---

# 22. 当前授权与下一步

截至本文冻结：

```text
PLAN_WRITTEN = true
CONTRACT_WRITTEN = true
ENGINEERING_FIXTURE_RUN = false
SLURM_SUBMITTED = false
A0_RUN = false
PHASE_B_AUTHORIZED = false
FRESH_AUTHORIZED = false
OPENED_AUTHORIZED = false
SEALED_AUTHORIZED = false
HOME_FILE_CHANGE = false
```

下一次用户明确恢复项目后，唯一允许动作是：

```text
E0_SYNTHETIC_ENGINEERING_FIXTURE
```

E0只验证数学与工程合同，不产生科学结论。

---

# 23. 条件性扩展：多药盒共现与 primary target 不明

这不是对当前 single-primary exact-instance benchmark 的重定义。当前配对 identity
仍是训练损失和单目标检索 evaluator 中的 primary label；D1、C128、H0/H1、P/V、CAL
和 SWITCH/HOLD 公式均不改变。该扩展解决另一个产品层事实：一张自然照片可能同时包含
多个图库药盒，而无额外交互时视觉本身不能推出用户意图中的“主要药盒”。

## 23.1 推理输出必须分开

每一个 candidate reference 都已经独立形成 `candidate-conditioned target hypothesis`。
因此 local forward 应为每个 candidate 输出以下**target-free**状态，而不是先把其余药盒
叫作背景：

```text
ABSENT_OR_UNVERIFIED
VERIFIED_PRESENT
MULTI_PRESENT
PRIMARY_UNDETERMINED
```

`VERIFIED_PRESENT` 只表示：该 candidate 的至少一个合法 H1 在 A→B 与 B→A 中均获得
candidate-vs-H0 的正验证证据。它不表示该 candidate 是训练 pair 的 target，也不允许
加入 target label、mask、box、point、polygon 或 oracle candidate。

当两个或以上 distinct exact labels 都满足该 target-free presence 合同时，forward 记录
`MULTI_PRESENT`。若系统没有用户给出的指向性输入（语音对象、触摸点、裁剪、历史上下文）,
则禁止把多个真实 presence 中的任一个解释为“用户 primary target”；single-label policy
应输出 `PRIMARY_UNDETERMINED` / HOLD，而不能以新阈值强行选一。

CAL 继续只授权一次单目标 SWITCH；C/P/Pq/Pr 是冻结模型的离线因果控制，不得作为每条
query 的在线 action condition。

## 23.2 当前 A0 的新增 ledger（不触发训练或模型选择）

在 local forward 完成后、target join 前，为每条 query 保存：

```text
verified_present_exact_labels
verified_present_count
candidate_hypothesis_ids_and_region_coordinates
two_direction_presence_flags
CAL_eligibility_and_coverage
single_primary_action
multi_present_flag
primary_undetermined_flag
```

这些是诊断 receipt，不修改 base ranking、local loss、candidate width、H1 数、sigma、
CAL 阈值或 action gate。评测 join 后另报 primary label 是否位于该集合，但不得把这个
诊断反向用于当前 A0 训练或调参。

## 23.3 条件触发的后续实验，而非现在新增 arm

仅当冻结 A0 ledger 实际出现 `MULTI_PRESENT` 或 `PRIMARY_UNDETERMINED` 时，才授权
一个独立的 `MTU`（multi-target/uncertain-intent）审计：

1. 固定 A0 模型、feature cache 和 candidate context，不重训、不调阈值；
2. 对触发样本由人工记录可见药盒实例与“是否存在可恢复的用户 primary 意图”；人工标注
   仅作离线 evaluator/数据集设计，绝不回流当前模型空间监督；
3. 分开报告 presence precision/recall、multi-label set recall、single-primary abstention
   precision 和用户意图不可辨识率；
4. 若需要把交互信息（触摸/语音/裁剪）用于 primary selection，必须在新的外部数据集、
   新合同中训练和验证，不能使用本 A0 population 的触发样本调参；
5. 该审计不会把 single-primary retrieval NO-GO 改写成 multi-object success，反之亦然。

这保证“图中真的有多个药盒”不被误记成背景噪声，也保证无法由图像识别的用户意图不会
被伪装成 ownership 失败。
