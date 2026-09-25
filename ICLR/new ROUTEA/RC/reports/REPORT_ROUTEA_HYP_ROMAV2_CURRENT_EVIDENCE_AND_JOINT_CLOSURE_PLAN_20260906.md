# Route A HYP：RoMaV2 当前证据、历史负结果与联合闭环方案

日期：2026-09-06

## 1. 总结判断

HYP 的正确含义是 `reference-conditioned latent target hypothesis`。

历史 ColNomic-coordinate H0、HYP-XF、固定 DINO/RGH readout 的负结果仍然有效，不能覆盖、改写或解释成成功。但这些结果否定的是特定的 proposal/readout family，不是对所有 reference-conditioned target hypothesis 的普遍否定。

截至目前，RoMaV2 已经给出两类重要的新证据：

1. candidate-specific reference 内容确实提供额外判别信息；
2. RoMa representation 中存在 pair-level、coordinate-sensitive 的局部证据。

尚未完成的是最关键的联合闭环：

> 在同一个冻结、identity/supergroup-disjoint 实验中，RoMa 生成的空间 witness 必须既能改善完整 target-free C128 决策，又必须在 candidate binding 或空间对应被破坏后失去该改善。

因此当前最准确的结论是：

```text
RoMa reference-conditioned spatial witness：已有内部 headroom
target-free deployment-level target hypothesis：尚未证明
spatial ownership：尚未证明
end-to-end exact-instance recognition：尚未证明
```

## 2. HYP 的严格定义

对自然 query 图像 `q` 和任意动态 candidate reference `g`，proposal 必须在不知道真实 target 的情况下生成：

\[
P(q,g)\rightarrow \{H_0,H_{g,1},\ldots,H_{g,K}\}.
\]

一个合法 hypothesis 至少包含：

\[
H_{g,k}=(Q_{g,k},R_{g,k},T_{g,k}),
\]

其中：

- `Q`：query 中的多 patch、连通、有面积区域；
- `R`：candidate reference 中与之对应的区域；
- `T`：query-reference 间可重放的 correspondence/assignment；
- `H0`：该 candidate 无法提供充分解释时的显式无 hypothesis 状态。

没有人工 mask 时，最终可严谨声称的是 `reference-generated latent witness`，而不是完整物体分割 mask。

## 3. 历史 H0/HYP 负结果仍然成立

历史失败的核心不是“完全没有局部相似度”，而是旧机制无法把局部相似度转化为可靠的 target-free spatial hypothesis。

代表性负证据包括：

- HYP-SEED：`69/190`；
- 完整旧 HYP-XF：`54/190`；
- D32：`18/32 → 14/32`；
- 固定 H0：`8/20`；
- coordinate-free ACTION_BAG：`16/20`；
- H0 相对 ACTION_BAG：`1 rescue / 9 break`；
- RGH V9B：`27/32 → 26/32`，`0 rescue / 1 break`。

旧 H0 的结构性问题是：

```text
ColNomic先猜一个坐标
→ 在该坐标或固定cell附近归一化/聚合
→ 用同一类局部相似度验证
```

坐标没有提供独立增量，错误 candidate 也能在相似文字、颜色或矩形结构上形成看似合理的区域。

正确的论文表述应是：

> 旧 ColNomic-coordinate H0 family 被正式否定；后续 RoMa correspondence 构成一个新的 reference-generated hypothesis family，必须独立评价。

不能写成“旧 H0 被新结果推翻”。

## 4. RoMaV2 相比旧 H0 新增了什么

RoMaV2 不再接受 ColNomic 给出的单一 predicted coordinate 作为答案。它直接读取自然 query 与 specific candidate reference，并输出稠密 cross-image correspondence、query visibility 和 reference visibility。

当前正确分工是：

```text
D1 / D1-MI
    → full-gallery ranking和自然C128

RoMaV2(q,g)
    → candidate-conditioned query/reference visibility
    → candidate-conditioned correspondence

ColNomic
    → 在相同visibility/support内提供exact-content evidence

NATIVE7
    → 对全部challengers进行target-free SWITCH/HOLD
```

旧的独立 DINO-V 不再是当前 RoMa 主线的必要模块。功能上的 proposal/verification 分离由以下方式实现：

- RoMa 负责提出 visibility/spatial witness；
- ColNomic 负责提供独立 identity evidence；
- NATIVE7 只做低容量 target-free action。

## 5. 已经成立的 RoMa 正证据

### 5.1 Fresh balanced-pair 空间门

RoMa balanced32 V2 的冻结结果是：

- pair correct：`29/32`；
- rescue/break：`14/1`；
- candidate maps distinct：`32/32`；
- REAL 胜 query-coordinate control：`25/32`；
- REAL 胜 reference-coordinate control：`27/32`。

结果路径：

`results/romav2_colnomic_visibility_xf_balanced32_v2/result.json`

这证明：

> RoMa representation 中已经存在 candidate-bound、coordinate-sensitive 的 pair-level evidence。

它仍是 postjoin balanced-pair 证据，不是完整 target-free C128 HYP 证明。

### 5.2 Full-negative target-free 内部 headroom

冻结内部 OOF 结果：

- ColNomic base：`24/32`；
- final HOLD/SWITCH：`27/32`；
- rescue/break：`3/0`；
- switch：`4`。

结果报告：

`reports/REPORT_ROMAV2_COLNOMIC_VISIBILITY_XF_INTERNAL_HEADROOM_20260831.md`

这说明 RoMa+ColNomic evidence 已能在匿名 challenger population 中形成少量 no-regret action headroom。

### 5.3 Difficult90 candidate binding

冻结 difficult90 回归结果：

| Path | Top-1 | Rescue | Break |
|---|---:|---:|---:|
| Base | 61/90 | — | — |
| REAL | 69/90 | 8 | 0 |
| C_BIND | 50/90 | 0 | 11 |
| R control | 69/90 | 8 | 0 |
| Q control | 70/90 | 9 | 0 |

结果路径：

`results/romav2_colnomic_difficult90_frozen_regression_v1/result.json`

由此可以确认：

- candidate binding 很强；
- REAL 的八个 rescue 在 C_BIND 下全部消失；
- 当前增益确实依赖 specific reference 内容。

但 Q/R 控制没有导致 deployment 退化，甚至 Q control 更高。因此 difficult90 只支持 candidate-bound evidence，不支持严格空间因果结论。

## 6. 为什么 target hypothesis 仍未证明

当前证据分别回答了两个问题：

```text
pair-level RoMa spatial evidence：有
target-free action headroom：有
```

但它们尚未在同一个冻结模型、同一个 held-out population、同一个完整 challenger tournament 中同时成立。

仍然无法排除：

- RoMa visibility 只是提供 candidate-specific 统计量；
- ColNomic 利用的是区域内 bag-of-patches 内容，而非坐标关系；
- 所谓 target 区域不连通、不完整或落在干扰物上；
- 打乱 query/reference 空间对应后，target-free action 仍然不变；
- 删除所谓 target 区域后，margin drop 不大于随机等面积区域。

所以当前允许的最强表述是：

```text
candidate-bound、target-present-conditional、
partially coordinate-sensitive exact-reference evidence
```

尚不允许：

```text
reference成功生成了target hypothesis
模型找到了完整目标区域
spatial ownership成立
```

## 7. 当前 N2 target-free RoMa 结果

当前 D1 条件下的 matched three-arm 结果：

- D1 base：`24/32`；
- `C_PAIRED REAL`：`26/32`；
- rescue/break：`2/0`；
- `C_PAIRED C_BIND`：`21/32`；
- REAL 严格胜过 D1 与 C_BIND；
- 但未超过冻结 RAW+C 的 `28/32`；
- MRR 未达到 `0.9014136904761905`；
- 每 fold 相对 RAW+C 非负门失败。

正式状态：

`N2_FRESH_D1_ACTION_CONDITIONAL_NO_GO`

结果路径：

`results/routea_n2_fresh_d1_matched_three_arm_postseal_evaluation_v1/result.json`

这不是“RoMa 没有 evidence”，而是：

> 当前 D1 条件下，RoMa evidence 尚不足以通过完整 no-regret action 门。

## 8. Candidate recall 仍是独立瓶颈

固定 hard-235 结果：

- fresh-D1 C128 target coverage：`190/235 = 80.85%`；
- deep-tail recovery：`0/22`；
- 状态：`N2_CANDIDATE_RECALL_NO_GO`。

RoMa mean-overlap full-gallery source E0a 也已因 query-independent hubness 停止：

`R5V6_ROMAV2_FULL_GALLERY_SOURCE_E0A_STOP_INDEPENDENTLY_VALIDATED`

这只否定 RoMa mean-overlap 作为 full-gallery candidate generator，不否定 RoMa 作为 candidate-conditioned verifier/evidence source。

Candidate recall 与 HYP 必须分开报告：

- target 不在 C128：candidate-source failure；
- target 在 C128 但选不对：target-free selection/HYP failure；
- candidate binding 过但空间控制不过：candidate-bound non-spatial evidence；
- spatial control 和 action 同时通过：才是 deployment-level spatial HYP。

## 9. 当前正确下一步：RoMa-TH Joint Closure

不再重新运行旧 RGH optimization-only atom coverage，也不恢复独立 DINO-V。

正确顺序是：

```text
D1-MI五折OOF
→ 冻结新的full-gallery ranking和自然C128
→ 按新的winner/challenger重新物化RoMa
→ 生成candidate-conditioned connected H_g(q)
→ ColNomic在相同support上提供identity evidence
→ NATIVE7对127个challengers执行一次SWITCH/HOLD
→ 同次运行candidate-binding与空间因果控制
```

### 9.1 Target hypothesis 对象

每个 `H_g(q)` 必须：

- target-free；
- 读取 specific reference `g`；
- 来自冻结 RoMa correspondence/visibility；
- 多 patch、四连通、有最小面积；
- 不使用人工 mask、框、target identity、rank 或 slot；
- 无合法区域时输出显式 H0/HOLD；
- 不允许 all-patch/global score 绕过 hypothesis bottleneck。

### 9.2 公平竞争

对于 challenger `g` 和 current winner `w`：

- 两个 candidates 必须在 byte-identical query support 上比较；
- 同时使用 `H_g` 与 `H_w` 两个 proposal owners；
- 不允许各候选只使用自己最有利的区域后直接相减；
- 所有 127 challengers 都必须评分；
- 最多执行一次 SWITCH，否则 exact HOLD。

### 9.3 同一实验中的三组门

#### Target-free action

- final Top-1 至少 `29/32`，严格超过 RAW+C 的 `28/32`；
- MRR 至少 `0.9014136904761905`；
- rescue > break；
- break ≤ 1；
- 每 fold 相对 RAW+C 非负；
- 严格胜过 D1-MI。

#### Candidate binding

- REAL 严格胜过 C_BIND；
- 至少一个 REAL rescue；
- C_BIND rescue retention < 1；
- 错绑 reference 后 target-free action 必须退化。

#### Spatial causality

- query-coordinate destruction；
- reference-coordinate destruction；
- joint/S11 topology destruction；
- selected-region pixel deletion；
- equal-area random deletion。

要求：

- REAL margin/action increment 严格大于每个空间控制；
- 各空间控制 rescue retention ≤ `0.50`；
- 删除 `H_g` 比删除等面积随机区域产生更大的 `g-vs-winner` margin drop；
- connected region 的变化方向与 action 变化一致。

只有 target-free action、candidate binding 和 spatial causality 三组门在同一冻结实验中全部通过，才能声明：

> RoMa reference-conditioned correspondence 生成了可部署的 latent target hypothesis。

## 10. D1-MI 与 HYP 的关系

D1-MI 不直接证明 target hypothesis。它负责修复当前 D1 对几乎所有 query token 进行近最大幅度修改的问题，并减少 held-out break。

D1-MI 当前只有 E0 工程资格：

`N2_D1_MI_E0_READY`

只有完成五折 OOF 并达到以下门，才允许重新物化 RoMa：

- Top-1 ≥ `749/987`；
- MRR ≥ `0.8133460700456977`；
- C128 coverage ≥ `942/987`；
- RAW→D1-MI rescue > break；
- break < 6；
- 每 fold 相对 RAW 非负；
- 表示漂移均值和 `>0.20` 比例低于 current D1；
- per-query drift CV 高于 current D1。

D1-MI GO 后必须按新 C128、新 winner 和新 challenger 重新计算 RoMa；不能直接复用 current-D1 的旧 RoMa local feature。

## 11. 当前结论

截至 2026-09-06：

```text
旧H0/HYP family：正式负结果，永久保留
RoMa candidate binding：内部正证据
RoMa pair-level spatial sensitivity：内部正证据
RoMa target-free action：有headroom，但严格conditional NO-GO
RoMa deployment-level target hypothesis：尚未证明
spatial ownership：尚未证明
candidate recall：NO-GO
D1-MI：E0工程READY，五折OOF未完成
论文级external confirmation：尚未完成
```

最核心的剩余问题不再是“RoMa 有没有空间信号”，而是：

> RoMa 的 candidate-conditioned 空间信号，能否在完整 target-free strongest-challenger 决策中成为不可替代的因果证据。

