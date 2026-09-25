# Route A 最终方案：Reference-Conditioned Latent Target Hypothesis

## 0. 一句话定义

训练时只给模型：

- 自然 query；
- 正确 reference；
- D1 找到的相似错误 references。

不给：

- mask；
- box；
- point；
- polygon；
- 人工 crop；
- identity-specific 参数。

模型必须自己学会：

> **在完整 query patch grid 中，哪些局部证据能够被某个 reference 解释；这些局部证据组成该 reference 的 latent Target Hypothesis；然后由另一条未参与 proposal 的证据通道验证该 hypothesis，最终让正确 reference 胜过相似错误 references。**

核心不是“先分割，再检索”。

而是：

> **retrieval supervision 自己产生 target。**

---

# 1. 整体系统

```text
Natural Query
      │
      ▼
完整 Query Patch Grid
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
Proposal 通道 P              Verification 通道 V
      │                          │
Candidate Reference g            │
      │                          │
      ▼                          │
Reference-conditioned            │
Local Support                    │
      │                          │
      ▼                          │
K 个 Latent Target Hypotheses    │
      │                          │
      └───────────────►──────────┘
                      │
                      ▼
              Cross-fit Verification
                      │
                      ▼
                 Candidate Evidence Eg
                      │
                      ▼
        与 natural hard candidates 比较
                      │
                      ▼
              Correct Reference?
```

最终部署再接：

```text
D1 global prior
+
Target-Hypothesis local evidence
        │
        ▼
     SWITCH / HOLD
```

---

# 2. 第一原则：不预先分区域

这是硬合同。

禁止：

```text
SAM
MaskCut
CutLER
人工 mask
人工 bbox
objectness teacher
人工 crop
预训练 detector 输出作为 target
```

Query 始终保持完整 patch grid：

```text
q1 q2 q3 ... qP
```

Reference 同样：

```text
r1 r2 r3 ... rJ
```

所谓“target region”不是输入。

它必须是模型根据：

```text
Query
+
Reference
```

自己产生的结果。

---

# 3. Proposal：Reference 如何提出 Target Hypothesis

对于任意 candidate reference `g`：

```text
Query patch qi
        ↓
和 reference g 的所有 local tokens 比较
        ↓
得到“qi 被 g 支持多少”
```

得到一张 candidate-specific support field：

```text
s1(g)
s2(g)
...
sP(g)
```

例如：

```text
桌子 patch       低
背景             低
药盒边缘         中
药盒 logo        高
药盒文字         高
手               低
```

关键：

```text
这还不是 target mask。
```

它只是：

> candidate `g` 对完整 query 提出的“哪里值得进一步解释”的软证据。

---

# 4. Latent Target Hypothesis

Proposal 层不输出一个自由 mask。

它输出固定数量：

```text
H0 + K 个 H1
```

建议第一版：

```text
H0 = candidate absent / no-match
K = 8
```

每个 H1 是一组 query patches：

```text
Hg,1
Hg,2
...
Hg,8
```

这些 patch 必须：

```text
共同受到 g 支持；
在 query 上具有局部连贯性；
在 reference 上也有对应支持；
数量规则对所有 candidate 完全相同。
```

但：

```text
不要求矩形；
不要求平面；
不要求 homography；
不要求完整物体可见。
```

因此可以表达：

```text
药盒正面的一部分；
猫的脸+身体局部；
医学图像中的局部异常区域；
任意形状的部分可见目标。
```

---

# 5. 为什么不能继续用 M0 Homography

当前 M0 做的是：

```text
先提出固定 H1 homography
→ 再验证这个几何变换。
```

而我们的目标应该是：

```text
Reference support
→ 自己形成 target hypothesis
→ geometry 只是后续 verification 的一种证据。
```

M0 在 190 个 `D1 wrong + target in C128` 样本中：

```text
HYP-SEED：69/190
完整 real HYP-XF：54/190
local top1：0/190
```

而 C-binding / P-spatial destruction 几乎不改变结果。

所以当前具体 `homography + synthetic G0 verifier + full-reference marginal` 不能继续作为主模型。

---

# 6. P/V 两条证据通道

这是防止自证的核心。

## Proposal P

只负责：

```text
“我认为 target 可能是这些 patches。”
```

例如使用：

```text
冻结 DINOv2 中层 local descriptors
```

它可以：

```text
计算 query-reference local similarity；
产生 support field；
生成 K 个 hypotheses。
```

但它的 raw score：

```text
不能直接进入最终 candidate score。
```

## Verification V

另一套证据只负责回答：

> **P 提出来的这组 patches，真的能够支持这个 reference 吗？**

可以使用另一套 frozen representation，例如：

```text
另一 DINO layer；
或独立 local matcher；
或不同冻结 descriptor bank。
```

硬合同：

```text
V 不重新选择 target patches；
V 不读取 P 的 raw confidence；
P 不读取 V 输出；
P/V 无共享 trainable adapter。
```

---

# 7. Cross-Fit 怎么做

不是把图切成左右两半。

同一张完整 query 产生两套 registered evidence：

```text
A
B
```

路径一：

```text
A 提 hypothesis
B 验证
```

路径二：

```text
B 提 hypothesis
A 验证
```

即：

```text
A → H_g^A → B 验证
B → H_g^B → A 验证
```

最终：

```text
candidate evidence = 两个方向平均
```

Cross-fit 的作用只有：

> **避免“我因为这几个 patch 分高，所以选它们；然后又因为这几个 patch 分高，所以证明我选对了”。**

它不保证统计独立。

---

# 8. Verification 到底学什么

这是整个新方案最重要的改变。

当前 M0 verifier 主要学的是：

```text
synthetic known-homography token correspondence
vs
shifted token no-match
```

而我们真正需要学：

> **自然 query 中，这组 local evidence 是否真正支持 reference y，而不是一个非常相似的错误 reference c。**

所以训练必须直接使用：

```text
自然 query
+
正确 reference y
+
natural hard references c1,c2,...
```

---

# 9. 训练数据

每个训练 episode：

```text
Query x

Positive:
    true reference y

Negatives:
    D1 hard rival
    D1 top-K near duplicates
    same-layout / same-color / same-text-style candidates
    independent local-retrieval hard negatives
    small number of random negatives
```

禁止：

```text
oracle mask
oracle region
target insertion at inference
identity-specific head
```

训练时当然可以知道：

```text
哪个 reference 是正确的
```

因为这本来就是 retrieval pair supervision。

不知道的只是：

```text
Query 中 target 在哪里。
```

---

# 10. Multiple Instance Learning

模型知道：

```text
(Q, y) 是正 pair
(Q, c) 是负 pair
```

但不知道：

```text
Q 中哪个 patch 属于 y。
```

因此：

```text
所有 query patches = bag
latent hypotheses = possible witnesses
```

训练要求：

```text
正确 reference y：
至少应该存在一个或几个 hypothesis，
能够被独立 Verification 稳定确认。

错误 reference c：
即使能找到偶然相似局部，
最终 hypothesis evidence 应该更低，
或者 H0/no-match 获胜。
```

---

# 11. Candidate Evidence

每个 candidate 独立做：

```text
g
↓
Hg,1 ... Hg,K
↓
Verification
↓
每个 hypothesis 得到 signed evidence
↓
与 H0 一起 marginalize
↓
得到 Eg
```

重点：

```text
Eg 是 candidate-specific evidence。
```

而不是旧 TK1V3 的：

```text
F(g;c)
```

Competition 放在最后：

```text
Ey
vs
Ec1
vs
Ec2
...
```

---

# 12. H0 / No-match

这一层必须有。

否则任何错误 reference 都能说：

```text
“我在背景里总能找到最像我的一个 patch。”
```

所以每个 candidate 都必须允许：

```text
H0：
Query 中没有一组足够可靠的局部证据支持我。
```

正确 candidate：

```text
某个 H1 > H0
```

错误 candidate：

```text
H0 > 所有 H1
```

或者至少：

```text
Ec < Ey
```

---

# 13. 训练 Loss

主 loss 两个。

## A. Candidate ranking

要求：

```text
正确 y > natural hard candidates
```

即：

```text
Ey 应该高于 Ec1, Ec2, ...
```

使用 candidate-listwise cross entropy。

## B. Presence / No-match

要求：

```text
Ey 正
Ec 负或接近 H0
```

这样错误 reference 不能只靠：

```text
找到一个局部相似碎片
```

获得高 evidence。

---

# 14. 防止只盯一个判别碎片

MIL 最大风险是：

```text
500mg
logo
一个角
```

就足够区分两个药盒。

所以要求 hypothesis 的验证证据来自多个局部 patch，而不是单个极端 patch。

可使用固定的：

```text
minimum effective support
```

或 evidence entropy / effective-sample-size 门。

只能在 train OOF 冻结，不能看 heldout 后调。

---

# 15. Candidate Recall 单独解决

Target Hypothesis 和 candidate recall 是两个问题。

最终 candidate set：

```text
C = D1 candidates
  ∪ independent local-retrieval candidates
```

第二个 source 也必须 target-free。

HYP 接受任意 candidate reference，然后问：

> **它是否真的能解释 Query 中某个 target hypothesis？**

---

# 16. D1 最终保留

D1 继续承担：

```text
强 global prior；
默认答案；
full-gallery ranking；
fallback。
```

最终：

```text
D1 winner = w
HYP best challenger = c
```

只有当：

```text
c 的 independently verified evidence
足以跨越 D1 原来的 gap
```

才 SWITCH。

否则 HOLD，并完整返回原 D1 score tensor。

---

# 17. Ownership

Target Hypothesis 负责：

```text
哪里可能是 target。
```

Ownership 负责：

> **为什么这些 patches 支持 candidate y 而不是 challenger c？**

最终 ownership 必须解释：

```text
Ey - Ec
```

而不是另外训练一张热图。

---

# 18. Unseen Identity

模型不能有：

```text
identity embedding
per-product classifier
per-reference trainable parameter
```

它学的是共享规则：

> **给我任意 Query 和任意 Reference，我怎样判断 Query 中哪些局部证据能够被这个 Reference 稳定解释。**

新 identity 只需要提供 reference，无需重训。

---

# 19. 动物和医学的机制泛化

机制不写：

```text
药盒
文字
平面
homography
矩形
```

只写：

```text
Query patches
+
Reference patches
+
shared matching rule
```

因此数学接口可迁移；不同模态允许使用不同冻结 encoder。

---

# 20. 与历史实验的区别

## B4/C1/C2

```text
直接预测 patch ownership
→ ownership 加权 retrieval
```

问题：score 可以绕过 ownership。

## TK1V3

```text
g vs c
→ 找最判别 cell
→ erase
→ functional surplus
```

问题：wrong candidate 也能找到 functional region。

## M0

```text
固定 homography hypotheses
→ synthetic G0 verifier
→ full-reference 1024-token marginal
```

问题：proposal 弱，natural verifier 不成立，C/P 不影响 score。

## 新方案

```text
Natural retrieval pair supervision
        ↓
Reference-conditioned latent hypothesis
        ↓
Independent verification
        ↓
Natural hard-candidate competition
```

且 local score 无法绕过 hypothesis。

---

# 21. 实验顺序

## Stage 0 — Engineering

确认：

```text
无 target 输入；
P/V 真分离；
hypothesis ID 可重放；
candidate reorder equivariant；
H0 可工作；
gradient 有限；
所有 candidate same-capacity。
```

## Stage 1 — Conditional Target Determination

只在 target 自然进入 candidate set 的 population 上测试：

> 不知道哪个 candidate 正确时，HYP 能否把 target 排在 hard competitors 前？

必须胜：

```text
all-patch retrieval-only
same-capacity no-HYP
query-only spatial control
candidate-binding destruction
spatial destruction
```

## Stage 2 — Candidate Recall

独立证明正确 reference 有足够概率进入候选。

## Stage 3 — No-Regret Retrieval

```text
D1 + HYP evidence
→ SWITCH/HOLD
```

报告：

```text
rescue
break
wrong-to-wrong
net rescue
false-action rate
```

## Stage 4 — Ownership / Pixel Causality

验证高 ownership 区域删除是否比随机同面积区域更破坏正确 candidate margin，并检查 candidate/spatial destruction 是否销毁 local increment。

---

# 22. 最终 GO

至少同时满足：

```text
1. Conditional target determination > matched retrieval-only
2. Identity-disjoint OOF 成立
3. Correct candidate hypotheses 胜 hard wrong candidates
4. H0 能拒绝错误 reference
5. Candidate-binding destruction 显著破坏效果
6. Spatial destruction 显著破坏效果
7. Final net rescue > 0
8. Break 受控
9. Pixel intervention 方向正确
10. 新 identity 无需重训
```

---

# 23. 总结

最终机制不是：

> **先找到物体，再做 retrieval。**

而是：

> **只有 retrieval pair supervision；模型通过正确 reference 与 hard wrong references 的竞争，自己发现 Query 中哪些局部 patches 能稳定解释正确 reference。这些 patches 形成 latent Target Hypothesis，另一条证据通道验证它。**

最终链：

```text
Retrieval supervision
        ↓
Reference-conditioned local support
        ↓
Latent Target Hypothesis
        ↓
Cross-fit Verification
        ↓
Candidate Evidence
        ↓
Hard-candidate Competition
        ↓
Safe D1 Correction
```
