# new HYP：reference 支持 max–min 的文献边界

2026-09-10。范围仅为三份原始论文/作者讲义与已冻结的
[TRAIN pilot 计划](../plan/RC_REFERENCE_SUPPORT_MAXMIN_TRAIN_PILOT_V1_20260910.md)。
本次未读取自然样本缓存、角色、标签、模型输出或容量证书，也未修改试验。
这是写作定位，不是 pilot 结果或新方法建议。

**当前算子可解释为：在固定候选集合内，为每个 query/reference 求一个
非负归一化的 token 支持，使其对最难竞争 reference 的线性 margin 最大。
其 max–min、LP 与原/对偶证书来自标准优化理论，不能单独列作新理论贡献。**

## 三份原始来源

| 来源 | 原问题及依据 | 与本轮的关系和差别 |
|---|---|---|
| Murray、Perronnin，[GMP 原文](https://arxiv.org/pdf/1406.0312)，CVPR 2014，§3–4、式(5)、(10) | 使各 patch 与池化表示的相似度趋于相等；正则化最小二乘及 patch Gram 矩阵给出加权表示。 | 支持“普通均值可能掩盖少量有用 patch”的一般动机；其权重由单图 descriptor 集合决定，并非本轮的逐 reference 最坏竞争 margin。原公式也没有本轮的非负 simplex 约束。 |
| Jégou、Zisserman，[Triangulation Embedding and Democratic Aggregation 原文](https://www.robots.ox.ac.uk/~vgg/publications/2014/Jegou14/jegou14.pdf)，CVPR 2014，§4、式(13)–(17) | 对局部描述子赋正权，使各项对集合自相似度的贡献相等，条件为 ΛKΛ1=C1。权重只依赖该 descriptor 及其所在集合。 | 同属优化池化权重；其目标是平衡集合内部贡献，不是让指定 reference 同时胜过全部其它候选。本轮不保证贡献均衡，也不要求每个 token 权重严格为正。 |
| Ghosh、Boyd，[Minimax and Convex-Concave Games 讲义](https://web.stanford.edu/class/ee392o/cvxccv.pdf)，Stanford EE392o，2003，§1 | 有限零和矩阵博弈的混合策略、最坏收益与 LP 表达。 | 本轮 D 矩阵、token 分布 p 与竞争者分布 α 可直接套用这一标准结构；不能把该代数变换或强对偶本身写成新发现。 |

上述比较只覆盖这三份来源，**不能据此宣称当前完整检索机制首次提出**。

## 本轮算子的准确含义

沿用计划的 D[h,i]=a_g(i)−a_h(i)，h 遍历固定 C128 的127个其它 reference：

    v_g = max_{p∈Δ_tokens} min_{h≠g} D[h,:]p
        = min_{α∈Δ_others} max_i (Dᵀα)_i

这是标准矩阵博弈在本任务的代入。可称“相对于当前候选集合的最坏竞争
线性区分”，其中“鲁棒”仅指这些有限竞争者；不意味着对新图像、域偏移、
错标、像素攻击或候选遗漏都鲁棒。p 依赖 query、被评分的 g 和整个 C128，
不是一次独立编码即可固定的图像级 GMP/democratic 描述子。

对任意可行 p、α，标准弱对偶给出

    min_h D[h,:]p ≤ v_g ≤ max_i (Dᵀα)_i。

因此本工程的精确可行分布与上下界检查，是给定有限表示类的数值证据。
正下界证明该候选存在可区分支持；非正上界排除此表示类中的正 margin。
它们都不等于目标身份预测成功，更不保证保留原28正确或产生额外救回。
相关数学依据为上述 [Ghosh–Boyd §1](https://web.stanford.edu/class/ee392o/cvxccv.pdf)；
对 D、p、α 的具体任务解释是本注根据冻结计划作出的推导。

## 不与 mean-other 或固定支持混用

下列区别是定义推导，不是新增试验：

- 固定支持的 J_g：先用同一 p_g 汇聚每个完整 reference，再取最强其它 reference；没有优化 p。
- 当前 max–min：优化同一个 p，使它同时面对全部127个完整 reference；没有逐 token 拼接“组合错误 reference”。
- 若改成 mean-other：目标为 pᵀ[a_g−mean_{h≠g}a_h]。在仅有 simplex 约束时，其最优值就是最大 token 系数，存在单 token 最优解。它可胜平均竞争者却输给最强竞争者，不能称为本轮 max–min。

对每个固定 p，最坏竞争 margin 不大于平均竞争 margin，故两者分别取
最大后仍有 v_maxmin≤v_meanother。这是标准不等式，不是新 HYP 的新定理。
max–min 本身同样不保证支持连通、稠密、对应物理物体或唯一用户意图。

论文当前可准确写成：**“我们在冻结的完整候选集合上，利用标准矩阵博弈
优化与原/对偶界，检验自由内容匹配表示是否存在 reference 条件化的非负
区分支持。”** 可讨论的任务贡献需来自这种表示、职责分配及后续检索证据；
不能由已知 LP 代数、GMP/democratic 的既有成绩或 TRAIN 可分性替代。
