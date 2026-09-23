# 冻结内容、区分表示来源与双图质量读出

用户在 2026-09-23 授权：先复查历史和未释放实验，再补必要区分。历史与重复性核对见 `reports/REPORT_H593_PAIR_QUALITY_HISTORY_AND_OVERLAP_20260923.md`。

## 问题与最小对照

保持原始 ColNomic tokens、自由完整 reference 内容分数 L0、自然 RAW C128、H593 原分组五折及 HOLD=0。只改变生成整体质量 M 的方法。四个学习臂：COL_ONLY/SINGLE、COL_ONLY/PAIR、COARSE/SINGLE、COARSE/PAIR。两个同训练预算对照：NONE（M=1）、ROMA（冻结原生 M）。COST1 为唯一损失，seed17，不选最好配置，不增加其他候选或损失搜索。

COL_ONLY 和 COARSE 均复用已封存 feature_fusion_cache。COL_ONLY读取原128维ColNomic token并作固定LayerNorm，不再加入随机通道投影；COARSE读取既有DINOv3第11/17层的原token格点池化和固定128维投影。质量输入等宽，原始ColNomic内容tokens始终不修改。COARSE负结果仅限制该池化/投影，不证明原始DINO或RoMa隐状态不含信息；两来源的预训练和池化差异必须披露。

共享质量网络：520→16→1，tanh 后 sigmoid，FP64，8353参数。PAIR 从完整有效双图相似度矩阵得到温度0.1的双向 softmax（无top-K），每个token保留自身128维、对侧soft-context128维、差128维、积128维，以及本地坐标2、匹配期望坐标2、位移2、归一化熵1、双向返回质量1。SINGLE 使用相同网络、参数量和单图输入宽度，将context换为自身、匹配位置换为自身、熵/位移为0、返回质量为1。两者均在逐token非线性质量之后汇总双侧均值的几何均值M。

这学习的是固定完整软关系上的质量读出，不声称重新实现完整RoMa或学习了全部attention权重。它没有硬几何合法门、atom family、selector、ownership或对应监督；也不拟合RoMa质量作为teacher。

评分器只使用 M×L0。共享五参数action读取 [RAW标准化差, sym(M L0), sym(M), sym(L0)] 和偏置。先在各折TRAIN用当前初始化质量拟合头2000 full-batch updates，再固定2000次query级联合更新；AdamW质量lr=.001、头lr=.003、wd=.001，初始化拟合头lr=.03。所有臂同一折TRAIN顺序、固定循环洗牌、完整128候选。每次更新先计算所有候选，再对COST1严格非零VJP候选反传，不做负例采样。NONE/ROMA同样进行2000次query级头更新；旧481只作历史对照，不能代替同预算对照。

## 执行与留存

已有5324图片特征全部复用，新增编码器或RoMa前向为0。先合成独立数值/梯度/等变/恢复检查，再对query000完整C128做无真实标签容量测试，临时参数丢弃。若任一学习臂单query更新时间>30秒或峰值显存>30GiB，停止扩展并报告成本，不通过删候选/降精度隐藏成本。

容量合格后执行6臂×原5折，共30配置，最多4个GPU任务并行；10分钟分片，450秒停止领新工作并保存，按配置续跑，最多128片。该并行限制是新分支预算，不更改既有融合队列。每折训练仅读取该折TRAIN角色；全部预测封存后才join held身份。593全部计分，23个target absent不排除。训练集拟合和OOF结果分别报告，单seed仅作开发证据。

保留质量网络、优化器、头、每25步记录及全部128候选M/L0/分数、两侧全部token质量、关系几何的八项均值和原token来源。完整softmax矩阵及520维soft-context由不可变来源、坐标与固定温度精确重建，不逐折重复落盘TB级上下文；明确区分实际保存和可重建中间量。旧结果、在运行源码与科学authority不修改。

预定比较：两来源分别PAIR−SINGLE，COARSE_PAIR−COL_PAIR，以及COL_PAIR对NONE/ROMA，共五项。报告准确率、MRR、rescue/break、64组件bootstrap区间及五项Holm配对检验；逐图检验是辅助，组区间为主要泛化描述。失败不能证明所有ColNomic读出不可能，成功也不自动证明与RoMa采用相同视觉线索。候选绑定破坏只在冻结预测期执行，不能当空间ownership证据。
