# 同一相位干预贯通外部与内部：固定模型机制回放

用户授权：2026-09-27「对，继续挖」。不改当前 QR/QR_VEC/QRR 训练，不新增 GPU 或重训。

## 固定范围

复用原 outcome-independent 相位面板8张、每张自然 ColNomic C128，12种 RoMa COARSE 输出。3张按纠错选择的病例不加入。使用每张原 H593 留出折的 COST1 NATIVE7、M×自由内容简化 COST1，以及 POST_REAL＋INTERNAL3 冻结端点；同图、同候选轴、同内容、同参数、同零切换阈值。

所有效应相对同一 COARSE NATIVE；HR1 原生只用于重放一致性检查。原头按 HR1 训练，COARSE 回放存在分布移动，不将小面板解释成新 H593 性能或独立泛化证据。

## 预先固定的对照

设原生 ell=log(M)，干预变化 d=log(M')−ell，a=mean_candidates(d)，r=d−a。

- FULL：exp(ell+a+r)，原实际干预。
- SCALE：exp(ell+a)，只改变所有候选共同尺度。
- RELATIVE：exp(ell+r)，只改变候选相对分布。
- NATIVE：原生 COARSE；ORIGINAL_HR1：原 H593 输入；CONSTANT：内部适配器条件置零。

全部11个非原生干预均做三种分解，不按结果挑相位、轴或符号。验算 FULL 的 logM 等于 SCALE+RELATIVE−NATIVE；不得裁剪超出[0,1]的合成M掩盖域外输入，如越界则阻断。恢复M必然恢复该唯一输入路径只是接口校验，不作为独立中介发现。

外部 NATIVE7 保持原局部内容与原滚动控制读出，所有带M的 real/query-control/reference-control score 同步按 M'/M 缩放，重新计算全部六项特征。简化头使用原自由内容与替换M。内部固定hidden、reference tokens、适配器、BF16投影和INTERNAL3，仅更换M；FP64归一化和MaxSim。

## 主要读出与验证

先封存所有候选输出，再接回已打开面板标签。主要对手按不变的自由ColNomic内容选择，不能按干预输出选择。计算固定target−wrong的logM、内部自由内容及各头logit差，另列target−HOLD、重新竞争后的决策。GLOBAL四方向与LOCAL四方向分别在输出后平均；不能先平均M再过非线性模型。

分别计算共同尺度与相对变化的二因素精确Shapley和交互项；这只是此分解基准下的计算路径贡献，不是唯一视觉根因或因果百分比。组bootstrap5000次，原小面板探索性区间，不将128候选当独立样本；不因区间跨零宣称等价。

保存每候选每世界的M、逐patch最大相似度及reference索引、L、127logit与HOLD=0、模型与源SHA。独立核算逐patch→L→头分数，校验原HR1/CONSTANT与已封存CPU回放，并抽查按预定候选位置0/64的完整适配器和MaxSim重算。候选边界续跑，无并发总文件写入。

## 执行和限制

2 CPU/查询，可4个查询在8 CPU单任务内并行，16GB，10分钟；候选级断点，预算退出续跑。只加载已有hidden与小投影，不加载LLM/RoMa。输出 `results/rc_m_phase_external_internal_bridge_v1/`。

本轮隔离M的使用，不能证明M信息充分、RoMa唯一不可替代或排除所有域外扰动解释。当前QR/QRR回答更丰富信息的可读出增量，属于不同问题；其正负结果均需与本轮区分。
