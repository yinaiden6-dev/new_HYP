# 单图表示与配对计算：CPU诊断的范围

六臂的 `SINGLE` 不是“所有单图表示方法”：`rc_pair_quality_core_v1.py` 让query、reference分别产生逐token标量，再取 `M(q,r)=sqrt(a(q)b(r))`。同query相对比较M、M×L时，query共同乘数抵消（忽略原1e-12 epsilon）。因此SINGLE较弱首先约束这一标量接口，不能排除单图高维向量本身有额外的身份区分线索。

真实可用缓存为 `results/rc_h593_feature_fusion_cache_v1/ready.json`，绑定5,324张单图；每张含原ColNomic128、coarse_11/17各1024维、六种fine特征、原cell坐标及有效掩码。coarse/fine提取程序明确禁止matcher/refiner pair前向。这些特征已池化到ColNomic原patch格，完整原生RoMa空间激活未保存；不能仅凭该缓存重演原生双图Transformer的所有内部作用。

最小新增读出使用既有71张执行序号0..70，保留每张自然128候选：固定8×8再池化，比较同维度源的全局均值余弦、正反MaxSim，以及同一相似度矩阵的互近邻、唯一命中覆盖和soft往返支持。不训练任何参数。原1024维粗特征不经过六臂使用的128维随机投影。池化ColNomic是同池化协议的内容对照。

解释顺序：先看原始单图表示经简单配对是否已能区分正确候选，再看同一矩阵里结构统计是否与native M同向。后者只是描述性证据；结构统计与M相关不说明原matcher使用了它，也不能把MaxSim相近的两个候选当作全部外观已配平。新读出失败也不能证明表示缺信息。

避免重复：原先已有常数M、候选错绑、reference主效应分解、J/A/P入口置零、低通/打乱以及粗阶段提前退出。新读出针对的是此前SINGLE没有覆盖的“单图向量保留到配对比较”替代解释。

结果及协议在 `feature_readout/`。该71面板已经打开；本轮只保证新分数先封存再join完整标签，不声称全盲独立测试。
