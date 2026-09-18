# V9暂停：不能将局部排查或数学可能性当作已证实根因

2026-09-09。用户明确质疑：尚未找到主要失败的真实原因，就开始新的V9。
这一质疑成立。此前诊断有确定结果，但没有闭合“主要错误由哪一个因素
导致、修复该因素能恢复识别并支持HYP”的因果链。

V9已暂停。只有runner与launcher草稿，core及独立core E0尚不存在，没有
execution authority，没有提交V9训练。运行代理已中断，草稿保留供审阅；
不得将合成检查点测试称为新方法资格或自动继续提交。
状态凭据：`registry/rc_cycle_metric_v9_draft_hold_20260909.json`。

## 哪些是已经知道的，哪些还不知道

- 确知：不同支持直接比较会产生局部方向反转；完整旧family没有新增target
  certificate query；缺失/分组均值单独不足以解决剩余wrong-positive证据。
- 确知：共同query测度下仍有17个wrong区域证书；严格唯一query数仍11。
- 不知：身份信息主要在哪一步损失；硬对应、支持遗漏、当前特征、读出容量
  与训练目标分别贡献多少实际错误。
- 不知：跨通道度量是否能修复真实错误。秩一矩阵反例只证明某种可能性，
  没有在这些自然失败样本上建立因果支持，因此不足以直接启动V9。

## 本轮追到的一处确定的信息路径差异

已对照旧冻结bundle的SHA验证源代码：

`programs/materialize_romav2_colnomic_current_runtime_bridge_roma_shard_v1.py`
从RoMa预测取的是`overlap_AB`、`overlap_BA`，形成query/reference可见性权重。
下游`programs/run_romav2_colnomic_visibility_xf_six_case_v1.py`的`score(q,r,wq,wr)`
先构造全部query/reference ColNomic token cosine矩阵，然后对每个query
token取`max_j(cos_ij * wr_j)`，再按wq聚合及visibility mass缩放。

所以旧27/32、69/90系统的这条下游内容评分使用可见性加权MaxSim；它没有
把RoMa warp指定的reference token当作必须遵守的身份配对。
相对地，当前P V6/V8身份项在RoMa指定的reference token上计算，并新增
区域约束、独立排序及新的损失。这是尚未验证能继承旧收益的信息路径改变。

该源码事实指出了具体未闭合的接口问题，但**尚不能宣布它就是P失败主因**。
还需固定真实query、candidate、token和观测支持，区分硬对应与内容匹配
读出的影响，而不是继续同时更换多个因素。

RoMa overlap本身可以携带候选相关和空间信息，因此也不能由此宣称旧系统
完全不使用几何、原收益只是运气，或所有HYP不可能。

源码事实及源哈希：`results/rc_hyp_information_path_correction_v1/result.json`。
本轮未执行自然模型推理、未提交新任务、未消费正式392、未进入V/action或
访问D1-MI。当前仍未实现HYP GO；四小时目标不能把未知因果或失败改成成功。
