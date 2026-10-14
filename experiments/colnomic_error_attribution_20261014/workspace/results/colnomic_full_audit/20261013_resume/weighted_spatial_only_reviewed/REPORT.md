# 审核框下纯空间 Matching × Query-only 加权的 2×2

γ=0、λ=0.1 的纯空间对照已经完成，不能由 γ=0.2、λ=0.1 的带对齐奖励结果替代。本轮全部 1117 查询保留完整 5405 图库候选界；η=0.2 仅由 173 张验证查询、21 个身份的 package-macro R@1、MRR、较小η次序选择。原始 Matching 的加权η也为其既有验证选择0.2。

|Matching|汇总|γ|λ|η|正确/1117|R@1|身份macro R@1|MRR|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|原始MaxSim|uniform|0|0|0|870|0.778871979|0.790922913|0.833983800|
|原始MaxSim|query-only weights|0|0|0.2|871|0.779767234|0.789960627|0.834603531|
|纯空间penalty|uniform|0|0.1|0|844|0.755595345|0.759429097|0.813125651|
|纯空间penalty|query-only weights|0|0.1|0.2|845|0.756490600|0.759487372|0.814889536|

Query-only权重为：每张查询内部，距查询Token均值的特征距离严格超过75%分位的Token取1+η，其余取1，并除以权重和。其输入只有冻结查询向量，不用坐标、OCR、框、身份或GT。纯空间Matching只在已经接受的审核框/OCR homography候选上，对主体Token使用FP32 dot−λ×clipped spatial distance后取MaxSim；没有γ奖励，其他Token/无H候选回退原始Matching。

## 修复/损害的比较基准

|比较|修复|损害|净增正确|118个source group配对macro ΔR@1|95% bootstrap CI|
|---|---:|---:|---:|---:|---|
|原始加权 vs 原始uniform|5|4|1|−0.001060145|[−0.008367862, 0.005043328]|
|纯空间加权 vs 纯空间uniform|3|2|1|0.000064201|[−0.005934129, 0.005341551]|
|纯空间uniform vs 原始uniform|5|31|−26|−0.026504487|[−0.041466796, −0.012940561]|
|纯空间加权 vs 原始加权|3|29|−26|−0.025380140|[−0.038363199, −0.013008727]|

另以原始uniform为共同基准，纯空间加权为6修复/31损害，而非上表“相对各自uniform”的3修复/2损害。原始uniform对照不混用。bootstrap按118个source group配对重采样5000次、seed=20261013；身份macro指标使用130个package，与source-group macro是不同汇总单位。

全量micro R@1的2×2交互为0：两个Matching下加权均净增1张。source-group macro交互为0.001124346，95% CI [−0.004010114, 0.006881313]。这些结果不能支持加权带来稳定R@1增益；纯空间penalty对全量的损害也没有被此汇总方式消除。验证与测试分开指标及逐图转移保存在CSV中，没有按测试调整η。

γ=0.2、λ=0.1的旧奖励控制仍单独保留：uniform R@1=0.826320501、weighted R@1=0.825425246。它们是奖励+空间组合，不是纯空间2×2的两个格。

## 实际计算与独立验收

无H候选从weighted_original和weighted_spatial_reviewed的exact缓存union逐位复用246412对；同Query mask与所有重叠无H标量均相同。全部20329个接受H候选重新计算纯λ dot/weighted maxima，没有从F32加γ后的maxima简单减γ。再补计算15231个可能影响top10或GT排名的无H候选。实际新计算35560对，19,592,956,649次Token向量点积；评分耗时362.71秒。只用了GPU3数学运算、TF32禁用、显存fraction0.1、CPU线程2，无encoder重跑；Torch峰allocated127147520B、reserved312475648B。

最终281972/6037385候选对有5η精确标量；5755413低界候选的标量继续显式NaN，每个候选保留有限上下界。η=0复用审核后完整纯空间FP32均值，因此原全1117 GT rank/top1与reviewed_boxes的B03 spatial_only逐位来源一致。每个η的top10、最佳GT rank、全部GT SHA重复行和稳定并列顺序由精确候选+严格界认证，不声称算出了全部加权图库标量。

independent_verification.json实际独立重放5585个排名、全部6037385候选界、宏指标、修复/损害和5000次配对bootstrap；最小未计算候选排名裕量1.9282338059589677e−8，独立上下界重建最大误差0。另以无padding的CPU直接dot/penalty实现检查63对、全部5η，最大分差0。2234个原缓存NPZ SHA和两个原rawbank SHA重新检查通过。

## 重放入口与边界

独立验收：`python scripts/colnomic_mechanism/weighted_spatial_only_reviewed.py --verify-only`。已有complete与partial检查点可恢复；本次GPU评分已完成，验收不启动GPU。

公共loader为 `load_spatial_inputs(manifest_path, d, selection_key='combined')`。纯空间调用显式传 `selection_key='spatial_only'` 或 `B03.spatial_only`。旧reviewed_boxes manifest保留原SHA；本目录spatial_scoring_manifest.json声明纯空间参数，并复用冻结scores/H/box/selection来源。loader_regression_validation.json核实默认combined全部6037385均值逐位不变，纯空间两种manifest调用均逐位一致。

审核框含已查看测试图的人工作辅助，不能将本结果称为无GT自动框方法或历史盲测。冻结Token/scoring可复核，历史图库神经编码的外部权重/来源缺失仍单独保留。

详细文件：validation_grid.csv、metrics.csv、factorial_2x2_and_reward_controls.csv、factorial_effects.csv、factorial_interaction_bootstrap.csv、per_photo_factorial_transitions.csv、independent_rank_certificates.csv、independent_CPU_math_checks.csv、workload_completed.json与每query的scores_and_bounds.npz/complete.json。
