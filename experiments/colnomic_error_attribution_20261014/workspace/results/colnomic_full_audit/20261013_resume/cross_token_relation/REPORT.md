# 冻结 cross-Token 关系特征头

本实验直接对完整 Query×Reference 原始 FP32 点积矩阵读取38维关系摘要，包含双向最佳/次优匹配分布与匹配质量聚集。与先分别池化或kernel mean后再配对不同，关系在汇总前被计算。它仍是有限关系摘要，不是完整 Token 集合信息测试。

输入没有坐标、原Token索引、图像ID或身份ID；ID只用于数据连接、划分和评价。全部非零Token（包括模板Token）参与，未重新归一化。精确并列的匹配质量均分，避免索引顺序成为特征。

原2234受限配对用于训练/验证/测试；训练身份独立，标准化只用train，C=1.0与阈值只在验证身份选择。1648自然困难对照覆盖1117查询，另有3个训练标签打乱对照。历史测试已被观察，不能称新的盲测。

全部1117×5405图库记录均保存38维float32关系特征。额外count-only基线输入4维显式有效Token数代理（log q数、log r数、绝对log差及q/r比），独立train标准化、val选C=0.1，并有3个null。两类共8个头均保存全图库标量、稳定排序和SHA设计排名。未插入真值、未限制top-K、未用bounds代替未算scalar。

|头|split|N|R@1|R@5|R@10|MRR|修复|损害|净变化|
|---|---|---|---|---|---|---|---|---|---|
|relation|all|1117|0.076992|0.263205|0.374217|0.172438|0|784|-784|
|relation|test|161|0.093168|0.279503|0.372671|0.195498|0|110|-110|
|shuffle_17|all|1117|0.001791|0.001791|0.001791|0.002507|0|868|-868|
|shuffle_17|test|161|0.006211|0.006211|0.006211|0.006997|0|124|-124|
|shuffle_31|all|1117|0.000000|0.000000|0.000000|0.000810|0|870|-870|
|shuffle_31|test|161|0.000000|0.000000|0.000000|0.000688|0|125|-125|
|shuffle_53|all|1117|0.000895|0.000895|0.002686|0.002296|0|869|-869|
|shuffle_53|test|161|0.006211|0.006211|0.012422|0.007623|0|124|-124|
|count_only|all|1117|0.000000|0.000000|0.000000|0.000491|0|870|-870|
|count_only|test|161|0.000000|0.000000|0.000000|0.000553|0|125|-125|
|count_only_shuffle_17|all|1117|0.000000|0.000000|0.000000|0.000752|0|870|-870|
|count_only_shuffle_17|test|161|0.000000|0.000000|0.000000|0.000748|0|125|-125|
|count_only_shuffle_31|all|1117|0.000000|0.000000|0.000000|0.000650|0|870|-870|
|count_only_shuffle_31|test|161|0.000000|0.000000|0.000000|0.000432|0|125|-125|
|count_only_shuffle_53|all|1117|0.000000|0.000000|0.000000|0.000496|0|870|-870|
|count_only_shuffle_53|test|161|0.000000|0.000000|0.000000|0.000463|0|125|-125|

候选范围的差异：relation在原受限test配对的准确率为0.944099，自然困难test的235对为0.765957，但全图库出现上表中的大幅损害。完整图库结果不支持将这个通用配对头作为检索改进。

训练配对只覆盖91个Reference ID，有效Token数范围448–770，图库最小为47。relation的1014/1117赢家数量超出训练范围；数量控制头的1个最高Reference ID覆盖全部查询。gallery_candidate_shift_audit.json保留各头赢家分布。这里只报告训练支持范围与数量差异，不能断言下降全部由数量导致。该盘点发生在固定模型评分之后，不据此重选参数。

独立数值验收：passed，20333项。完整评分直接头重放误差1.7763568394002505e-15；原mean MaxSim最大误差1.7881393432617188e-07；稳定排名与SHA轴不一致数0。另以固定种子在32对原始Token矩阵执行独立CPU关系摘要重算，最大误差1.7881393432617188e-07。

本头评估自然身份配对及完整图库，不把结果升级为自然同文字布局证据；文字内容、包装面及其它可见因素仍可能共同作用。Probe成功不证明原MaxSim使用信息，失败不证明特征没有信息。自然共同短语ROI Probe另行记录额外ROI监督与整图上下文。

## 复现与只验入口

两个小显存矩阵worker最终按query index奇偶分配不重叠查询，原缓存只读。前段硬件基准后因两卡持续速度相近调整分区，已有提交复用；rebalance_receipt.json保留该记录，未按模型效果分配。每worker allocator上限约675MiB，CPU1线程，FP32禁TF32。分区锁、逐Query提交与receipt支持断点续跑。重放特征需使用实际运行的冻结源码快照，使提交与receipt源码SHA一致；CPU训练和验证使用当前模块。

```bash
PYTHONPATH=/mnt/lsdf_zyin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 /home/jwei/miniconda3/envs/rebut/bin/python -B results/colnomic_full_audit/20261013_resume/cross_token_relation/source_executed_fb515f5c8425.py --stage features --batch-size 32 --partition-count 2 --partition-pattern 0,1 --partition-id 0 --cpu-threads 1
PYTHONPATH=/mnt/lsdf_zyin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 /home/jwei/miniconda3/envs/rebut/bin/python -B results/colnomic_full_audit/20261013_resume/cross_token_relation/source_executed_fb515f5c8425.py --stage features --batch-size 32 --partition-count 2 --partition-pattern 0,1 --partition-id 1 --cpu-threads 1
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.cross_token_probe --stage aggregate
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.cross_token_probe --stage train
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.cross_token_probe --stage verify
```

`restricted_*`、`natural_hard_*`与`full_gallery/`保留不同候选范围。`feature_file_manifest.json`、`source_executed_*`、各worker runtime/receipt与`independent_validation.json`提供对应证据。
