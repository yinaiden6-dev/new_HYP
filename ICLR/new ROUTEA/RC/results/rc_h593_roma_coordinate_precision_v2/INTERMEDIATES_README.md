# 中间文件索引

本目录留存整个H593坐标精度实验。每个query单独queryNNN/；每16候选一个part00.pt..part07.pt，成功分片不覆盖，支持恢复。query000是无标签选择的完整C128资格试运行。

- workers.json：593 query顺序、源图SHA、原RAW/ColNomic缓存和原RoMa缓存的路径/摘要；source_index确定缓存内原query。冻结query/reference图和tokens通过这些原文件复用。
- queryNNN/partXX.pt：128候选轴的一段。每对coarse_matcher保留粗网格双向warp/confidence；四臂arms包含query_visibility/reference_visibility、四项原评分、12次坐标量化诊断；intermediate包含下述逐token证据。
- intermediate.query_coordinates/reference_coordinates：原坐标图尺寸、token单元bbox/有效mask、单元中心坐标、FP64单元均值/协方差/越界比例、64×64坐标抽样。坐标位于对侧EXIF-oriented图归一化[-1,1]空间。64×64仅抽样，不能称全部1280稠密输出。
- intermediate.content：自由/可见性加权MaxSim的reference token索引和数值、被选位置的原cosine、每个query token对最终real_score的贡献。
- queryNNN/payload.json：自然C128和全库RAW顺序、四臂逐候选四统计、127×6完整头输入、原RAW winner和挑战者轴、parts SHA。
- queryNNN/validation.json：每候选四臂独立FP64评分、原生逐位复现、MaxSim命中与贡献回放。
- ../rc_h593_roma_coordinate_eval_v2/foldN/payload.json：完成后保存所有模型参数及每个query的全部127分数；HOLD固定0。
- ../rc_h593_roma_coordinate_eval_v2/coordinate_diagnostics.json、result.json、validation.json：全部593完成后生成汇总及验证。

读取part使用torch.load(path,map_location='cpu',weights_only=True)。本次不删除旧缓存，不重新编码ColNomic；不是完整高分辨率隐藏激活归档。Embedding融合尚未启动；其缓存规范见plan/RC_ROMA_FEATURE_COLNOMIC_FUSION_FEASIBILITY_20260922.md。
