# POST-LLM 内部模型：GroZi480 冻结外部迁移

固定 H593 fold0 POST_REAL 及其配套头与 TRAIN 标准化；无外部训练、调参或挑折。
完整480张、27个视频组；自然C128包含target410张，另外70张仍计入分母。

| 模型 | 正确/480 | MRR | 救回 | 误伤 | 净增 | RAW正确误伤率 |
|---|---:|---:|---:|---:|---:|---:|
| RAW | 321 | 0.710784 | 0 | 0 | +0 | 0.00% |
| POST_REAL | 339 | 0.734302 | 18 | 0 | +18 | 0.00% |
| POST_REAL_CONSTANT | 314 | 0.704013 | 2 | 9 | -7 | 2.80% |
| POST_REAL_SHUFFLED | 308 | 0.698073 | 4 | 17 | -13 | 5.30% |
| NO_ADAPTER_INTERNAL3 | 304 | 0.691910 | 2 | 19 | -17 | 5.92% |
| EXTERNAL_ADDITIVE4 | 342 | 0.738689 | 24 | 3 | +21 | 0.93% |
| EXTERNAL_PRODUCT5 | 342 | 0.738689 | 24 | 3 | +21 | 0.93% |

相对RAW：救回 18、误伤 0、净增 18。
等视频权重增益95% bootstrap区间：[0.00949667616334283, 0.03787831253109027]。

这项结果检验一个预先固定的内部模型能否跨数据集使用真实M；是否有效按本次真实结果解释，不预先宣称GO。
恒定/错绑是同一冻结模型的输入干预；无适配器和外部头均来自同一个源域折。
MRR按最终选中候选置首、其余图库候选保持原RAW次序计算；误伤率分母为RAW原正确321张。
此前已使用该外部面板；本次不称全新未触碰确认，也不建立区域IoU、分割精度或ownership结论。

[完整结果](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_postllm_grozi_external_v1/result.json) · [独立核算](/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC/results/rc_postllm_grozi_external_v1/validation.json)
