# 后 LLM 内部 M 改造：H593 原五折结果

原 ColNomic 自然 C128；冻结编码器与检索投影；仅训练后 LLM 适配器和不直读 M 的 INTERNAL3 头。每折从零输出适配器与本折 TRAIN 头开始，固定8轮；没有复用旧 TRAIN16 的参数。

| 方法 | 正确 / 593 | 救回 RAW | 误伤 RAW | 净增 |
|---|---:|---:|---:|---:|
| RAW | 426 | 0 | 0 | +0 |
| POST_REAL | 478 | 54 | 2 | +52 |
| EXTERNAL_INTERNAL3 | 426 | 0 | 0 | +0 |
| EXTERNAL_ADDITIVE4 | 481 | 58 | 3 | +55 |
| EXTERNAL_PRODUCT5 | 481 | 58 | 3 | +55 |
| POST_REAL_CONSTANT | 426 | 5 | 5 | +0 |
| POST_REAL_SHUFFLED | 369 | 8 | 65 | -57 |
| POST_CONSTANT | 406 | 5 | 25 | -20 |
| POST_SHUFFLED | 429 | 8 | 5 | +3 |

POST_REAL_CONSTANT / SHUFFLED 为同一个训练完的 POST_REAL 模型仅在推理时更换内部 M；POST_CONSTANT / SHUFFLED 为等预算单独训练的对照。EXTERNAL_* 为本轮各折用实际 fresh_L0 从 TRAIN 训练的输出端头，不能与历史481混称同一次重放。

全部593张都计入，23张 target 不在自然 C128。RAW426、召回570与原来源一致。先逐项验证全部128分、127 challenger logits、HOLD=0、strict>0切换、候选轴和快照，再读取held curator标签；独立NumPy最大分数差为 5.68e-14。

配对分组 bootstrap 区间、逐折结果、逐查询候选 logits、救回/误伤清单与内部M对照均在result.json中。该面板此前已反复用于开发，不能称为新外部确认；内部有效不以超过外部头为必要条件。

[完整结果](../results/rc_postllm_h593_v1/result.json) · [独立验证](../results/rc_postllm_h593_v1/validation.json) · [读标签前封存](../results/rc_postllm_h593_v1/all_predictions_prelabel_seal.json)
