# GroZi480 内部模型冻结迁移：完整实验归档

上一份完整实验归档是 `aeeeeb42dcee3635aeebea82866267113387c44d`；随后 `82cb2889214ea5db34c89e4a91a3634968ef5139` 上传了论文、统一解释和 GroZi 汇总。本次补齐该汇总所对应的完整实验记录，保留 workspace 原相对目录及文件字节。

## 直接入口

- [完整实验目录](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/)
- [480张逐候选预测与保存的中间状态](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/predictions/)
- [60片验收](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/shards/)
- [预测完成后的标签读取前封存](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/all_predictions_prelabel_seal.json)
- [原始结果](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/result.json)、[科学验收](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/validation.json)、[独立复算](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/operations/independent_final_recount_20260925.json)
- [冻结程序快照](../../ICLR/new%20ROUTEA/RC/results/rc_postllm_grozi_external_v1/frozen_protocol/)
- [实验报告](../../ICLR/new%20ROUTEA/RC/reports/REPORT_POSTLLM_GROZI_EXTERNAL_V1_20260925.md)、[提交及修复过程](../../ICLR/new%20ROUTEA/RC/reports/REPORT_POSTLLM_GROZI_EXTERNAL_SUBMITTED_20260925.md)
- [上次归档后实验增量审查](experiment_delta_audit.json)
- [逐文件清单及排除项](files.json)、[导出完整性验收](validation.json)
- [独立导出审查：源文件、Git导出、封存链与程序绑定](independent_export_validation.json)

## 实验与结果

固定既有 H593 fold0 的 POST_REAL、配套头、TRAIN 标准化和0切换阈值，使用 GroZi 原 ColNomic 自然 C128。外部不训练、不调参、不挑折。480张全部纳入，来自27个视频；410张的target在C128，另70张仍保留在分母。该外部面板此前已经使用。

| 模型 | 正确/480 | 对RAW救回/误伤 |
|---|---:|---:|
| RAW | 321 | 0/0 |
| 固定POST_REAL | 339 | 18/0 |
| 同模型恒定M | 314 | 2/9 |
| 同模型错绑M | 308 | 4/17 |
| 源fold0无适配器INTERNAL3 | 304 | 2/19 |
| 同源外部ADDITIVE4 / PRODUCT5 | 342 / 342 | 各24/3 |

这些是同一个预定源fold0实验束，不是原完整头的355、另一次简化头的349，也不是H593五折的478/593。

## 保留了哪些可分析数据

每张最终JSON包含六臂各自的128个内容分数、127个挑战者logit、128项动作分数，以及候选身份/顺序、头参数、特征、输入指纹、源模型指纹与最终动作。共368,640个候选内容分数、365,760个挑战者logit。480份partial状态另存，不将它们算成额外图像或最终预测。

同时保留每张图的编码等价检查与缓存SHA元数据、全部60片验收、原始封存代码、CPU身份轴修复代码和既有回归检查。v1汇总失败与v2修复记录均保留：5533张物理图库图片对应5532个身份，修复只处理验收的身份轴，未重新训练或改变预测。

本次不包含原图、模型/适配器二进制权重、token/隐藏表示payload、运行锁或err/out日志。`encoder_cache/`下只上传JSON等价性和指纹元数据；这些不是可替代实际token缓存的特征。外部推理或逐patch重算仍需原来保留的模型和缓存。

## 核验

从仓库根目录执行：

```bash
python tools/verify_grozi_internal_completion_20260925.py
```

该命令不调用模型或GPU，检查全部导出文件哈希、480张完整轴、六臂分数长度、60片封存链和汇总引用。它是归档验收；原独立NumPy核算和分组统计验收按原文件保存。

历史归档清单仍对应其原提交。当前实验根中的完整字节清单以本增量为准；没有修改旧清单来伪造旧版本已包含本次数据。
