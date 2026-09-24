# new HYP — 主路线研究归档

Reference-conditioned Joint Evidence Calibration for Fine-grained Retrieval.

2026-09-19 从原 workspace 整理。仓库保存源码、配置、理论、结果、展示材料及小头参数；大模型权重放在同一私有仓库的 **Release 附件**。原始 query/reference 图片另行备份，本仓库保留清单。历史实验特征缓存及 err/out/log 不上传。

[下载模型权重 Release](https://github.com/yinaiden6-dev/new_HYP/releases/tag/mainline-backup-20260919)：9 个原始权重文件，34.6 GB；37 个分片及清单均已通过 GitHub 服务端 SHA256 核验。

## 从这里阅读

- **[2026-09-24 ColQwen base完整结果、简单融合对照与蒸馏状态](FINAL_CONTROLS_20260924.md)**

- **[2026-09-24 ColPali 自有 C128：H593 五折完成，RAW 283 → MASS5 COST1 348 / CE 365](COLPALI_RESULTS_20260924.md)**

- **[2026-09-23 当前实验与结果：完成项、进行中分支及数据索引](CURRENT_EXPERIMENTS_20260923.md)**
- [ColNomic rerank 实验：脚本、历史结果、逐图预测和工作记录](experiments/rerank_colnomic/README.md)

- [2026-09-20 补充对照执行状态：六项消融与 learned ColNomic-only（完整结果待完成）](<ICLR/new ROUTEA/RC/reports/REPORT_H593_ADDITIONAL_CONTROLS_EXECUTION_V1_20260920.md>)
- [早期 query–reference 联合 RCDE：原始负结果、停止依据与协议修正](<ICLR/new ROUTEA/RC/reports/rcde_joint_history_20260920/README.md>)
- [完整中文结果总账](<ICLR/new ROUTEA/RC/reports/new_hyp_complete_results_20260916_v1/complete_results_zh.md>)
- [理论定义与命题](<ICLR/new ROUTEA/RC/reports/new_hyp_complete_results_20260916_v1/theory_zh.md>)
- [全部统计表 Excel](<ICLR/new ROUTEA/RC/reports/new_hyp_complete_results_20260916_v1/all_results.xlsx>)
- [processed128 结果](<ICLR/new ROUTEA/RC/reports/new_hyp_complete_results_20260916_v1/processed128_zh.md>)
- [展示图和讲义](<ICLR/new ROUTEA/RC/reports/figures/new_hyp_showcase_20260915_v1>)
- [COST1 药盒热图：12 张高清案例与下载包](<ICLR/new ROUTEA/RC/reports/figures/new_hyp_cost1_medicine_visibility_20260919_v1>)
- [药盒与商品可见性案例](<ICLR/new ROUTEA/RC/reports/figures/new_hyp_visibility_cases_20260915_v1>)
- [模型与数据恢复说明](RESTORE.md) · [执行入口与环境](REPRODUCTION.md)

## 当前机制

冻结 ColNomic 全图库检索产生自然 C128 → RoMa 生成候选条件下的软可见性 → ColNomic full-reference weighted MaxSim 与匹配质量共同形成六项证据 → 检索关系监督训练共享小头 → 对全部 127 个 challenger 进行 HOLD/SWITCH 决策。

本任务新增监督只有 query/reference 身份关系；不增加框、mask、点或对应位置标签。基础模型已有预训练必须另行披露。COST1 为预定主模型，CE 为预定次模型；空间 ownership 和完整过目不忘系统属于未来工作。

| 面板 | RAW | 对应原配方 | COST1 / CE | 范围 |
|---|---:|---:|---:|---|
| 旧 EVAL32 | 25/32 | ORIGINAL7 28/32 | 不混用后续头 | 已打开内部面板 |
| 旧 EVAL128 | 88/128 | ORIGINAL7 99/128 | 不混用后续头 | 已打开内部面板 |
| H593 | 426/593 | 同折 COST4 440/593 | 481 / 486 | 分组五折 OOF 开发 |
| GroZi480 | 321/480 | 固定 COST4 326/480 | 355 / 367 | 正式商品外部确认 |
| ISIC537 | 466/537 | 固定 COST4 469/537 | 500 / 513 | 已打开队列跨域探索 |
| processed128 | 121/128 | 固定 COST4 121/128 | 122 / 123 | 合成处理回归，净增区间含 0 |

RPC 只作参考信息不足诊断，不列为正式外部确认。不同面板、头和训练协议不能合并计分。历史失败及边界保留在原报告中。

## 归档范围与真实性

原 workspace 相对目录结构保留在 `ICLR/new ROUTEA/RC/`，相关旧源码作为主线依赖或历史证据保留。原始源码、实验合同和结果按字节复制，未为了上传而重新训练或修改预测。

`backup/copied_files.json` 记录每个源文件的 SHA256；`backup/excluded_files.json` 记录 Git 文件树排除项，包括改存 Release 的大模型文件与未上传的缓存、日志；`MODEL_DEPENDENCIES.json` 记录冻结模型来源指纹；`model_assets_manifest.json` 给出大模型原文件与每个分片的 SHA256。上传完成证据见 `backup/publication_receipt.json`。

这是研究归档，不声称 `git clone` 后即可无条件重跑。旧绝对路径、源码封存 SHA、Slurm 要求和历史 deadline 仍按原样保留；恢复运行还需要原图、重建被排除的缓存并适配执行环境，详见复现说明。
