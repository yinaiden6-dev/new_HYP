# new HYP — 主路线研究归档

Reference-conditioned Joint Evidence Calibration for Fine-grained Retrieval.

2026-09-19 从原 workspace 整理。仓库保存源码、配置、理论、结果、展示材料及小头参数；大模型权重放在同一私有仓库的 **Release 附件**。原始 query/reference 图片另行备份，本仓库保留清单。历史实验特征缓存及 err/out/log 不上传。

[下载模型权重 Release](https://github.com/yinaiden6-dev/new_HYP/releases/tag/mainline-backup-20260919)：9 个原始权重文件，34.6 GB；37 个分片及清单均已通过 GitHub 服务端 SHA256 核验。

## 从这里阅读

<!-- TOKEN_PUBLICATION:rc_token_competition_f128_v2:BEGIN -->
- **[F128 native token competition V2 / 原生 token 竞争完整结果](backup/token_competition_f128_v2_20260927/README.md)**：15 项独立验收，fixed0 主结果与继承阈值次结果；同面板 B_CAL、MULTI/ANCHOR、救回/误伤及开发面板限制。
<!-- TOKEN_PUBLICATION:rc_token_competition_f128_v2:END -->

- **[2026-09-27 M归因128图：信息性质、恢复与外部／内部传播](backup/m_causal128_attribution_20260927/README.md)**：9类验收通过：128图冻结传播、20项入口补偿、新9组性质复验与同模型传播、旧7组到新9组有界性质补偿；并列正向证据、内部反向结果及仍未分开的原因。

- **[2026-09-27 QR/QRR、M归因与统一采集快照](backup/rebut_and_unified_sampling_20260927/README.md)**：71张60项关系读出结果、481/492逐图比较、外部/内部相位干预和逐patch证据；附冻结基线v2准备，以及128张恢复与剩余465张四个长任务的提交记录。明确区分已完成结果与运行中的采集。

- **[2026-09-27 H593全部112张失败案例图（英文）](backup/h593_failures_20260927/README.md)**：沿用原90图的COST1口径与放大reference版式；92未纠正、17错换、3误伤，包含23张正确reference未进C128的案例。附全部127挑战分数、HOLD、真实热图与独立核算。

- **[2026-09-25 四项贡献补强与最新中英文稿](backup/paper_contributions_20260925/README.md)**：完整保留检索器benchmark、可复用空间支持接口、外部/内部统一框架与机制归因；区分定义、推导和实验发现。学长尚未汇总的模型比较明确待填，科学结果保持不变。

- **[2026-09-25 GroZi内部迁移完整实验包](backup/grozi_internal_complete_20260925/README.md)**：补齐480张×六臂的逐候选分数、60片验收、缓存等价性元数据、冻结代码及身份轴修复记录。RAW321→POST_REAL339，18救0损；与下方写作材料分开归档。

- **[2026-09-25 上一版论文主稿、空间支持统一说明与新颖性核对](backup/paper_support_novelty_20260925/README.md)**：分别更新 reports、reports/data 和 plan 原目录；补齐固定内部模型 GroZi480 的321→339（18救0损）结果及验收。此项记录当时的写作与证据快照；最新文档见上方四项贡献更新。旧归档清单及校验脚本须在各自归档提交运行，不能用于验证后来修订的同名文档。

- **[2026-09-25 后 LLM 全量实验与机制归因完整归档](backup/postllm_and_attribution_20260925/README.md)**：H593 内部 POST_REAL 478/593；共同响应 477、patch 剩余 430、固定内容命中 469。含完整逐候选记录、独立核算、上游干预及无 RoMa 对照的负结果与边界。下方 V3/V4 快照保留为历史记录。

- **[90张英文纠错图（放大 reference）及内部M V4最新快照](backup/rescue90_english_and_internal_m4_20260924/README.md)**：药盒／GroZi／ISIC各30张，附完整候选分数、原生热图与验证记录。

- **[2026-09-24 内部M失败定位与V4阶段快照、ZIP下载](backup/internal_m_diagnostics_v4_snapshot_20260924/README.md)**：补齐V3读出/条件尺度诊断，V4仍在训练，阶段数据不作为最终结论。

- **[2026-09-24 论文收口稿与内部M V3最终结果](backup/internal_m_v3_final_closure_20260924/README.md)**：内部真实M／恒定M均8/16，外部加性／乘积均12/16；完整终点已封存，未获得内部纠错收益。

- **[2026-09-24 Qwen乘积项完整结果与内部M V3阶段快照](backup/internal_m_v3_and_qwen_product6_20260924/README.md)**：Qwen CE 536→537/593（2救1损，分组区间跨零），COST1维持504/593；内部M真实输入线仅归档至88/128步，尚非最终结果。

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
