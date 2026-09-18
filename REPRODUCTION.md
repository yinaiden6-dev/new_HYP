# 执行入口与复现边界

该仓库保存研究使用的原脚本与结果；以下是定位入口，不是承诺可以在任意机器直接执行的快速启动脚本。

| 环节 | RC/programs 下的主要入口 |
|---|---|
| H593 图片/分组清单 | `freeze_rc_new_hyp593_oof5_v1.py` |
| RAW/RoMa 原数值循环 | `materialize_rc_new_hyp593_inputs_v1.py`；实际续跑序列化修复见 `materialize_rc_new_hyp593_inputs_v2_serialization.py` |
| H593 证据特征 | `cache_rc_new_hyp593_features_v1.py` |
| COST4/原小头拟合 | `run_rc_new_hyp593_oof5_v1.py` |
| COST1/CE 与绑定检验 | `run_rc_six_cause_loss_binding_v1.py` |
| 外部使用的五个固定头 | `run_rc_new_hyp_external_head_freeze_v1.py` |
| GroZi/ISIC 推理 | `run_rc_new_hyp_grozi120_inference_v1.py`、`run_rc_new_hyp_isic_inference_v1.py` |
| processed128 | `run_rc_new_hyp_processed128_v1.py` 与 `run_rc_new_hyp_processed128_v2_serialization.py` |
| CRISP/手填对照 | `run_rc_crisp_manual_baseline_v1.py` |
| 三头计时 | `benchmark_rc_head_training_time_v2_cost4.py` |
| 文档归档 | `update_rc_results_theory_processed128_v1.py` |

所有相对入口位于 `ICLR/new ROUTEA/RC`。`src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py` 保存六统计与 action 定义；原全文内容匹配路径保持为 full-reference weighted MaxSim，不是单点硬匹配。

## 环境

运行时应以仓库内原资格检查记录为准，尤其是 `results/rc_original7_eval128_token_raw_v2_compat_preflight/1ce5aaffabf17d5816a31181517535f0305820bcfac553fea586c82999393ce7.json` 及 `registry/rc_original7_eval128_roma_source_profile_v1_20260910.json`。

原 RAW 合格环境记录包括 Python 3.13.11、PyTorch 2.9.1/CUDA 12.8、Transformers 5.6.2、ColPali Engine 0.3.15、PEFT 0.18.1、NumPy 2.2.6。RoMa 使用独立环境与源码指纹。单独查询当前交互式 Python 可能被 user-site 包覆盖，不能据此替换历史运行环境记录。

RoMaV2 和 DINOv3 的封存源码及上游许可证一并保留；DINOv3 的固定源版本为 `adc254450203739c8149213a7a69d8d905b4fcfa`。主 ColNomic 模型来源为 `nomic-ai/colnomic-embed-multimodal-7b`。这次导出没有训练新编码器或重新拟合头。

`third_party/runtime_source_snapshots/` 还保留合同中点名的 ColPali/Transformers/PEFT 核心源码快照，供核对精确实现；它不是整个虚拟环境，也不是替代正常安装这些依赖的软件包。`backup/copied_files.json` 同时记录快照在原 workspace 的 `source_path` 与仓库保存路径。

## 恢复运行前

1. 恢复原图和模型，核对 gallery 身份顺序、query 及分组清单。
2. 在另一个运行副本中映射旧绝对路径，安装与记录一致的依赖；原归档保持不动。
3. 按脚本重建未上传的 token、gallery 与 RoMa 缓存，完成原代码要求的验证。历史冻结脚本还带有 deadline、SLURM 和 authority 检查，不能删除这些条件后声称仍是原封存试验。
4. 区分 H593 的五折头、外部的 full-H593 固定头以及历史 ORIGINAL7/FROZEN_C；始终保留自然 C128 缺席样本的分母。

文档与校验记录足以查看已完成结果，但本次上传并未执行跨机器端到端复现，不能把备份完成等同于重新验证了全部科学实验。
