# 原图、冻结缓存与重新编码的证据边界

本说明来自 2026-10-13 恢复任务中的只读审核。原始数据、embedding bank 与历史结果未改写。这里将“从冻结特征重放分数”和“重新运行编码器复现旧特征”分别核验，不能合并为一句“原缓存没通过”。

## 已确认

- 五个重点照片按 `dataset` 原图的 `photo_id`、字节 SHA256 与 `cache/image_embeddings/colnomic7b/photos.pt` 行号对应；证据为 `results/token_mechanism_experiments/20261013_v2/original_image_embedding_audit.json`。`wrong/` 中的图是错误浏览图，只用于人工查验；不能编码浏览图，也不能用重点照片筛选训练或测试身份。
- F03 冻结矩阵回放的历史验收记录覆盖 615 个候选矩阵，矩阵平均 MaxSim 与缓存分数最大差为 `1.7881393432617188e-7`，完整图库候选账本覆盖 1117 查询。此记录位于 `results/token_mechanism_analysis/F03/reproduction/validation.json`，证明范围为已保存矩阵与冻结账本，不能据此宣称重新编码全部 5405 图库成功。
- 近期生成的 `photo_41de3eb383934ec3` 和 `photo_21bd7df9ed6795e7` 在默认 torchvision 后端、FP32 LoRA 与当前运行时中重新编码，和其原始缓存逐位一致；自动加载与显式加载结果相同。
- 旧 `encode_worker.py` 将处理器后端默认改为 PIL。对同一原图，v1 与 v2 受控 PIL 路径复现出完全相同漂移：`41de…` 最大分量差 `0.3798828125`、平均 Token 余弦 `0.9612646103`；`21bd…` 最大差 `0.3828125`、平均余弦 `0.9780603051`。因此近期照片这部分漂移的原因已定位为处理器后端变更。
- 扩大布局实验 `stage4_extended_photos_v3/original_cache_replay.csv` 的 38 个原图（20 训练、8 验证、10 测试身份）全部 `exact=True`、最大误差 0。恢复任务还从保存 tensor 与输入字节独立核验了 228 个编码输出，而非只相信 CSV 标志；见相邻恢复检查点 `checkpoint_2108/checkpoint_validation.json`。
- 自动与显式 FP32 LoRA 加载均有 394/394 参数键匹配、394/394 参数值逐位一致，无缺失与额外键。这排除了被测运行时下“适配器没有加载”这一解释。将 FP32 LoRA 改成 BF16 本身会产生小误差；`bf16_adapter_loading.json` 中的 `exact=True` 是先将期望值转成 BF16 再比较，不能解释为 FP32/BF16 算术完全相同。
- v1 的 32 个保存输出在同运行时复核中全部逐位一致（`within_runtime_repeatability.json`）。resident 内存驻留优化之后，近期 `41de…` 仍能逐位复现缓存。没有证据把漂移归因于 GPU 型号或 CPU 权重卸载。

## 历史图库尚未复现

所有 5405 图库记录均标记为 `reused`。其 provenance 指向：

`backup/previous_release/cache/backbones/colnomic_7b/image.canonical.pt`

上游为历史 `eval/workspace/colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt`，记录源文件 SHA256 为 `74f3bd6278fb7f36d1eabef569abc329cf9f9b315778c8353677694750d8a98e`。

当前目录不存在上述 `backup/previous_release`。`cache/image_embeddings/metadata.json` 记录的 Torch 2.9.1、Transformers 5.6.2 等运行时来自本轮生成照片的流程；5405 图库全部复用，不能将这份运行时记录当作图库历史编码环境。

默认当前重新编码 `reference_000229` 与 `reference_000233` 的最大分量误差分别是 `0.400390625` 与 `0.4072265625`，平均 Token 余弦为 `0.9681343436` 与 `0.9622225761`。自动与显式 FP32 加载完全一致，因此适配器加载方式没有解释这两条图库漂移。已保存的 PIL、BF16 LoRA、eager attention、GPU resize、双线性 resize、通道顺序及 SigLIP 归一化候选流程均未逐位复现历史图库；不是再尝试过一个参数就能确认根因。

同形状缓存行对比中，各自原 ID 的 Token 余弦远高于其他行，支持其对应关系，但这项局部检查不足以全局证明图库从未错行。

尚缺历史处理器张量、视觉骨干输出、投影前后特征与完整历史编码代码。因此不能确定历史图库差异首次出现于哪层，也不能确认唯一根因。历史软件环境、预处理与数值执行路径均保留为未区分解释。

## 文件范围与解释限制

- `extra_recipe_metrics.csv` 曾被 `--remaining` 执行覆盖；完整已保存输出由 `all_recipe_metrics_recovered.csv` 恢复，应以具体 tensor 及恢复后的指标交叉核验。
- 早期保存的通道实验文件名为 `channel_BRG_verification`，当前源码使用 `channel_BGR_verification`；文件命名与当前源码已有差异，不能凭当前源码宣称完整历史执行配置逐项可恢复。
- 新干预的因果响应应使用同运行时重新编码的 query 与比较候选。新 query 混用旧图库时只能称为固定图库条件排名，不能称为新运行时完整图库准确率。
- 冻结 MaxSim、匹配规则、贡献分析与 Probe 可以继续复用原始历史 embedding；图库重编码不复现不使这些冻结评分分析失效。新旧运行时必须分别标记。

可信结论是：冻结缓存评分能够在已核验范围内重放；近期照片的 PIL 后端漂移已定位并修正；38 个布局原图可逐位复现；历史图库重新编码仍不能逐位复现，根因未确定。
