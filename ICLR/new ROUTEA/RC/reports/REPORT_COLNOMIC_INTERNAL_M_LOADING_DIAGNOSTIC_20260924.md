# ColNomic 内部 M：首图编码一致性失败已定位

GPU 诊断 `5161422` 在同一个 TRAIN query 上确认：`5161411` 的 `SOURCE_TOKEN_PARITY` 失败来自 processor 被强制设为 `use_fast=False`。保持 strict loader、BF16 权重及 SDPA 不变，仅恢复原始 `processor.from_pretrained(...)` 的默认预处理，原缓存的 FP16 图像 tokens 即逐元素完全一致。原验收阈值无需调整。

诊断对象为 `H593-979d4429602e6c4a69aa6724`，原 `DECODED_RAW_BEFORE_EXIF` 图像、原 ColNomic C128、原冻结 COST1 fold0 输出头；使用已有 reference tokens，没有训练、RoMa 重跑或新增 reference 编码。

| Loader / LoRA dtype | Processor | 与旧图像 token 的平均余弦 | L₀ 最大绝对差 | 原 FP16 tokens 完全一致 | 原 gate |
|---|---|---:|---:|---|---|
| strict / BF16 | 显式 `use_fast=False` | 0.9474738240 | 0.0035806220 | 否 | FAIL |
| strict / BF16 | 原默认 | **1.0000000000** | **1.1271e-11** | **是** | **PASS** |
| strict / 源 FP32 | 显式 `use_fast=False` | 0.9467400908 | 0.0051405080 | 否 | FAIL |
| strict / 源 FP32 | 原默认 | 0.9995729327 | 0.0004256957 | 否 | PASS |

四臂最终 action 都保持原候选位置 87。沿用原 gate：平均余弦 ≥ 0.999、L₀ 最大差 ≤ 0.005、action 完全相同。FP32 诊断重新加载原始 FP32 LoRA 数值，未将已舍入 BF16 数值简单升精度。

## 根因与最小修复

当前 `transformers 5.6.2` 下，原默认是 `Qwen2VLImageProcessor` / `torchvision`；显式 `use_fast=False` 是 `Qwen2VLImageProcessorPil` / `pil`。两个 processor 的 `input_ids`、`attention_mask`、`mm_token_type_ids` 和 `image_grid_thw` 逐元素完全相同，但 `pixel_values` 最大差为 0.0150079429、平均绝对差为 4.6577e-5。它们不是同一像素预处理路径。

修复应在新的版本/authority 中只恢复原 processor 默认路径，同时记录实际 backend；保留原 strict 权重加载、BF16、SDPA 和全部 parity gates。源 FP32 LoRA 不需要成为修复内容，其输出反而偏离原缓存。冻结 v1 及其失败记录应保留。

这项结果只证明首图的工程根因。修复后的全部 16 TRAIN＋8 probe 编码仍必须逐项通过原 parity、零初始化和真实反向传播验收，才能推进四臂训练；不能把本诊断写成内部 M 有效性的实验结果。

## 原 native loader 的报告接口错误

诊断额外调用原 `ColQwen2_5.from_pretrained(adapter_dir, torch_dtype=bfloat16)` 时添加了 `output_loading_info=True`。该报告选项触发 `AttributeError: 'NoneType' object has no attribute 'to_dict'`：当前 Transformers 的 `load_adapter` 无返回值，`from_pretrained` 将其赋给 `loading_info` 后尝试 `.to_dict()`。这是报告接口错误，**不能解释为原 native 权重加载失败**，也未产出可用于比较的 native forward。

原 base checkpoint index 包含 `custom_text_proj.weight` 和 `custom_text_proj.bias`，原 TRAIN128 native 日志仅有 `lm_head.weight` 为 unused。CPU 上实际 ColQwen 小模型加 native PEFT injection 的 pre/post hook 零初始化一致性及冻结骨干反向传播检查通过，但无需因此更换本轮 loader。

## 证据与复核

- [机器诊断报告](../results/rc_internal_m_loading_diagnostic_v1/5161422/report.json)：四臂结果、processor 输入差异、完整 LoRA/projector fingerprints、接口错误堆栈；耗时约 47 秒。
- [诊断程序](../programs/run_rc_internal_m_loading_diagnostic_v1.py) 与 [独立启动脚本](../slurm/rc_internal_m_loading_diagnostic_v1.sbatch)。诊断只写入独立结果目录，未推进原 pilot。
- [原失败 parity](../results/rc_prellm_m_adapter_v1/encoder_cache/H593-979d4429602e6c4a69aa6724/parity.json)。诊断的 BF16＋PIL 臂精确复现其平均余弦及 L₀ 误差。
- 独立字段/绑定复核已通过：四臂文件存在，原 gate 未变，原失败两项数值精确复现，BF16＋原默认的 `tokens_half_exact=true`；authority、诊断程序和源模型验收记录的 SHA256 与报告一致。此项复核未独立重跑 GPU。
