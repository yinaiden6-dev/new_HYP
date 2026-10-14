# 独立补充审计复现

从项目根目录运行。以下命令只消费冻结缓存、历史输入索引和已存在编辑图，不加载编码器、不启动 GPU、不修改原结果。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.completion_audit --recover-position
```

它重放全部 signed-layout 像素/输入/权重/选参/null/编辑控制，恢复旧位置 Probe 的 9 组缺失 null 权重与全样本预测，并生成逐要求库存。

统一视觉/报告完成后可仅刷新库存：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /home/jwei/miniconda3/envs/rebut/bin/python -B -m scripts.colnomic_mechanism.completion_audit --inventory-only
```

`signed_layout_independent_validation.json` 和 `position_null_recovery/independent_validation.json` 的 passed 只适用其明确数值范围。`artifact_inventory.json` 与 `requirement_matrix` 不宣称全 goal 验收完成。自然同文字身份布局、真实阅读行和旧图库来源缺失的限制继续保留。
