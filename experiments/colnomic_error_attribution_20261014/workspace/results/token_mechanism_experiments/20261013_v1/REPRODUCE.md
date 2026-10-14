# 复现四阶段实验

这个包包含总报告、全图库新分数、候选/成功对照表、逐 Token 正负差额、完整重点相似度矩阵、Probe 头、改图原图/编辑图、新 Token embedding、实际执行的编码代码和校验清单。

完整的原始冻结图库与 7B 权重是外部依赖。原始图库、OCR、标注和划分可从此前 F03/B03 补充归档或 `/mnt/lsdf_zyin` 工作区取得；权重未包含。此包不是不依赖这些数据的独立编码器发行版。

## 读取已运行结果

先读 `REPORT.md` 和 `validation.json`。`stage2_geometry/full_gallery_scores.npz` 采用 FP32；其轴在 `score_axes.json`，数组含原分数、6 个 λ 的几何奖励/惩罚分数、几何支持分和坐标网格分。所有 5405 个参考行均保留，排名时按 SHA 稳定去重。

`stage4_interventions/validated_embeddings/*.pt` 使用 `torch.load(..., weights_only=True, map_location='cpu')`。其中 `embedding` 是完整 FP16 Token 矩阵（不是文本 token ID），并保存图像 SHA、模型配置、运行时和执行代码 SHA。相似度 NPZ 保留 FP32 矩阵与两个原始 Token 索引轴；`*_contributions.csv` 中 `delta = positive_maximum_similarity - maximum_similarity`，对有效 Query Token 求均值即正负分差。

`portable_encoding_manifest.json` 使用相对路径，无重复 ID；原执行清单有一个相同原图的重复条目，因此 32 条主清单对应 31 个不同主输出，另有 1 个处理器回放输出。同运行时两次编码的 32 个输出逐位一致，见 `within_runtime_repeatability.json`。

主几何方法含有对齐奖励，纯惩罚是另一项消融。OCR 对齐可用性控制是事后诊断，不能作为新的盲测结果。原图与历史缓存的编码一致性没有通过，不能把第四阶段的编辑响应当作原缓存失败的因果证明。

## 在原工作区重新执行

这些命令不会修改原始缓存。另设 `COLNOMIC_EXPERIMENT_OUTPUT` 并复制候选清单、人工审核和划分核验到新目录，可以保存新运行的结果。完整原始数据必须与 `input_manifest.json` 的 SHA 一致。代码快照在 `source_code`；以下命令使用工作区对应脚本。

```bash
export COLNOMIC_WORKSPACE=/mnt/lsdf_zyin
export COLNOMIC_EXPERIMENT_OUTPUT=/mnt/lsdf_zyin/results/token_mechanism_experiments/replay_v1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/colnomic-token-analysis-mpl
mkdir -p "$COLNOMIC_EXPERIMENT_OUTPUT"
cp /mnt/lsdf_zyin/results/token_mechanism_experiments/20261013_v1/stage1_candidate_ledger.json "$COLNOMIC_EXPERIMENT_OUTPUT/"
cp /mnt/lsdf_zyin/results/token_mechanism_experiments/20261013_v1/stage1_manual_audit.json "$COLNOMIC_EXPERIMENT_OUTPUT/"
cp /mnt/lsdf_zyin/results/token_mechanism_experiments/20261013_v1/split_verification.json "$COLNOMIC_EXPERIMENT_OUTPUT/"

/home/jwei/miniconda3/envs/rebut/bin/python scripts/colnomic_mechanism/geometry.py
/home/jwei/miniconda3/envs/rebut/bin/python scripts/colnomic_mechanism/probes.py
/home/jwei/miniconda3/envs/rebut/bin/python scripts/colnomic_mechanism/interventions.py

CUDA_VISIBLE_DEVICES=3 /home/jwei/miniconda3/envs/rebut/bin/python scripts/colnomic_mechanism/encode_worker.py \
  --manifest "$COLNOMIC_EXPERIMENT_OUTPUT/stage4_interventions/encoding_manifest.json" \
  --output "$COLNOMIC_EXPERIMENT_OUTPUT/stage4_interventions/validated_embeddings" \
  --processor-backend pil

/home/jwei/miniconda3/envs/rebut/bin/python scripts/colnomic_mechanism/analyze_controls.py \
  --embedding-dir "$COLNOMIC_EXPERIMENT_OUTPUT/stage4_interventions/validated_embeddings" \
  --output "$COLNOMIC_EXPERIMENT_OUTPUT/stage4_interventions/analysis_validated_runtime"
/home/jwei/miniconda3/envs/rebut/bin/python scripts/colnomic_mechanism/diagnostics.py
```

GPU 编号应按空闲资源调整；本轮使用 RTX 2080 Ti 和 CPU 权重卸载。精确软件版本在 `runtime.json` 和 `validated_embeddings/complete.json`。不要终止其他 GPU 任务来复现。

原图回放严格阈值为最大 embedding 分量误差 `<1e-5`。本轮失败的默认 torchvision、PIL、历史环境回放记录均保留；通过同运行时重复性不意味着通过历史缓存一致性。

最终整理脚本 `finalize.py` 为本次证据交付核验而写，会检查本轮重复编码和执行代码哈希；对新的运行应先核实相应执行记录，不应强行套用旧记录。复现科学结果的入口是上面的各阶段脚本。

## 校验

```bash
/home/jwei/miniconda3/envs/rebut/bin/python -m pytest \
  tests/test_colnomic_mechanism_experiments.py \
  tests/test_colnomic_token_analysis.py \
  tests/test_colnomic_archive_supplement.py -q
```

19 项测试覆盖 OCR 顶点顺序、已知单应变换、MaxSim 掩码/均值、EXIF 坐标、像素守恒、Probe 标签打乱与训练标准化，以及旧归档评分核验。最终包另核验全部成员 SHA/ZIP CRC、14 个重点相似度矩阵和 12 张 Token 差额表。归档排除错误顶点顺序的旧输出、重复的初版编码与初版分析；它们仍保留在原工作区。
