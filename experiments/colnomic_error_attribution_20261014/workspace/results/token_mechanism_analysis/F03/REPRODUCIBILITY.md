# F03 原始特征与复现包

本轮仅补齐归档。已有对应/贡献诊断仍然有效；没有执行几何约束重新排序、Probe 或布局干预。

- 原分析目标 145 例、258 组对照。
- 附带 **全部 1117 张照片、5405 行图库的原始 FP16 token embedding**，保留数值、顺序、零向量、非图像模板 token 和历史 provenance。
- 本包 216 张目标/成功对照、248 个参考候选的原图；字节和 EXIF 均保留。
- **615 个完整 FP32 token×token 相似度矩阵**，两端全部缓存行都在矩阵中，不是最大值、Top-k、热力图或量化近似。覆盖原始所有对照，以及每张选定照片的正确候选、图像/融合实际 Top-1 和图像/融合最高分错误候选。重复使用的候选对只存一次，由索引链接。
- image/BM25/fusion 的完整 [1117,5405] 分数缓存、全部候选清单、身份表、划分表、OCR 多边形、确认主体框、小型处理器/分词器配置和编码源代码。

## 文件入口

| 内容 | 路径 |
| --- | --- |
| 原始 embedding 银行 | `reproduction/cache/image_embeddings/colnomic7b/photos.pt`、`references.pt` |
| 每个 ID 的银行行、布局、有效 token 索引、EXIF、向量 SHA | `reproduction/token_metadata.json` |
| 完整候选对矩阵及原始轴索引 | `reproduction/similarities/<photo_id>/<reference_id>.npy` |
| 候选对、原始对照目录、分数、矩阵 SHA | `reproduction/pair_index.json` / `.csv` |
| 本包比较对象 / 全部 1117 张比较对象 | `reproduction/candidate_ledger_selected.json` / `candidate_ledger_all1117.json`，各附 CSV |
| 同面核实表 | `reproduction/visible_face_review.csv`（未核实，不作为已完成同面对照） |
| 矩阵数值及 CSV 复现验证 | `reproduction/validation.json` |
| 每个补充文件 SHA-256 | `reproduction/FILES.sha256` |
| 编码入口 / 原始编码实现 | `source_code/scripts/encode_colnomic_images.py` / `source_code/evaluation/encoders.py` |
| 模型版本、数据类型和预处理配方 | `reproduction/model_recipe.json` |
| 后续实验协议 | `NEXT_EXPERIMENTS.md` |

## 比较对象与评分

`image_top1` / `fusion_top1` 是正式排序实际第一名，可能是正确候选。`image_strongest_wrong` / `fusion_strongest_wrong` 是排除所有正样本图像 SHA 后的最高分候选；即使方法检索正确也保存此候选。先稳定排序，再按图像 SHA 去重；同分遵循正式参考行顺序。身份歧义仅按具体正负图像 SHA 配对转移，不能只按照片转移。

矩阵 `[i,j] = photos[qi][i].float() @ references[ri][j].float()`；没有二次归一化。用于 MaxSim 的轴通过 `valid_token_indices` 选择（范数 >1e-8）。每个有效 query token 取有效 reference token 的最大值，再取 query 均值。非零模板 token 参与原始分数。原始网格 `row_raw/col_raw` 位于未应用 EXIF 的编码帧；OCR、主体框和显示坐标属于 EXIF 转正帧，可用快照中的 `display_xy` 转换。

矩阵覆盖上述选定候选对，**没有物化全部 1117×5405 个 token 矩阵**。全图库 embedding 与分数已齐全，可按需计算任意候选对。原图只覆盖本包候选与照片；以后执行全图库几何估计还需要其余原图。所有身份标签保持数据集定义，包装面和同文字不同布局类别还需人工核实；原始 train/val/test 表不能自动代替按包装身份划分的 Probe 协议。

## 无需 GPU 的复现

解压后进入 `F03` 目录。使用 `source_code/requirements-pinned.txt` 中的冻结依赖；CUDA Torch 构建需与机器适配。矩阵和特征读取、核查不需要模型权重。

```bash
python source_code/scripts/replay_colnomic_bundle.py --bundle . --verify-files --verify-matrices
PYTHONPATH=source_code python -m pytest -q source_code/tests
```

测试导入需要 `PYTHONPATH=source_code`，或先进入 `source_code` 执行 `python -m pytest -q tests`。

单个矩阵可以直接 `np.load(path, mmap_mode='r', allow_pickle=False)`。银行可用 `torch.load(path, map_location='cpu', weights_only=True, mmap=True)`；ID 的正式索引已保存在 `pair_index` 中。

重新运行原有分析（会估计 SIFT 几何并绘图，需要已附原图，不加载编码器）：

```bash
cd reproduction
python ../source_code/scripts/analyze_colnomic_tokens.py --root . --configuration F03 --all-errors --output ../rerun_F03
```

## 对新图/编辑图重新编码

7B base 和 adapter **模型权重不在压缩包内**；完整冻结特征已在包内。权重的仓库与 revision 位于 `model_recipe.json`，需使用匹配的完整本地 checkpoint。原有分片编码配方要求两张可见 CUDA GPU，入口 `--device cuda:0`。处理器配置已经附带，`--prepare-only` 不需要模型权重或 GPU。

```bash
python source_code/scripts/encode_colnomic_images.py --root reproduction --image dataset/path/to/image.png --prepare-only --output prepared.json
python source_code/scripts/encode_colnomic_images.py --root reproduction --image /absolute/path/edited.png --adapter /absolute/path/complete_adapter --base /absolute/path/complete_base --device cuda:0 --output edited_embeddings.pt
```

也可用 `--manifest images.csv` 批量编码，CSV 包含 `id,path`，相对路径以 `--root` 为基准。脚本写入新输出；不会更新附带的冻结银行。预处理已验证，未在本轮重新加载 7B 权重进行模型推理；不同硬件重新编码不承诺逐位一致，应另报告和原始特征的数值偏差。
