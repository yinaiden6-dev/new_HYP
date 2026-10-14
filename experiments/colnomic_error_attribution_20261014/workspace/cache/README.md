# DrugPack 计算缓存

基础缓存依据 `dataset/lists/photos.csv` 和 `references.csv` 的完整清单生成。标注或图片目录变化不要求重新编码；图片字节、模型或输入文字变化会触发对应记录的补算。

2026-10-10 已补齐至 **1,117 张照片、5,405 条参考图**，原六个图像模型与四个文字模型共 39,132 个图像向量、26,088 条文字向量记录及 27 份分数缓存。旧 1,000 张的向量、OCR 文本与分数行逐元素一致，参考向量文件字节保持一致。[本次完整核验](completions/20261010T142424Z/verification.json)。method 缓存保留历史范围；35 组主结果和 foreground 向量/分数已更新至 1,117 张。前景框、掩码及裁剪输入已单独补齐到 1,117 张。全部 1,117 张框已由作者确认，最终矩形裁剪与 35 组结果已完成核验；新 ColQwen2.5 v0.2 的图像库及 OCR 查询向量已生成。[主体标注说明](../dataset/annotations/README.md)。

新增 ColQwen2.5 v0.2 已生成 1,117 条图像查询、5,405 条参考图像向量和 1,117 条 OCR 查询向量。七个模型合计 45,654 条图像向量记录，文字向量合计 27,205 条；本次六种检索形式使用图像参考库。训练权重和 Query: 查询模板已核验，35 组结果见[最终汇总](../results/README.md)。

| 目录 | 内容 |
|---|---|
| `ocr/` | PP-OCRv6 文本、文本框和识别配置 |
| `image_embeddings/` | ColPali、ColNomic 3B/7B、Qwen3-VL-Embedding-2B、DINOv3、SigLIP2、ColQwen2.5 v0.2 图像向量 |
| `ocr_embeddings/` | 原四种文字模型的照片与参考 OCR 向量，以及 ColQwen2.5 v0.2 的照片 OCR 查询向量 |
| `scores/` | 完整参考图库上的基础分数、CRISP 和固定权重融合分数 |
| `completions/` | 增量更新前的快照、任务日志、复用统计与核验记录 |
| `method/`、`foreground/` | 绑定各自实验协议的专用缓存；适用范围由各自 manifest/config 决定 |

激活主模型环境，并设置 `DRUGPACK_OCR_PYTHON` 后，在仓库根目录运行：

```bash
python -B -m drugpack cache
python -B -m drugpack cache-scores
```

这两个命令更新缓存。实验评估使用 `drugpack baseline`。选择模型或单独生成图像/文字向量可使用 `drugpack cache --model colnomic3b --view image`。

保留更新前的缓存快照后，分数补算可以复用旧行：

```bash
python -B -m drugpack cache-scores --before cache/completions/<run>/before
python -B scripts/verify_cache_completion.py --snapshot cache/completions/<run>
```

只有旧向量逐元素一致、图库一致、分数签名与评分代码一致的记录才复用。其余记录重新评分。更新前快照保留旧缓存字节，供历史实验核对。
