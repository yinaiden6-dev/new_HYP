# DrugPack

药品包装图像检索：冻结 baseline 与参考条件重排老方法。当前数据为 1,117 张照片、130 个包装设计、5,405 条 reference（0.8.0-rc1）。2026-10-10 新增 17 个包装、117 张照片，来源组划分为 783／173／161 张 train/val/test。

| 位置 | 内容 |
|---|---|
| `drugpack/`、`evaluation/` | 统一命令入口、老方法与 baseline 代码 |
| `configs/` | 模型配置和冻结实验协议 |
| `dataset/` | 核心数据、标注、数据工具；审核记录在 `review/` 子目录 |
| `other_datasets/` | GroZi、MILK10k、FORB 及其有效缓存和结果 |
| `weights/`、`cache/`、`results/` | 模型、计算缓存、实验结果；老方法各在 `method/` 子目录 |
| `scripts/`、`tests/` | 必要安装 / 验证工具和测试 |

Linux / Python 3.12，激活环境后在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e '.[method,test]'
python -B -m drugpack doctor --profile method
python -B -m drugpack baseline --help
python -B -m drugpack cache --help
python -B -m drugpack cache-scores --help
python -B -m drugpack method evaluate --help
python -B -m drugpack review
python -B -m drugpack verify results --read-only
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m pytest -q
```

已备齐依赖的环境可使用 `--no-deps` 省去依赖安装。安装后可用 `drugpack --root /path/to/DrugPack ...`；各命令用 `--help` 查看参数。

主解释器可用 `DRUGPACK_PYTHON` 指定。OCR 使用 `DRUGPACK_OCR_PYTHON`，BioMedCLIP 使用 `MILK_BIOMED_PYTHON`；默认分别在 `.venvs/ocr/` 和 `.venvs/biomed/`。安装入口为 `scripts/setup_ppocrv6.sh`（Python 3.11）和 `scripts/setup_milk10k_baselines.sh`。

[Baseline 结果](results/README.md) · [老方法说明](drugpack/method/README.md) · [老方法结果](results/method/REPORT.md)。

当前数据资产可用 `python -B dataset/tools/check.py verify` 完整核验，v4 划分可从本地补充前快照复现。OCR、六种图像向量、四种文字向量和 27 份基础分数缓存已补齐到 1,117 张照片，并通过全量核验。新增 117 张的场景标签和前景遮挡均已完成人工校验。[缓存说明与增量命令](cache/README.md)；[本次核验记录](cache/completions/20261010T142424Z/verification.json)。主体框建议、自动分割掩码及裁剪输入已补齐至 1,117 张。全部 1,117 张主体框已由作者确认（695 张逐张保存、422 张批量确认），已完成最终裁剪和 35 组全量统计；[主体标注说明](dataset/annotations/README.md)。35 组主结果及 `foreground/` 已更新至 1,117 张；`method/` 缓存与训练方法结果仍为历史范围。原 29 组已更新至全量，新增 ColQwen2.5 v0.2 的 6 组；[35 组汇总](results/all_results.csv)和[独立核验](results/verification.json)均已完成。旧路径在读取时映射，历史冻结记录保留原样。

源码包按需生成到项目外：`python -B -m drugpack build-release --output /path/to/DrugPack-source.tar.gz`，仅包含源码、配置、测试和文档。

新增 [L01 OCR 坐标实验](results/bm25_layout_fusion/colnomic7b/fixed_v1/REPORT.md)，复用 ColNomic 7B 与 BM25；全量结果包含 L01 对照。运行 `python -B -m drugpack layout-fusion`，使用固定参数对 F03 候选作局部文字布局重排。

新增 [L02 显式相对坐标实验](results/bm25_relative_layout_fusion/colnomic7b/fixed_v1/REPORT.md)，布局评分只使用文字行之间的相对位移、上下左右关系和相对宽高，绝对坐标不参与。全量结果现为 37 组。运行 `python -B -m drugpack relative-layout-fusion`。
