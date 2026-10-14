# Rebut experiments

所有表格来自实际完成并通过检查的运行。模型和超参数只根据药盒验证集选择；跨数据集直接使用同一检查点，无重新训练或外部数据调参。

药盒测试 R@1：验证集选定的主适配器在严格训练／图库可见训练下分别为 80.27%／82.99%；对应外部头为 82.31%／82.31%，纯 OCR＋BM25 为 91.16%。比较训练收益时，需要同时考虑这些简单对照及配对置信区间。

FORB 上，无需训练的固定 weighted_L 规则为 95.62%，两套主适配器分别为 89.63%／90.45%。这些结果区分了几何匹配本身的收益与训练适配器的收益；不能只根据相对原始图像模型的提升来判断训练是否必要。

## Protocol

- 药盒：706 张训练、147 张验证、147 张测试；75/16/16 个来源组；完整图库 5,405 行、5,330 个不同图像。所有照片存在历史开发接触，不能称为从未接触过的盲测。
- Strict：训练只使用 80 个训练包装的正设计图作为候选。Catalog：完整固定设计图库参与训练负样本；只有训练照片标签参与参数拟合，不能称为未见设计图训练。
- 两种训练范围也改变了负样本池的大小和难度；二者差异不能单独归因于是否见过验证／测试设计图。各项方法和消融都在各自相同训练范围内比较。
- 候选来自原始 ColNomic 7B 自然排名，不插入正确答案。Catalog 中正确图不在 Top-128 的训练照片排除于重排损失；验证和测试全部保留。
- 25%/50% 指从 75 个训练来源组选择 19/38 个完整组，实际使用 194/381 张训练照片，约为全部 706 张的 27.5%/54.0%。这是查询标注量消融，80 张训练设计图的负样本库保持不变，不能解释为同时缩小设计图库。
- 主体裁剪补充对照复用已有完整结果：SAM3 自动选取最高置信度包装 mask；历史 human_boxes 实为 200 张人工确认框 + 800 张 SAM3 默认框的混合矩形裁剪。人工框在 train/val/test 分别为 120/22/58 张，其余为默认框。这是混合区域信息诊断，不是完整人工框上界，也不是纯自动方法；两路的区域选择与背景处理同时不同。均使用相同 ColNomic 7B 与完整设计图库。
- 检索指标使用完整图库排名。只改排 Top-K，剩余名次不变；Top-K 外的正样本仍计入 MRR。区间按来源组配对重采样；FORB 的 23 对重复查询图像保留在指标中，并按相同 SHA 归为同一重采样组，宏平均按域。
- GroZi：468 个官方 val-new-cl 包围框裁剪查询、1,063 张设计图。MILK：479 个测试病灶的临床图→皮肤镜图配对检索；是项目改编协议，不是官方诊断任务。MILK 未提供患者 ID，无法确认患者独立性，区间按病灶重采样。FORB：8 个域、8,995 个查询、43,585 张图库图像；固定 Top-32 重排，保留全部查询和多正样本。
- 原始查询 embedding 一致性是缓存检查的一部分。FORB 的单卡编码路径在 91 张查询上与原双卡缓存不一致，已按原路径重算；原基线、候选、标签和训练权重保持不变。全部 FORB 查询的实际向量均与原缓存逐项核验，不通过检查的特征不进入报告。
- FORB 评估曾有一次非有限评分中断和两次进程启动崩溃。失败查询在 CPU 上复算一次、在原 GPU 上复算 10 次，均通过，GPU 重复结果完全一致；随后三个原配置分片全部恢复完成，有限性检查未放宽。根因未在新进程中复现，原日志和跨设备对照见[恢复记录](../cache/intermediate/audits/evaluation_retry_20260925.json)。

## What is trained

ColNomic 7B、原有检索投影及 RoMa 全部冻结。RoMa 双向可信度按图像 token 网格做面积池化，得到查询支持 u、参考支持 v，以及 M = sqrt(mean(u) × mean(v))。RAW 使用原缓存的全部有效 token；L0 使用归一化图像 token。INTERNAL3 对照控制 L0 分支本身的影响。

无需训练的固定对照也在主表报告：M 只用整体匹配置信度；weighted_L 对参考 token 的相似度乘以 v，再对 MaxSim 按查询支持 u 加权平均；S = M × weighted_L。这些规则没有在测试或迁移数据上选权重。

原始 Post 适配器把标准化 log(M) 与每个图像位置的 LayerNorm 隐状态拼接，经过 16 维瓶颈和 0.1 倍残差，再经冻结的检索投影得到该候选对应的查询向量。局部条件版本将整体 M 换成各位置的 u。隐藏层版本训练 118,304 个适配器参数及 3 个评分参数；128 维 embedding 版本训练 4,256 + 3 个参数。所有条件标准化统计量只来自训练对。

适配器的 3 个评分输入是标准化 RAW 分差、相对内容分差和偏置；M 只通过条件化进入表示，不直接进入这个评分头。INTERNAL3 使用同样的评分输入但不训练图像表示，ADDITIVE4 则额外直接加入 M 的相对分差。常量条件适配器和 INTERNAL3 的推理不需要 RoMa。

外部头也训练：INTERNAL3 / RAW_M / ADDITIVE4 / PRODUCT5 / WEIGHTED7 分别拟合 3 / 3 / 4 / 5 / 7 个评分参数。RoMa-free Reader 直接读取冻结的 token 对应关系，包含 COMPRESSED、SET 和 SPATIAL 三种对照，推理无需 RoMa。

线性头每次拟合使用全训练集批次的 2,000 步 AdamW；ADDITIVE4、PRODUCT5 和 ADDITIVE_UV 先额外拟合 INTERNAL3 初始化。适配器和非线性头也用相应线性头初始化，再按单查询更新 8 轮。训练照片保持相同，线性头与非线性模型的优化步数不同。

方向性对照分别使用 U = mean(u)、V = mean(v)：RAW_U / RAW_V 各 3 个参数，ADDITIVE_UV 用 5 个参数分别融合两侧覆盖率与视觉内容。这用于检查实拍局部视图与完整展开设计图的覆盖范围差异；U/V 本身的无训练排名也单独报告。

空间打乱和 180° 旋转控制只对 SET/SPATIAL Reader 执行；COMPRESSED 没有对应的邻域消息传递，不把无操作当成空间消融。

非线性外部头使用 RAW/M/L0 相对分数以及候选和原第一名的标准化 log(M)，在 ADDITIVE4 上增加两层 tanh MLP。宽度 61 和 340 分别对应 4,210 和 118,321 个 MLP 参数，另有 4 个基础评分参数，与两种 token 适配器的参数量接近；它不修改图像 token。它使用相同训练照片、8 轮训练、COST1/CE 对照及验证选择规则。

候选分数相对原始第一名建立 HOLD 决策。COST1 对正确候选及最强错误候选约束，另有 CE 对照。适配器训练先计算完整候选前向，再对非零损失梯度对应部分重算反向，每步检查分数一致性。重算实现已增加原前向批次形状保持，避免低精度批形状变化；一次受保护中断从最后有效检查点恢复，修正及逐步校验记录见 protocol.json。冻结验证选择后才评估新模型的测试和迁移结果。

## Medicine test

| Setting / validation-selected family | R@1 | R@5 | R@10 | MRR | Rescued / harmed vs image |
|---|---:|---:|---:|---:|---:|
| baseline: Qwen3-VL-Embedding-2B/image | 71.43 | 77.55 | 82.99 | 0.7505 | 8 / 18 |
| baseline: bm25/bm25 | 91.16 | 97.96 | 98.64 | 0.9444 | 22 / 3 |
| baseline: bm25_image_fusion/Qwen3-VL-Embedding-2B/alpha_0.5 | 89.12 | 99.32 | 99.32 | 0.9353 | 21 / 5 |
| baseline: bm25_image_fusion/colnomic7b/alpha_0.5 | 89.12 | 99.32 | 99.32 | 0.9381 | 18 / 2 |
| baseline: colnomic3b/image | 76.19 | 86.39 | 88.44 | 0.8051 | 10 / 13 |
| baseline: colnomic7b/image | 78.23 | 91.16 | 91.84 | 0.8419 | 0 / 0 |
| baseline: colpali/image | 57.82 | 65.31 | 68.71 | 0.6144 | 2 / 32 |
| baseline: dinov3/image | 17.01 | 22.45 | 27.89 | 0.2100 | 1 / 91 |
| mixed-box diagnostic: human_boxes/colnomic7b | 78.23 | 88.44 | 90.48 | 0.8312 | 8 / 8 |
| baseline: sam3_auto/colnomic7b | 70.07 | 80.27 | 82.99 | 0.7502 | 3 / 15 |
| baseline: siglip2/image | 25.85 | 34.69 | 36.05 | 0.3021 | 2 / 79 |
| baseline: colpali/bm25_val_calibrated | 91.84 | 97.96 | 98.64 | 0.9480 | 22 / 2 |
| baseline: colnomic3b/bm25_val_calibrated | 90.48 | 97.96 | 98.64 | 0.9410 | 21 / 3 |
| baseline: colnomic7b/bm25_val_calibrated | 89.12 | 99.32 | 100.00 | 0.9401 | 18 / 2 |
| baseline: Qwen3-VL-Embedding-2B/bm25_val_calibrated | 88.44 | 98.64 | 99.32 | 0.9303 | 20 / 5 |
| fixed rule: L0 | 78.23 | 91.16 | 91.84 | 0.8425 | 1 / 1 |
| fixed rule: M | 65.31 | 70.75 | 78.91 | 0.6928 | 11 / 30 |
| fixed rule: weighted_L | 66.67 | 74.83 | 76.87 | 0.7078 | 10 / 27 |
| fixed rule: S | 68.03 | 74.83 | 76.19 | 0.7183 | 11 / 26 |
| strict/hidden | 84.35 | 93.20 | 94.56 | 0.8830 | 10 / 1 |
| strict/patch_hidden | 81.63 | 92.52 | 94.56 | 0.8643 | 6 / 1 |
| catalog/patch_hidden | 78.91 | 91.84 | 93.20 | 0.8450 | 2 / 1 |
| catalog/hidden | 82.99 | 93.20 | 94.56 | 0.8733 | 10 / 3 |
| catalog/embedding | 80.95 | 92.52 | 94.56 | 0.8616 | 5 / 1 |
| strict/embedding | 80.27 | 91.16 | 93.20 | 0.8529 | 5 / 2 |
| strict/reader | 77.55 | 91.84 | 92.52 | 0.8411 | 0 / 1 |
| catalog/reader | 76.19 | 91.16 | 91.84 | 0.8283 | 2 / 5 |
| strict/external | 82.31 | 92.52 | 93.20 | 0.8704 | 7 / 1 |
| catalog/external | 82.31 | 91.16 | 93.20 | 0.8687 | 8 / 2 |

## OCR fusion

RRF 在完整图库融合 BM25 排名与视觉重排后的排名，offset 固定为 60；权重仅在药盒验证集选择。相同的 RRF 基线也单独校准，避免只给新方法调权重。

| Method | OCR weight | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| RAW | 0.50 | 83.67 | 95.24 | 95.92 | 0.8883 |
| strict/hidden | 0.50 | 88.44 | 95.92 | 96.60 | 0.9187 |
| strict/patch_hidden | 0.50 | 86.39 | 96.60 | 97.28 | 0.9110 |
| catalog/patch_hidden | 0.55 | 85.03 | 94.56 | 97.28 | 0.8983 |
| catalog/hidden | 0.30 | 84.35 | 95.24 | 95.24 | 0.8900 |
| catalog/embedding | 0.65 | 86.39 | 95.92 | 97.96 | 0.9087 |
| strict/embedding | 0.50 | 85.71 | 94.56 | 95.92 | 0.9021 |
| strict/reader | 0.50 | 85.71 | 95.24 | 95.92 | 0.9023 |
| catalog/reader | 0.75 | 86.39 | 95.92 | 97.96 | 0.9086 |
| strict/external | 0.50 | 87.76 | 95.24 | 95.24 | 0.9135 |
| catalog/external | 0.15 | 84.35 | 93.20 | 93.88 | 0.8852 |

## Frozen transfer

| Dataset | Checkpoint selected on medicine validation | K | R@1 | R@5 | R@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| grozi | baseline: Qwen3-VL-Embedding-2B/image | full | 93.80 | 100.00 | 100.00 | 0.9638 |
| grozi | baseline: bm25/bm25 | full | 60.90 | 76.50 | 80.77 | 0.6803 |
| grozi | baseline: colnomic3b/image | full | 45.94 | 69.23 | 76.71 | 0.5665 |
| grozi | baseline: colnomic7b/image | full | 81.20 | 96.15 | 97.86 | 0.8772 |
| grozi | baseline: colpali/image | full | 91.88 | 99.15 | 99.57 | 0.9503 |
| grozi | baseline: dinov3/image | full | 90.60 | 99.57 | 99.79 | 0.9416 |
| grozi | baseline: siglip2/image | full | 87.18 | 97.01 | 99.15 | 0.9116 |
| milk | baseline: Qwen3-VL-Embedding-2B | full | 12.32 | 29.44 | 39.67 | 0.2128 |
| milk | baseline: biomedclip | full | 8.77 | 21.29 | 30.48 | 0.1601 |
| milk | baseline: colnomic3b | full | 8.14 | 20.04 | 28.60 | 0.1501 |
| milk | baseline: colnomic7b | full | 12.73 | 25.26 | 35.70 | 0.2023 |
| milk | baseline: colpali | full | 9.60 | 20.25 | 28.18 | 0.1591 |
| milk | baseline: dinov3 | full | 4.38 | 11.27 | 16.91 | 0.0881 |
| milk | baseline: siglip2 | full | 9.39 | 21.71 | 28.60 | 0.1632 |
| forb32 | baseline: Qwen3-VL-Embedding-2B/image | full | 93.61 | 98.48 | 98.80 | 0.9591 |
| forb32 | baseline: bm25/bm25 | full | 70.96 | 86.63 | 88.85 | 0.7802 |
| forb32 | baseline: colnomic7b/image | full | 87.15 | 96.81 | 97.70 | 0.9157 |
| forb32 | baseline: dinov3/image | full | 75.66 | 86.16 | 89.08 | 0.8056 |
| forb32 | baseline: siglip2/image | full | 67.65 | 88.50 | 92.75 | 0.7677 |
| grozi | fixed rule: L0 | 128 | 79.49 | 94.87 | 97.01 | 0.8621 |
| grozi | fixed rule: M | 128 | 85.68 | 97.86 | 98.50 | 0.9045 |
| grozi | fixed rule: weighted_L | 128 | 86.32 | 96.79 | 97.65 | 0.9091 |
| grozi | fixed rule: S | 128 | 88.89 | 97.44 | 97.86 | 0.9241 |
| milk | fixed rule: L0 | 128 | 12.94 | 26.10 | 35.49 | 0.2054 |
| milk | fixed rule: M | 128 | 22.55 | 28.18 | 33.61 | 0.2651 |
| milk | fixed rule: weighted_L | 128 | 25.05 | 29.65 | 34.66 | 0.2880 |
| milk | fixed rule: S | 128 | 24.22 | 29.65 | 34.66 | 0.2807 |
| forb32 | fixed rule: L0 | 32 | 86.65 | 96.64 | 97.54 | 0.9120 |
| forb32 | fixed rule: M | 32 | 94.34 | 97.13 | 97.44 | 0.9571 |
| forb32 | fixed rule: weighted_L | 32 | 95.62 | 97.74 | 97.85 | 0.9666 |
| forb32 | fixed rule: S | 32 | 95.45 | 97.63 | 97.79 | 0.9651 |
| grozi | strict/hidden | 128 | 85.90 | 99.57 | 100.00 | 0.9173 |
| milk | strict/hidden | 128 | 16.49 | 31.94 | 41.54 | 0.2493 |
| forb32 | strict/hidden | 32 | 88.90 | 97.74 | 98.17 | 0.9292 |
| grozi | strict/patch_hidden | 128 | 85.26 | 98.50 | 98.93 | 0.9099 |
| milk | strict/patch_hidden | 128 | 12.11 | 24.43 | 33.82 | 0.1940 |
| forb32 | strict/patch_hidden | 32 | 88.78 | 97.89 | 98.21 | 0.9302 |
| grozi | catalog/patch_hidden | 32 | 86.97 | 98.50 | 99.15 | 0.9214 |
| milk | catalog/patch_hidden | 32 | 11.69 | 25.26 | 36.53 | 0.1993 |
| forb32 | catalog/patch_hidden | 32 | 88.13 | 97.28 | 97.90 | 0.9242 |
| grozi | catalog/hidden | 128 | 87.61 | 99.57 | 99.57 | 0.9263 |
| milk | catalog/hidden | 128 | 19.21 | 34.66 | 43.01 | 0.2718 |
| forb32 | catalog/hidden | 32 | 90.45 | 98.03 | 98.28 | 0.9399 |
| grozi | catalog/embedding | 128 | 86.32 | 99.36 | 99.79 | 0.9203 |
| milk | catalog/embedding | 128 | 21.29 | 36.53 | 42.59 | 0.2908 |
| forb32 | catalog/embedding | 32 | 90.08 | 97.99 | 98.19 | 0.9379 |
| grozi | strict/embedding | 128 | 85.68 | 99.15 | 99.79 | 0.9153 |
| milk | strict/embedding | 128 | 20.88 | 35.28 | 43.42 | 0.2891 |
| forb32 | strict/embedding | 32 | 89.63 | 97.92 | 98.12 | 0.9350 |
| grozi | strict/reader | 128 | 83.76 | 96.79 | 98.08 | 0.8920 |
| milk | strict/reader | 128 | 12.73 | 23.80 | 33.61 | 0.1956 |
| forb32 | strict/reader | 32 | 87.34 | 96.93 | 97.77 | 0.9177 |
| grozi | catalog/reader | 8 | 80.34 | 95.94 | 97.86 | 0.8688 |
| milk | catalog/reader | 8 | 11.48 | 22.96 | 35.70 | 0.1882 |
| forb32 | catalog/reader | 32 | 83.20 | 95.29 | 96.81 | 0.8861 |
| grozi | strict/external | 128 | 86.54 | 99.57 | 99.79 | 0.9206 |
| milk | strict/external | 128 | 23.38 | 35.70 | 43.01 | 0.3009 |
| forb32 | strict/external | 32 | 90.86 | 98.10 | 98.27 | 0.9438 |
| grozi | catalog/external | 128 | 86.54 | 99.36 | 99.79 | 0.9195 |
| milk | catalog/external | 128 | 21.50 | 34.86 | 42.59 | 0.2879 |
| forb32 | catalog/external | 32 | 92.24 | 98.04 | 98.23 | 0.9503 |

## Online timing

下表为每张查询的平均秒数，三个主要方法使用相同硬件和查询样本。ColNomic 图库向量已经建好并驻留，查询图片重新读取和编码。RoMa 图像描述子缓存在每张查询前清空，因此计入了参考图描述子的重复计算；同一查询内会复用查询描述子。模型加载时间与 PyTorch 峰值显存另列在 online_timing.csv 中。

| Method | Queries | K | Encode + search | RoMa | Pool + rerank | Total seconds |
|---|---:|---:|---:|---:|---:|---:|
| strict/embedding | 10 | 128 | 3.220 | 20.506 | 0.389 | 24.115 |
| strict/external | 10 | 128 | 3.194 | 20.418 | 0.228 | 23.840 |
| strict/reader | 10 | 128 | 3.170 | 0.000 | 0.968 | 4.138 |

硬件：i14s47 / Quadro RTX 6000; Quadro RTX 6000.

## Figures

![Paired R@1 changes](figures/paired_r1_changes.png)

点为相对冻结 ColNomic 7B 的 R@1 差值，线为按来源组配对重采样的 95% 区间；包括无需训练的 weighted_L 对照。学习方法跨域均使用同一药盒验证选择的检查点。[PDF](figures/paired_r1_changes.pdf)。

![Candidate sensitivity](figures/candidate_sensitivity.png)

候选数敏感性使用冻结的验证最佳检查点，测试曲线只作诊断；正式 operating K 仍来自验证集。[PDF](figures/candidate_sensitivity.pdf)。

## Files and reproduction

- `tables/all_results.csv`：全部基线、验证、测试、候选数和条件干预结果。R 指标为 0–1，正文表为百分数。
- `tables/ablations.csv` / `seeds.csv`：消融和各次训练随机种子结果；每次运行的检查点及 K 均按验证集选择。
- `tables/transfer.csv`：跨数据集全部结果；FORB 另含官方分母的 mAP 指标。
- `tables/paired_comparisons.csv`：主适配器相对外部评分头、RoMa-free Reader、固定几何对照和主要基线的逐查询配对差值及 95% 区间；按数据集来源组重采样，不只比较纯图片基线。
- `tables/online_timing.csv`：重新读取照片、编码、完整图库检索、必要的 RoMa 和重排的实测端到端耗时；固定图库已建库，模型初始化另计。缓存重排可来自不同 GPU worker，只用于说明计算范围；公平速度比较以注明硬件的端到端实测为准。
- `selection.json`：冻结的选择依据、检查点路径和 SHA-256；`runs/` 保存训练输出；`evaluation/` 保存逐查询预测。
- `verification_final.json`：数据、缓存、训练来源、完整图库指标与检查点一致性检查。

基于现有正式图片、基础 embedding、OCR 和权重缓存复现；以下缓存 worker 可在独立 GPU 上重复启动，文件锁防止同一查询重复写入。训练队列可分别指定空闲 GPU，统计与选择命令应在相应输入完成后执行。

```bash
for dataset in medicine medicine_catalog grozi milk forb forb32; do
  bash scripts/forb_python.sh -m drugpack.method.prepare --dataset "$dataset"
done
CUDA_VISIBLE_DEVICES=0,1 bash scripts/forb_python.sh -m drugpack.method.hidden --datasets medicine grozi milk forb
CUDA_VISIBLE_DEVICES=0 bash scripts/forb_python.sh -m drugpack.method.roma --precision fp16 --datasets medicine medicine_catalog grozi milk forb32 --watch
bash scripts/forb_python.sh -m drugpack.method.features --datasets medicine medicine_catalog grozi milk forb32 --watch
bash scripts/forb_python.sh -m drugpack.method.queue --gpu 0
bash scripts/forb_python.sh -m drugpack.method.queue --gpu cpu0 --jobs configs/method/head_jobs.json
for dataset in medicine medicine_catalog; do
  for loss in cost ce; do
    for kind in INTERNAL3 RAW_M ADDITIVE4 PRODUCT5 WEIGHTED7; do
      bash scripts/forb_python.sh -m drugpack.method.heads --dataset "$dataset" --loss "$loss" --kind "$kind" --wait
    done
  done
done
bash scripts/forb_python.sh -m drugpack.method.baselines --phase validation
bash scripts/forb_python.sh -m drugpack.method.select
bash scripts/forb_python.sh -m drugpack.method.fusion --phase validation
bash scripts/forb_python.sh -m drugpack.method.dispatch
bash scripts/forb_python.sh -m drugpack.method.queue --gpu 0 --jobs configs/method/evaluation_gpu_jobs.json
bash scripts/forb_python.sh -m drugpack.method.queue --gpu cpu0 --jobs configs/method/evaluation_cpu_jobs.json
bash scripts/forb_python.sh -m drugpack.method.baselines --phase final
bash scripts/forb_python.sh -m drugpack.method.fusion --phase final
# Online benchmark: two 24 GB GPUs, matching the measured RTX 6000 setup.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/forb_python.sh -m drugpack.method.benchmark --suite --single-encoder
bash scripts/forb_python.sh -m drugpack.method.verify
bash scripts/forb_python.sh -m drugpack.method.report
```
