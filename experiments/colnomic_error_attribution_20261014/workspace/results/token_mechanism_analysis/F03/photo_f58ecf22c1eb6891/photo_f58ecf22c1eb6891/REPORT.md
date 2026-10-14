# photo_f58ecf22c1eb6891

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_003534`，F03 错误参考 `reference_003536`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_003534`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_003534 |
| BM25 真值排名 | 5 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.002102 |
| BM25 正负分差 | -3.967812 |
| 0.5 × 图像 z 分差 | +0.037482 |
| 0.5 × BM25 z 分差 | -0.233890 |
| 最终融合分差 | -0.196408 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.8619**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 300 个。正向贡献 +0.020492，负向贡献 -0.018390，相加等于图像分差 +0.002102。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 468 | -0.002127 | -0.003541 |
| nonvisual | 11 | -0.000103 | -0.007302 |
| subject_nontext | 152 | -0.000088 | -0.000451 |
| subject_text | 148 | +0.004420 | +0.023267 |

仅保留主体 query token 的两候选分差：+0.011250。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.002127。主体框内 token 占有效序列 38.5%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 465 | 70.2% | True | +4.256115 | 54.5% |
| 错误 | 67 | 14.3% | False | 不可评估 | 不可评估 |

主体中 148 个 token 与 OCR 相交，29 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 130 | background | -0.220575 |  | NDC 81929-004-02 | NDC 81929-004-01 |
| 358 | subject_nontext | -0.187282 |  |  | DERMAN |
| 143 | background | -0.170441 |  | ANTIFUNGAL CREAM | DERMAN |
| 305 | subject_nontext | -0.163541 |  |  | DERMAN |
| 440 | subject_nontext | -0.162302 |  | DERMAN | DERMAN |
| 671 | background | -0.156377 | antworten Ihre Fragen dazu. |  |  |
| 214 | background | -0.149596 |  |  | DERMAN |
| 407 | subject_text | -0.144847 | ERMAN |  | DERMAN |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 567 | subject_text | +0.544967 | NET WT. 0.88 OZ (25 g) | NET WT. 0.88 OZ (25 g) | NET WT. 1.76 OZ (50 g) |
| 535 | subject_text | +0.513581 | NET WT. 0.88 OZ (25 g) | NET WT. 0.88 OZ (25 g) | NET WT. 1.76 OZ (50 g) |
| 566 | subject_text | +0.446073 | NET WT. 0.88 OZ (25 g) | NET WT. 0.88 OZ (25 g) |  |
| 539 | subject_text | +0.378914 | NET WT. 0.88 OZ (25 g) | NET WT. 0.88 OZ (25 g) | NET WT. 1.76 OZ (50 g) |
| 571 | subject_text | +0.362508 | NET WT. 0.88 OZ (25 g) | NET WT. 0.88 OZ (25 g) | NET WT. 1.76 OZ (50 g) |
| 598 | subject_text | +0.328930 | NET WT. 0.88 OZ (25 g) | NET WT. 0.88 OZ (25 g) | NET WT. 1.76 OZ (50 g) |
| 59 | background | +0.315699 |  | NET WT. 0.88 OZ (25 g) | NET WT. 1.76 OZ (50 g) |
| 589 | subject_nontext | +0.298954 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Tolnaftate 1% | 6 | -0.000027 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
