# photo_92b44955f8a40dbb

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_005348`，F03 错误参考 `reference_000021`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_005348`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_005348 |
| BM25 真值排名 | 4 |
| 融合真值排名 | 4 |
| 图像正负分差 | +0.040558 |
| BM25 正负分差 | -60.419388 |
| 0.5 × 图像 z 分差 | +0.727918 |
| 0.5 × BM25 z 分差 | -2.241044 |
| 最终融合分差 | -1.513126 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.7548**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 273 个。正向贡献 +0.156159，负向贡献 -0.115600，相加等于图像分差 +0.040558。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 495 | -0.068034 | -0.107068 |
| nonvisual | 11 | +0.001844 | +0.130605 |
| subject_nontext | 118 | +0.021120 | +0.139428 |
| subject_text | 155 | +0.085628 | +0.430351 |

仅保留主体 query token 的两候选分差：+0.304604。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.068034。主体框内 token 占有效序列 35.0%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 154 | 45.2% | True | +13.650980 | 72.2% |
| 错误 | 33 | 34.0% | True | +11.427021 | 98.5% |

主体中 155 个 token 与 OCR 相交，35 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 389 | background | -0.700881 | Dewormer | dispensing information included. | Dewormer |
| 622 | subject_nontext | -0.673393 |  | Read accompanying directions |  |
| 678 | background | -0.670357 |  | 140 mg | Dewormer |
| 327 | background | -0.658966 | Canine |  | Canine |
| 759 | background | -0.652548 |  | Temozolomide |  |
| 551 | background | -0.647374 |  | 2072308 | (Taenia pisiformis). |
| 520 | background | -0.646915 | la). Whipworms (Trichuris vulpis), and Tapeworms |  |  |
| 582 | background | -0.645944 |  | Temozolomide |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 149 | subject_text | +0.706305 | CAMBER | CAMBER | Genveticon |
| 148 | subject_text | +0.689898 | CAMBER | CAMBER |  |
| 117 | subject_text | +0.677150 | CAMBER | CAMBER | parasitism. |
| 147 | subject_text | +0.671733 | CAMBER | CAMBER |  |
| 150 | subject_text | +0.668374 | CAMBER | CAMBER |  |
| 370 | subject_text | +0.664163 | Capsules, USP | Capsules, USP | parasitism. |
| 338 | subject_text | +0.660184 | Capsules, USP | Capsules, USP |  |
| 339 | subject_text | +0.656688 | Capsules, USP | Capsules, USP | parasitism. |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 140 mg | 12 | +0.005158 |
| 5 Capsules | 10 | +0.005598 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_420054ef52bb8e29` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.040558 | +0.234227 |
| 仅主体分差 | +0.304604 | +0.341968 |
| 主体 token 数 | 273 | 322 |

[对照 token 数据](../photo_420054ef52bb8e29/tokens.csv) · [对照摘要](../photo_420054ef52bb8e29/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
