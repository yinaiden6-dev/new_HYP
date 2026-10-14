# photo_45b5970f1ebe4f19

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_004975`，F03 错误参考 `reference_004976`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_004975`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_004975 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.001765 |
| BM25 正负分差 | -15.098572 |
| 0.5 × 图像 z 分差 | +0.043184 |
| 0.5 × BM25 z 分差 | -0.589828 |
| 最终融合分差 | -0.546644 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.9318**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 99 个。正向贡献 +0.020648，负向贡献 -0.018883，相加等于图像分差 +0.001765。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 669 | +0.001278 | +0.001488 |
| nonvisual | 11 | -0.000085 | -0.006040 |
| subject_nontext | 41 | -0.000607 | -0.011540 |
| subject_text | 58 | +0.001180 | +0.015842 |

仅保留主体 query token 的两候选分差：+0.004502。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_nontext`，贡献 -0.000607。主体框内 token 占有效序列 12.7%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 191 | 39.4% | True | +6.094430 | 70.8% |
| 错误 | 117 | 33.6% | True | +6.186881 | 65.3% |

主体中 58 个 token 与 OCR 相交，26 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 625 | background | -0.274970 |  | Ronly |  |
| 161 | background | -0.254873 |  | Ronly |  |
| 538 | background | -0.223139 |  | Ronly |  |
| 537 | background | -0.218586 |  |  |  |
| 49 | background | -0.203078 |  |  |  |
| 81 | background | -0.197052 |  |  |  |
| 490 | background | -0.175167 |  |  |  |
| 235 | background | -0.156733 |  | Water for Injection. pH adjusted with | Water for Injection. pH adjusted with |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 599 | background | +0.371201 | FoyerA |  |  |
| 770 | background | +0.317016 |  |  |  |
| 467 | subject_text | +0.284958 | Protect from light | Protect from light. |  |
| 23 | background | +0.236241 | armacy |  |  |
| 431 | subject_text | +0.232560 | Each mL contains 2.5 mg bupivacaine |  |  |
| 55 | background | +0.208127 | armacy |  | Manufactured by: |
| 521 | background | +0.203143 |  |  |  |
| 367 | subject_text | +0.197256 | Each mL contains 2.5 mg bupivacaine |  | Bupivacaine Hydrochloride |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Each mL contains 2.5 mg bupivacaine | 12 | +0.000660 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_bbbf0bcaee0cc855` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.001765 | +0.013909 |
| 仅主体分差 | +0.004502 | +0.035750 |
| 主体 token 数 | 99 | 144 |

[对照 token 数据](../photo_bbbf0bcaee0cc855/tokens.csv) · [对照摘要](../photo_bbbf0bcaee0cc855/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
