# photo_b080c24ec8e8e236

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000100`，F03 错误参考 `reference_000104`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_000100`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_000100 |
| BM25 真值排名 | 4 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.003187 |
| BM25 正负分差 | -12.869080 |
| 0.5 × 图像 z 分差 | +0.062646 |
| 0.5 × BM25 z 分差 | -0.288292 |
| 最终融合分差 | -0.225646 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.8215**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 273 个。正向贡献 +0.024267，负向贡献 -0.021079，相加等于图像分差 +0.003187。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 495 | -0.002518 | -0.003962 |
| nonvisual | 11 | -0.000228 | -0.016140 |
| subject_nontext | 146 | +0.003419 | +0.018241 |
| subject_text | 127 | +0.002514 | +0.015422 |

仅保留主体 query token 的两候选分差：+0.016929。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.002518。主体框内 token 占有效序列 35.0%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 119 | 47.2% | True | +11.251738 | 87.0% |
| 错误 | 30 | 7.6% | False | 不可评估 | 不可评估 |

主体中 127 个 token 与 OCR 相交，52 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 309 | background | -0.377730 |  |  | G003117 AHIH |
| 8 | background | -0.367152 | For oral use In dogs only | infestations on cats |  |
| 629 | background | -0.312478 |  |  | G003117 AHIH |
| 628 | background | -0.255614 |  |  |  |
| 339 | subject_text | -0.219979 | 0 mL (16.91 fl oz) |  |  |
| 464 | subject_nontext | -0.211740 |  |  |  |
| 468 | background | -0.169052 | es And Prevents Most Athletes Foot |  |  |
| 153 | background | -0.166397 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 6 | background | +0.508482 | For oral use In dogs only | For oral use |  |
| 527 | subject_text | +0.477579 | For oral use | For oral use |  |
| 7 | background | +0.394506 | For oral use In dogs only | For oral use |  |
| 144 | subject_nontext | +0.272804 |  | A single dose of CAPSTAR should kill the adult fleas on your pet. If your pet gets re-infested with |  |
| 721 | subject_nontext | +0.255597 |  |  |  |
| 525 | subject_nontext | +0.237866 |  |  |  |
| 297 | subject_nontext | +0.233302 |  |  |  |
| 241 | subject_nontext | +0.208126 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 11.4mg | 3 | +0.000163 |
| For cats 2-25 lbs6 tablets (11.4 mg each) | 18 | +0.000083 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_2d3809ffa08d826a` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.003187 | +0.056790 |
| 仅主体分差 | +0.016929 | +0.075831 |
| 主体 token 数 | 273 | 390 |

[对照 token 数据](../photo_2d3809ffa08d826a/tokens.csv) · [对照摘要](../photo_2d3809ffa08d826a/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
