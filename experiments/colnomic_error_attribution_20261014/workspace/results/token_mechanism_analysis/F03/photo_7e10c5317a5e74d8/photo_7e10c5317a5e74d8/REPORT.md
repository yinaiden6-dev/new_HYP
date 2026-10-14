# photo_7e10c5317a5e74d8

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001971`，F03 错误参考 `reference_001973`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001971`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001971 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.001255 |
| BM25 正负分差 | -15.733479 |
| 0.5 × 图像 z 分差 | +0.012202 |
| 0.5 × BM25 z 分差 | -2.125995 |
| 最终融合分差 | -2.113793 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.9943**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 460 个。正向贡献 +0.020855，负向贡献 -0.019600，相加等于图像分差 +0.001255。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 308 | +0.002157 | +0.005454 |
| nonvisual | 11 | -0.000129 | -0.009146 |
| subject_nontext | 212 | +0.001720 | +0.006321 |
| subject_text | 248 | -0.002493 | -0.007830 |

仅保留主体 query token 的两候选分差：-0.001308。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `subject_text`，贡献 -0.002493。主体框内 token 占有效序列 59.1%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 124 | 40.4% | True | +10.343383 | 87.3% |
| 错误 | 40 | 16.4% | True | +7.240805 | 85.9% |

主体中 248 个 token 与 OCR 相交，67 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 217 | subject_text | -0.262221 | mg TABLETS/ | ACRUAL 92 | BUPROFEN (NSAID) 125 mg |
| 411 | subject_text | -0.254172 | IBUPRO FEN (NSAD) | ACETAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | BUPROFEN (NSAID) 125 mg |
| 120 | subject_nontext | -0.251011 |  |  |  |
| 257 | background | -0.247606 |  | ACETAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | BUPROFEN (NSAID) 125 mg |
| 443 | subject_text | -0.244677 | IBUPRO FEN (NSAD) | ACETAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | BUPROFEN (NSAID) 125 mg |
| 410 | subject_text | -0.227887 | IBUPRO FEN (NSAD) | ACETAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | BUPROFEN (NSAID) 125 mg |
| 346 | subject_text | -0.215426 | 125 | ACETAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | 125 mg TABLETS / PAIN RELIEVER |
| 442 | subject_text | -0.214507 | IBUPRO FEN (NSAD) | ACETAMINOPHEN 250 mg & IBUPROFEN (NSAID) 125 mg TABLETS / | BUPROFEN (NSAID) 125 mg |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 667 | subject_text | +0.240279 | IBUPRO FEN (NSAD) |  | ACETAMINOPHEN 250 mg |
| 634 | subject_text | +0.218207 | IBUPRO FEN (NSAD) |  |  |
| 561 | subject_text | +0.202773 | Dual Pain |  |  |
| 93 | subject_nontext | +0.185426 |  |  |  |
| 680 | background | +0.169304 |  |  |  |
| 401 | subject_text | +0.165527 | Dual Pain | 1 |  |
| 240 | subject_nontext | +0.156978 |  |  |  |
| 26 | subject_nontext | +0.142370 |  |  | TABLETS/PAIN RELIEVER |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 250mg | 24 | -0.001219 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_ca454f8a9d866abb` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.001255 | +0.001656 |
| 仅主体分差 | -0.001308 | -0.010628 |
| 主体 token 数 | 460 | 264 |

[对照 token 数据](../photo_ca454f8a9d866abb/tokens.csv) · [对照摘要](../photo_ca454f8a9d866abb/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
