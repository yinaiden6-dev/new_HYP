# photo_c798e089f34805e9

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001012`，F03 错误参考 `reference_002230`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_002223`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_002223 |
| BM25 真值排名 | 6 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.017477 |
| BM25 正负分差 | -24.814056 |
| 0.5 × 图像 z 分差 | +0.284985 |
| 0.5 × BM25 z 分差 | -0.286470 |
| 最终融合分差 | -0.001485 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5013**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 264 个。正向贡献 +0.039227，负向贡献 -0.021749，相加等于图像分差 +0.017477。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 504 | +0.002306 | +0.003564 |
| nonvisual | 11 | +0.000020 | +0.001396 |
| subject_nontext | 73 | +0.004846 | +0.051713 |
| subject_text | 191 | +0.010306 | +0.042032 |

仅保留主体 query token 的两候选分差：+0.044709。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 +0.000020。主体框内 token 占有效序列 33.9%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 386 | 73.6% | True | +6.650672 | 63.9% |
| 错误 | 10 | 16.0% | True | +7.112451 | 71.0% |

主体中 191 个 token 与 OCR 相交，104 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 731 | background | -0.639263 | CVS | Drug Facts |  |
| 763 | background | -0.386201 | pharmacy |  | Distributed by: CVS Pharmacy, Inc. One CVS Drive. |
| 39 | background | -0.360753 | Kids Ear Refief |  |  |
| 328 | background | -0.328937 |  |  |  |
| 478 | subject_text | -0.310729 | overdose, get medical help or contact a Poison | Keep out of reach of children. In case of | medical help or contact a Poison Control Center right away |
| 762 | background | -0.305134 | pharmacy | Drug Facts | Distributed by: CVS Pharmacy, Inc. One CVS Drive. |
| 377 | subject_text | -0.285914 | you have rectal bleeding or fail to have a | a serious condition. | you have rectal bleeding or fail to have a bowel |
| 648 | background | -0.284640 | Relleves redness, watery | Laxative |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 341 | subject_nontext | +0.412952 |  | have stomach pain, nausea or vomiting | Ask a doctor before use if you have |
| 499 | subject_text | +0.402697 | generally produces bowel movement in 12 |  | If pregnant or breast-feeding, ask a health |
| 406 | subject_text | +0.399675 | have stomach pain, nausea or vomiting | Ask a doctor before use if you | Ask a doctor before use if you have |
| 342 | subject_text | +0.392589 | have stomach pain, nausea or vomiting | have stomach pain, nausea or vomiting | Stop use and ask a doctor if |
| 505 | subject_text | +0.385434 | you have rectal bleeding or fail to have a | Stop use and ask a doctor if | noticed a sudden change in bowel habits that |
| 484 | background | +0.381895 |  |  |  |
| 340 | subject_nontext | +0.375265 |  |  |  |
| 531 | subject_text | +0.361396 | generally produces bowel movement in 12 |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Docusate Sodium 100 mg. | 13 | +0.001157 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_b5df0dce47884a73` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.017477 | +0.043838 |
| 仅主体分差 | +0.044709 | +0.105758 |
| 主体 token 数 | 264 | 242 |

[对照 token 数据](../photo_b5df0dce47884a73/tokens.csv) · [对照摘要](../photo_b5df0dce47884a73/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
