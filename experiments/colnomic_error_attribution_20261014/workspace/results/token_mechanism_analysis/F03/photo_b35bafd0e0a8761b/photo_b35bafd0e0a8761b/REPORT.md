# photo_b35bafd0e0a8761b

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001012`，F03 错误参考 `reference_002644`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_001012`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_001012 |
| BM25 真值排名 | 8 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.012761 |
| BM25 正负分差 | -16.748001 |
| 0.5 × 图像 z 分差 | +0.346037 |
| 0.5 × BM25 z 分差 | -0.594683 |
| 最终融合分差 | -0.248645 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.6322**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 210 个。正向贡献 +0.035606，负向贡献 -0.022844，相加等于图像分差 +0.012761。缓存复现误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 558 | -0.000434 | -0.000606 |
| nonvisual | 11 | -0.000031 | -0.002185 |
| subject_nontext | 96 | +0.007794 | +0.063242 |
| subject_text | 114 | +0.005432 | +0.037121 |

仅保留主体 query token 的两候选分差：+0.049062。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.000434。主体框内 token 占有效序列 27.0%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 28 | 33.0% | True | +12.524919 | 93.2% |
| 错误 | 5 | 21.2% | False | 不可评估 | 不可评估 |

主体中 114 个 token 与 OCR 相交，31 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 359 | background | -0.318449 |  |  |  |
| 391 | background | -0.293515 |  | after the first bowel movement, or as directed |  |
| 8 | background | -0.279746 | ay | Softener | Gentle Relief |
| 713 | background | -0.234046 |  |  | Provides safe, gentle |
| 133 | background | -0.194365 |  |  |  |
| 152 | background | -0.173167 | Thè nupe te aege d è ee letiss co | NDC37808-582-01 | Inactive Ingredlents cit ic add, FD &C rod# 40, FD&C yello w #6, gdatin, gycerin, podlyethylene |
| 159 | background | -0.172852 | Measurea Sperding Iime in | Gentle | store at roo m tamperat ure 15°-30℃ (59-86°F) |
| 394 | background | -0.163453 |  |  | mar |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 747 | background | +0.565515 |  |  |  |
| 409 | subject_nontext | +0.564715 |  |  | Gentle Relief |
| 30 | background | +0.561149 | gel medical hel | professional before use. | drug |
| 634 | subject_nontext | +0.551270 |  |  |  |
| 410 | subject_nontext | +0.532521 |  | MADE WITH PRIDE AND CARE FOR H-E-B® |  |
| 377 | subject_text | +0.512398 | H·E· | H·E·B | take 1 to 3softgels daly.This d ose may be taken |
| 27 | background | +0.479129 | Keep ouk of reach of |  | Ask a d oct or o r ph armacist b ef ore use if you are presently talin g min eral oil |
| 90 | background | +0.463178 |  |  | Questions orcomments? Call toll fee: 1-800-833-6278 |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Docusate Sodium, 100 mg | 28 | -0.000305 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_b5df0dce47884a73` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.012761 | +0.037795 |
| 仅主体分差 | +0.049062 | +0.098308 |
| 主体 token 数 | 210 | 242 |

[对照 token 数据](../photo_b5df0dce47884a73/tokens.csv) · [对照摘要](../photo_b5df0dce47884a73/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
