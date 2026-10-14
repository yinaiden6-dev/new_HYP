# photo_a0dda9c59f0160fe

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_004360`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_004360 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.018815 |
| BM25 正负分差 | -0.530151 |
| 0.5 × 图像 z 分差 | -0.411200 |
| 0.5 × BM25 z 分差 | -0.005729 |
| 最终融合分差 | -0.416928 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 276 个。正向贡献 +0.019214，负向贡献 -0.038029，相加等于图像分差 -0.018815。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 492 | -0.011033 | -0.017468 |
| nonvisual | 11 | -0.000111 | -0.007880 |
| subject_nontext | 40 | -0.001486 | -0.028931 |
| subject_text | 236 | -0.006186 | -0.020419 |

仅保留主体 query token 的两候选分差：-0.021652。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.011033。主体框内 token 占有效序列 35.4%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 227 | 68.2% | True | +7.787549 | 69.7% |
| 错误 | 10 | 30.4% | True | +17.524031 | 97.1% |

主体中 236 个 token 与 OCR 相交，186 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 262 | background | -0.550775 | CVS | 73460 UW C3 | See inside for more details or |
| 230 | background | -0.512014 |  | 73460 UW C3 | See inside for more details or |
| 263 | background | -0.501586 | CVS | *This product is not manufactured or distrbuted by GlaxoSmithKline | See inside for more details or |
| 294 | background | -0.481856 | pharmacy | *This product is not manufactured or distrbuted by GlaxoSmithKline |  |
| 84 | background | -0.389603 |  |  |  |
| 535 | subject_text | -0.380534 | 6 hours. Do not use more than 20 lozenges per day.  it is |  | it is important to complete treatment. If you feel you |
| 270 | subject_nontext | -0.377742 |  |  |  |
| 252 | background | -0.364242 |  | USES, DIRECTIONS AND WARNINGS. |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 599 | subject_text | +0.287084 | 6 hours. Do not use more than 20 lozenges per day.  it is |  | 1 lozenge every |
| 434 | subject_text | +0.266643 | tingling sensation occasionally move the lozenge from one | Do not chew or swallow lozenge. you may feel a warm or | you may feel a warm or tingling sensation |
| 506 | subject_text | +0.259983 | keep vial tightly closed and protect from light | Other informationstore at 20-25℃ (68-77°F) | do not use more than one lozenge at a time or continuously |
| 545 | background | +0.209876 |  |  |  |
| 396 | subject_text | +0.195612 | first cigarette more than 30 minutes after waking up, use | If you smoke your first cigarette WITHIN | do not eat or drink 15 minutes before using or while the |
| 284 | subject_nontext | +0.190418 |  |  | Active ingredient (in each lozenge) |
| 538 | subject_text | +0.189162 | keep vial tightly closed and protect from light | hon | you get symptoms of nicotine overdose such as nausea, vomiting, dizziness, diarrhea, weakness |
| 232 | background | +0.188231 | 24125514 | 094 09 0036 R00 C-002262-01-008-0000 |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| waking up, use 4 mg nicotine lozenge if you smoke your | 22 | +0.000474 |
| 2 mg nicotine lozenge aocording to the following 12 week | 19 | -0.000163 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.018815 | +0.045373 |
| 仅主体分差 | -0.021652 | +0.094385 |
| 主体 token 数 | 276 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
