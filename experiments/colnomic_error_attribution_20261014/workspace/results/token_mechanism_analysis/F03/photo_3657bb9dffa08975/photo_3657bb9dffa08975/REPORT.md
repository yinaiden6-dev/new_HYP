# photo_3657bb9dffa08975

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000099`，F03 错误参考 `reference_000105`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_000099`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_000099 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.011478 |
| BM25 正负分差 | -4.083092 |
| 0.5 × 图像 z 分差 | +0.271534 |
| 0.5 × BM25 z 分差 | -0.486662 |
| 最终融合分差 | -0.215128 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.6419**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 391 个。正向贡献 +0.031104，负向贡献 -0.019626，相加等于图像分差 +0.011478。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 377 | +0.000354 | +0.000731 |
| nonvisual | 11 | -0.000064 | -0.004501 |
| subject_nontext | 244 | +0.005063 | +0.016166 |
| subject_text | 147 | +0.006125 | +0.032456 |

仅保留主体 query token 的两候选分差：+0.022290。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 -0.000064。主体框内 token 占有效序列 50.2%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 277 | 55.2% | True | +8.784557 | 76.2% |
| 错误 | 49 | 44.0% | True | +12.155275 | 94.3% |

主体中 147 个 token 与 OCR 相交，44 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 254 | subject_nontext | -0.255031 |  |  |  |
| 311 | subject_nontext | -0.236243 |  |  |  |
| 471 | subject_nontext | -0.229164 |  |  |  |
| 249 | subject_nontext | -0.189052 |  |  |  |
| 485 | background | -0.181794 |  |  |  |
| 506 | subject_text | -0.170576 | oral use | For oral use | infestations on dogs |
| 538 | subject_text | -0.165547 | oral use | For oral use | infestations on dogs |
| 651 | background | -0.163439 |  | (nitenpyram) |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 317 | subject_nontext | +0.624133 |  | For dogs over 25 lbs \| 6 tablets (57.0 mg each) |  |
| 541 | subject_text | +0.587611 | he reach of children | For dogs over 25 lbs \| 6 tablets (57.0 mg each) |  |
| 509 | subject_text | +0.567062 | he reach of children | For dogs over 25 lbs \| 6 tablets (57.0 mg each) |  |
| 573 | subject_text | +0.499826 | he reach of children | Keep this and all drugs out of the reach of children |  |
| 669 | subject_nontext | +0.481607 |  |  |  |
| 605 | subject_text | +0.461582 | he reach of children | Keep this and all drugs out of the reach of children |  |
| 94 | subject_nontext | +0.433770 |  | Keep this and all drugs out of the reach of children |  |
| 96 | background | +0.411666 |  | Keep this and all drugs out of the reach of children |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 57.0 mg | 6 | +0.000588 |
| tablets (57.0 mg each) | 17 | -0.000258 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_42f6f613cad79d1c` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.011478 | +0.034701 |
| 仅主体分差 | +0.022290 | +0.045368 |
| 主体 token 数 | 391 | 390 |

[对照 token 数据](../photo_42f6f613cad79d1c/tokens.csv) · [对照摘要](../photo_42f6f613cad79d1c/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
