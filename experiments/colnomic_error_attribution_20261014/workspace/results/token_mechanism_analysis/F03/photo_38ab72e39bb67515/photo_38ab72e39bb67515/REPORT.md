# photo_38ab72e39bb67515

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000929`，F03 错误参考 `reference_002389`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_001896`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2167 |
| 图像 Top-1 参考 | reference_001896 |
| BM25 真值排名 | 5 |
| 融合真值排名 | 5 |
| 图像正负分差 | -0.008881 |
| BM25 正负分差 | -1.048689 |
| 0.5 × 图像 z 分差 | -0.268588 |
| 0.5 × BM25 z 分差 | -0.139476 |
| 最终融合分差 | -0.408064 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 144 个。正向贡献 +0.016533，负向贡献 -0.025414，相加等于图像分差 -0.008881。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 576 | -0.005793 | -0.007352 |
| nonvisual | 11 | -0.000057 | -0.003808 |
| subject_nontext | 113 | -0.002338 | -0.015127 |
| subject_text | 31 | -0.000692 | -0.016312 |

仅保留主体 query token 的两候选分差：-0.015382。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.005793。主体框内 token 占有效序列 19.7%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 17 | 9.5% | False | 不可评估 | 不可评估 |
| 错误 | 9 | 29.4% | True | +18.111413 | 96.5% |

主体中 31 个 token 与 OCR 相交，10 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 427 | background | -0.233137 |  |  |  |
| 253 | background | -0.221442 |  |  |  |
| 501 | background | -0.219014 |  |  |  |
| 199 | background | -0.211824 |  | Healthcare, L.P., distributor of Flonase® Allergy Relief. |  |
| 186 | background | -0.203852 |  | Healthcare, L.P., distributor of Flonase® Allergy Relief. |  |
| 515 | subject_nontext | -0.189462 |  | Compare to the active ingredient |  |
| 82 | background | -0.187132 |  | Healthcare, L.P., distributor of Flonase® Allergy Relief. |  |
| 575 | background | -0.180611 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 87 | background | +0.486046 |  |  | keep this label and enclosed materials. |
| 448 | background | +0.200493 |  |  |  |
| 643 | background | +0.183854 |  |  |  |
| 720 | background | +0.181402 |  |  |  |
| 513 | subject_nontext | +0.180599 |  |  | Inactive ingredients |
| 556 | background | +0.168164 |  |  |  |
| 165 | background | +0.159955 |  |  |  |
| 43 | background | +0.156477 |  |  |  |

## 成功对照

`photo_cdd1aa422a4f6dc0` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008881 | +0.072623 |
| 仅主体分差 | -0.015382 | +0.098508 |
| 主体 token 数 | 144 | 150 |

[对照 token 数据](../photo_cdd1aa422a4f6dc0/tokens.csv) · [对照摘要](../photo_cdd1aa422a4f6dc0/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
