# photo_5964b5e1bd6ef958

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_003368`，F03 错误参考 `reference_003367`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 657 |
| 图像 Top-1 参考 | reference_003367 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 18 |
| 图像正负分差 | -0.069397 |
| BM25 正负分差 | +7.244972 |
| 0.5 × 图像 z 分差 | -2.041923 |
| 0.5 × BM25 z 分差 | +0.343817 |
| 最终融合分差 | -1.698107 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w < 0.1441**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 156 个。正向贡献 +0.039705，负向贡献 -0.109102，相加等于图像分差 -0.069397。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 612 | -0.066963 | -0.085236 |
| nonvisual | 11 | -0.001328 | -0.094073 |
| subject_nontext | 109 | -0.006270 | -0.044812 |
| subject_text | 47 | +0.005165 | +0.085606 |

仅保留主体 query token 的两候选分差：-0.005519。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.066963。主体框内 token 占有效序列 20.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 53 | 23.3% | True | +10.342774 | 76.1% |
| 错误 | 9 | 2.9% | False | 不可评估 | 不可评估 |

主体中 47 个 token 与 OCR 相交，19 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 693 | background | -0.643253 |  |  | 7 |
| 703 | background | -0.618148 |  |  | 7 |
| 745 | background | -0.614889 |  |  |  |
| 297 | background | -0.610800 | CVS |  | CVS |
| 685 | background | -0.610053 |  |  | 7 |
| 616 | background | -0.607698 |  |  | 7 |
| 709 | background | -0.605354 |  |  | 7 |
| 458 | background | -0.595653 | NASAL AL | TIIN cen | NASAL MIST |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 338 | subject_nontext | +0.519880 |  |  |  |
| 628 | subject_nontext | +0.485965 |  | ElizabethArden | temporarily relieves sinus congestion and pressure |
| 465 | subject_nontext | +0.464518 |  |  |  |
| 65 | background | +0.461112 |  |  |  |
| 536 | subject_text | +0.458512 | PLACE UPC HERE | PLACE UPC HERE |  |
| 309 | subject_nontext | +0.457546 |  |  | or down throat |
| 468 | subject_text | +0.450932 | ©Elizabeth Arden, Inc. | New York, NY 10003 | more often than every 10 to 12 hours. Do not exceed |
| 307 | subject_nontext | +0.450509 |  |  | or down throat |

## 成功对照

`photo_9872ed71c7af7565` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.069397 | +0.043611 |
| 仅主体分差 | -0.005519 | +0.176978 |
| 主体 token 数 | 156 | 225 |

[对照 token 数据](../photo_9872ed71c7af7565/tokens.csv) · [对照摘要](../photo_9872ed71c7af7565/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
