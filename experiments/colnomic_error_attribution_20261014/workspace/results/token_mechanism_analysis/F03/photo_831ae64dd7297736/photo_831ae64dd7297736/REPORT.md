# photo_831ae64dd7297736

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000142`，F03 错误参考 `reference_000143`。图像 Top-1 原本正确，融合后才变错。 纯图像 Top-1 为 `reference_000142`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1 |
| 图像 Top-1 参考 | reference_000142 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | +0.006583 |
| BM25 正负分差 | -4.165375 |
| 0.5 × 图像 z 分差 | +0.100438 |
| 0.5 × BM25 z 分差 | -0.114060 |
| 最终融合分差 | -0.013623 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**图像权重 w > 0.5318**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 98 个。正向贡献 +0.025853，负向贡献 -0.019270，相加等于图像分差 +0.006583。缓存复现误差 1.2e-07。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 622 | +0.002989 | +0.003513 |
| nonvisual | 11 | +0.000009 | +0.000612 |
| subject_nontext | 33 | +0.000865 | +0.019157 |
| subject_text | 65 | +0.002720 | +0.030587 |

仅保留主体 query token 的两候选分差：+0.026738。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `nonvisual`，贡献 +0.000009。主体框内 token 占有效序列 13.4%。主体 token 支持正确候选，仍需检查其他图库候选与最终融合。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 351 | 65.7% | True | +1.380525 | 36.7% |
| 错误 | 252 | 53.7% | True | +2.302958 | 45.9% |

主体中 65 个 token 与 OCR 相交，42 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 481 | background | -0.295356 |  |  | Approved by FDA under NADA # 141-528 |
| 631 | background | -0.213277 |  | CAUTION: Federal law restricts |  |
| 655 | background | -0.206466 |  | this drug to use by or on the order |  |
| 461 | background | -0.202029 |  |  | CAUTION: Federal law restricts |
| 465 | background | -0.195268 |  | of a licensed veterinarian. | CAUTION: Federal law restricts |
| 175 | background | -0.185443 |  | CAT (lotilaner) |  |
| 562 | background | -0.182350 |  |  |  |
| 234 | subject_text | -0.176362 | Cat &  | 1 TAÁBLET | Cats & |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 353 | subject_text | +0.390339 | 12 mg chewable tablet | 12 mg chewable tablet |  |
| 115 | background | +0.371945 |  | 2.0-4.0 lbs |  |
| 239 | background | +0.321364 |  | 2.0-4.0 lbs | 4.1-17.0 lbs |
| 216 | background | +0.308149 |  | 2.0-4.0 lbs | 4.1-17.0 lbs |
| 454 | subject_nontext | +0.306563 |  |  | CAT (lotilaner) |
| 176 | background | +0.302577 |  | 2.0-4.0 lbs | 4.1-17.0 lbs |
| 448 | background | +0.298949 |  |  |  |
| 235 | subject_text | +0.298550 | 2.0-4.0 lbs | 2.0-4.0 lbs | 4.1-17.0 lbs |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 2.0-4.0 lbs | 4 | +0.001188 |
| 12 mg chewable tablet | 8 | +0.000711 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_74c95fac5a37018f` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | +0.006583 | +0.013427 |
| 仅主体分差 | +0.026738 | +0.042084 |
| 主体 token 数 | 98 | 75 |

[对照 token 数据](../photo_74c95fac5a37018f/tokens.csv) · [对照摘要](../photo_74c95fac5a37018f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
