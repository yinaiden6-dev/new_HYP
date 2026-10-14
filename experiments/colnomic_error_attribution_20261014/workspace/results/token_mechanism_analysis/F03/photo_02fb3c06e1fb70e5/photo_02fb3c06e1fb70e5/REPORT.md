# photo_02fb3c06e1fb70e5

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001129`，F03 错误参考 `reference_003367`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 189 |
| 图像 Top-1 参考 | reference_003367 |
| BM25 真值排名 | 52 |
| 融合真值排名 | 48 |
| 图像正负分差 | -0.087715 |
| BM25 正负分差 | -88.262589 |
| 0.5 × 图像 z 分差 | -2.169254 |
| 0.5 × BM25 z 分差 | -2.277582 |
| 最终融合分差 | -4.446836 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 209 个。正向贡献 +0.044109，负向贡献 -0.131825，相加等于图像分差 -0.087715。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 559 | -0.102431 | -0.142744 |
| nonvisual | 11 | -0.001600 | -0.113339 |
| subject_nontext | 164 | +0.007518 | +0.035711 |
| subject_text | 45 | +0.008798 | +0.152307 |

仅保留主体 query token 的两候选分差：+0.060815。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.102431。主体框内 token 占有效序列 26.8%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 97 | 33.3% | True | +11.146492 | 79.2% |
| 错误 | 5 | 21.2% | False | 不可评估 | 不可评估 |

主体中 45 个 token 与 OCR 相交，21 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 627 | background | -0.643976 |  |  | congestion relief |
| 713 | background | -0.627520 |  | 094 09 0036 R00 C-002262-01-008-0000 | 1 FL OZ (30 mL) |
| 619 | background | -0.606379 | congestion relief | Usereduces withdrawal symptoms, including nicotine | congestion relief |
| 600 | background | -0.597234 |  |  | 7 |
| 696 | background | -0.592110 |  | Lozenge | 7 |
| 555 | background | -0.590414 | 12 HOUR PUMP MIST |  |  |
| 631 | background | -0.590299 |  |  | 7 |
| 455 | background | -0.589283 | SAL L |  | NASAL MIST |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 399 | subject_text | +0.553704 | source where proof of age cannet be verified | source where proof of age cannot be verified | Shake well before use. Before using the first time, |
| 408 | subject_nontext | +0.527680 |  | source where proof of age cannot be verified |  |
| 439 | subject_nontext | +0.508045 |  | source where proof of age cannot be verified | SAFETY SEALED: Do not use If Imprinted seal on bottle |
| 461 | subject_text | +0.493667 | not for sale to those under 18 years of age | source where proof of age cannot be verified | No Drip |
| 494 | subject_text | +0.491673 | proof of age required |  | 1-800-shop-CVS |
| 664 | background | +0.483553 |  | source where proof of age cannot be verified | *CVS/pharmacy® Nasal Mist is not |
| 493 | subject_text | +0.479061 | not for sale to those under 18 years of age | source where proof of age cannot be verified | ASAL L |
| 573 | background | +0.469639 |  | ACTUAL SIZE | 050428588352 |

## 成功对照

`photo_80eb2d50ed90738e` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.087715 | +0.132694 |
| 仅主体分差 | +0.060815 | +0.260755 |
| 主体 token 数 | 209 | 273 |

[对照 token 数据](../photo_80eb2d50ed90738e/tokens.csv) · [对照摘要](../photo_80eb2d50ed90738e/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
