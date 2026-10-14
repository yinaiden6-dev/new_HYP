# photo_c576c0c30f74cd4f

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_003298`，F03 错误参考 `reference_003299`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003300`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 10 |
| 图像 Top-1 参考 | reference_003300 |
| BM25 真值排名 | 8 |
| 融合真值排名 | 10 |
| 图像正负分差 | -0.008658 |
| BM25 正负分差 | -8.549927 |
| 0.5 × 图像 z 分差 | -0.109326 |
| 0.5 × BM25 z 分差 | -0.127634 |
| 最终融合分差 | -0.236960 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 85 个。正向贡献 +0.010271，负向贡献 -0.018929，相加等于图像分差 -0.008658。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 635 | -0.006376 | -0.007340 |
| nonvisual | 11 | -0.000160 | -0.010624 |
| subject_nontext | 22 | -0.000801 | -0.026620 |
| subject_text | 63 | -0.001321 | -0.015327 |

仅保留主体 query token 的两候选分差：-0.018250。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.006376。主体框内 token 占有效序列 11.6%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 486 | 56.0% | True | +6.548046 | 67.1% |
| 错误 | 496 | 56.0% | True | +6.531611 | 63.5% |

主体中 63 个 token 与 OCR 相交，53 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 172 | background | -0.369001 |  |  |  |
| 292 | subject_text | -0.200502 | PORE-CLOGGING OILS |  |  |
| 451 | subject_text | -0.174465 | PRÉSENTÉ SOUS FORME DE STICK | PRÉSENTÉ SOUS FORME DE STICK | OFFRE LES BIENFAITS D'UN |
| 591 | subject_nontext | -0.173151 |  |  |  |
| 28 | background | -0.162746 |  |  |  |
| 532 | subject_text | -0.148493 | HYPOALLERGÉNIQUE | HYPOALLERGÉNIQUE | HYPOALLERGÉNIQUE |
| 535 | subject_nontext | -0.140849 |  |  |  |
| 595 | subject_nontext | -0.140388 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 109 | background | +0.138657 |  |  |  |
| 191 | background | +0.128226 |  |  | coco-caprylate/caprate, butylene glycol, |
| 138 | background | +0.112980 |  |  |  |
| 103 | background | +0.107850 |  |  |  |
| 422 | background | +0.098360 |  |  | CE FOND DE TEINT LIQUIDE |
| 428 | background | +0.092679 |  |  | CE FOND DE TEINT LIQUIDE |
| 244 | background | +0.088101 |  |  | bareMinerals |
| 602 | background | +0.084653 |  | hydroxide, disteardimonium hectorite, microcrystalline | SANS COMPROMIS |

## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
