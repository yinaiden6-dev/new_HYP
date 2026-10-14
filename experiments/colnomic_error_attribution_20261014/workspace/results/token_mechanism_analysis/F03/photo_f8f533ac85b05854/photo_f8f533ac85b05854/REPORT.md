# photo_f8f533ac85b05854

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_003298`，F03 错误参考 `reference_003297`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003312`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 27 |
| 图像 Top-1 参考 | reference_003312 |
| BM25 真值排名 | 6 |
| 融合真值排名 | 6 |
| 图像正负分差 | -0.007754 |
| BM25 正负分差 | -11.053436 |
| 0.5 × 图像 z 分差 | -0.078499 |
| 0.5 × BM25 z 分差 | -0.072001 |
| 最终融合分差 | -0.150499 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 85 个。正向贡献 +0.025478，负向贡献 -0.033233，相加等于图像分差 -0.007754。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 635 | -0.005665 | -0.006521 |
| nonvisual | 11 | -0.000143 | -0.009531 |
| subject_nontext | 27 | -0.000744 | -0.020132 |
| subject_text | 58 | -0.001203 | -0.015160 |

仅保留主体 query token 的两候选分差：-0.016739。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.005665。主体框内 token 占有效序列 11.6%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 917 | 58.1% | True | +8.511767 | 75.3% |
| 错误 | 918 | 59.2% | True | +7.917737 | 64.7% |

主体中 58 个 token 与 OCR 相交，54 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 328 | background | -0.474764 |  |  | Do not use on damaged or broken skin |
| 533 | subject_text | -0.450163 | Other Information |  |  |
| 325 | background | -0.384367 |  |  | Do not use on damaged or broken skin |
| 217 | background | -0.378244 |  |  | Do not use on damaged or broken skin |
| 293 | subject_text | -0.346861 | (see Directions), decreeses the risk of skin cancer and | Purpose | Uses • helps prevent sunburn • if used as |
| 336 | subject_text | -0.324489 | When using this product keep out of eyes. Rinse with |  | Do not use on damaged or broken skin |
| 539 | background | -0.308414 |  |  |  |
| 282 | background | -0.304142 |  | Purpose | Uses • helps prevent sunburn • if used as |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 234 | background | +0.353164 |  |  |  |
| 235 | background | +0.349791 |  |  | reapply at least every 2 hours•children under |
| 369 | background | +0.322007 |  |  | rer |
| 58 | background | +0.292707 |  |  | reapply at least every 2 hours•children under |
| 55 | background | +0.283260 |  |  | reapply at least every 2 hours•children under |
| 236 | background | +0.255449 |  |  | rer |
| 718 | background | +0.252067 |  |  |  |
| 231 | background | +0.215014 |  |  |  |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Titanium diaxide 4.15% | 3 | -0.000278 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
