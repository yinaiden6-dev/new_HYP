# photo_449ade96538a6b03

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_003298`，F03 错误参考 `reference_003301`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 9 |
| 图像 Top-1 参考 | reference_003301 |
| BM25 真值排名 | 8 |
| 融合真值排名 | 9 |
| 图像正负分差 | -0.008797 |
| BM25 正负分差 | -5.491394 |
| 0.5 × 图像 z 分差 | -0.109720 |
| 0.5 × BM25 z 分差 | -0.081125 |
| 最终融合分差 | -0.190845 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 256 个。正向贡献 +0.012389，负向贡献 -0.021186，相加等于图像分差 -0.008797。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 464 | -0.007651 | -0.012054 |
| nonvisual | 11 | -0.000241 | -0.015990 |
| subject_nontext | 67 | -0.000017 | -0.000181 |
| subject_text | 189 | -0.000888 | -0.003435 |

仅保留主体 query token 的两候选分差：-0.002584。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.007651。主体框内 token 占有效序列 35.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 242 | 68.6% | True | +8.740378 | 62.5% |
| 错误 | 238 | 66.3% | True | +8.665479 | 65.2% |

主体中 189 个 token 与 OCR 相交，152 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 91 | subject_text | -0.361135 | PORE-CLOGGING OILS |  |  |
| 179 | background | -0.222237 |  |  |  |
| 101 | background | -0.221571 |  |  | (see Directions), decreases the risk of skin cancer and |
| 180 | background | -0.191067 |  |  |  |
| 61 | background | -0.180217 |  | FULL OF FRESH, | CLEAN COVERAGE |
| 548 | background | -0.178810 |  |  |  |
| 300 | background | -0.177882 |  |  | (see Directions), decreases the risk of skin cancer and |
| 566 | background | -0.174845 |  | PORE-CLOGGING OILS |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 158 | background | +0.392804 |  |  | THAT'S CLEAN, FRESH AND |
| 289 | subject_text | +0.351225 | OPTHAMOLOGIST TESTED |  | OPTHAMOLOGIST TESTED |
| 393 | subject_text | +0.189021 | DE LA PEAU |  |  |
| 96 | subject_nontext | +0.124412 |  |  |  |
| 574 | subject_nontext | +0.111977 |  |  |  |
| 702 | background | +0.107050 |  |  |  |
| 254 | subject_nontext | +0.106580 |  | bareMinerals |  |
| 632 | subject_text | +0.099649 | TESTÉ SOUS CONTRÔLE |  | OPTHAMOLOGIST TESTED |

## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
