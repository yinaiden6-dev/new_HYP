# photo_206a9b7d61884264

已有身份歧义标记，不能直接当作模型失败证据。 用户指定主题：尚未人工分类。

正确参考 `reference_003298`，F03 错误参考 `reference_003302`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003299`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 10 |
| 图像 Top-1 参考 | reference_003299 |
| BM25 真值排名 | 8 |
| 融合真值排名 | 10 |
| 图像正负分差 | -0.008121 |
| BM25 正负分差 | -7.289894 |
| 0.5 × 图像 z 分差 | -0.094102 |
| 0.5 × BM25 z 分差 | -0.473650 |
| 最终融合分差 | -0.567753 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 270 个。正向贡献 +0.017584，负向贡献 -0.025705，相加等于图像分差 -0.008121。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 450 | -0.004246 | -0.006898 |
| nonvisual | 11 | -0.000084 | -0.005601 |
| subject_nontext | 179 | -0.000676 | -0.002762 |
| subject_text | 91 | -0.003114 | -0.025013 |

仅保留主体 query token 的两候选分差：-0.010261。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.004246。主体框内 token 占有效序列 36.9%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 248 | 34.9% | True | +8.457510 | 67.3% |
| 错误 | 271 | 35.7% | True | +9.005948 | 67.2% |

主体中 91 个 token 与 OCR 相交，34 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 613 | subject_text | -0.328288 | poids net wt. |  | poids net wt. |
| 322 | background | -0.324999 |  | 10 g / 0.35 oz. |  |
| 142 | background | -0.313338 |  | 10 g / 0.35 oz. |  |
| 172 | subject_text | -0.220723 | bareMineral | Dist. by/par bareMinerals, New York | bareMineral |
| 13 | background | -0.209590 |  |  |  |
| 191 | subject_text | -0.202375 | bareMineral | Dist. by/par bareMinerals, New York | bareMineral |
| 150 | subject_text | -0.193909 | bareMineral | Dist. by/par bareMinerals, New York | bareMineral |
| 297 | subject_nontext | -0.192437 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | subject_nontext | +0.435852 |  |  |  |
| 357 | subject_nontext | +0.433781 |  |  |  |
| 438 | subject_nontext | +0.426713 |  |  |  |
| 500 | background | +0.329721 |  |  |  |
| 184 | background | +0.319556 |  |  |  |
| 536 | subject_nontext | +0.300707 |  |  |  |
| 337 | subject_nontext | +0.300246 |  |  |  |
| 183 | background | +0.293291 |  |  |  |

## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
