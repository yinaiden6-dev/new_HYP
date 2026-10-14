# photo_438026ead38bb48f

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_001023`，F03 错误参考 `reference_001326`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 3 |
| 图像 Top-1 参考 | reference_001326 |
| BM25 真值排名 | 7 |
| 融合真值排名 | 6 |
| 图像正负分差 | -0.034037 |
| BM25 正负分差 | -19.918098 |
| 0.5 × 图像 z 分差 | -0.761183 |
| 0.5 × BM25 z 分差 | -1.161558 |
| 最终融合分差 | -1.922740 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 99 个。正向贡献 +0.024268，负向贡献 -0.058305，相加等于图像分差 -0.034037。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 621 | -0.029677 | -0.034933 |
| nonvisual | 11 | -0.000042 | -0.002767 |
| subject_nontext | 69 | -0.001171 | -0.012403 |
| subject_text | 30 | -0.003148 | -0.076702 |

仅保留主体 query token 的两候选分差：-0.031887。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.029677。主体框内 token 占有效序列 13.5%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 72 | 23.8% | True | +4.101495 | 58.6% |
| 错误 | 34 | 15.7% | True | +10.358634 | 85.9% |

主体中 30 个 token 与 OCR 相交，11 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 53 | background | -0.527683 |  |  |  |
| 232 | background | -0.524581 |  | MADE IN MEXICO |  |
| 295 | subject_text | -0.513394 | Bayer, the Bayer Cross and Alka-Seltzer Plus | Bayer |  |
| 353 | subject_nontext | -0.481168 |  |  |  |
| 242 | background | -0.462405 |  |  |  |
| 233 | background | -0.461048 |  | MADE IN MEXICO |  |
| 352 | subject_text | -0.449903 | MADE IN MEXICO |  |  |
| 666 | background | -0.443481 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 253 | subject_nontext | +0.453919 |  | 20 EFFERVESCENT TABLETS | Whippany, NJ 07981 |
| 266 | background | +0.329091 |  | 20 EFFERVESCENT TABLETS | PARENTS |
| 143 | background | +0.328568 |  | 20 EFFERVESCENT TABLETS |  |
| 634 | background | +0.325083 |  | 20 EFFERVESCENT TABLETS |  |
| 434 | background | +0.305309 |  | 20 EFFERVESCENT TABLETS |  |
| 114 | background | +0.304237 |  | 20 EFFERVESCENT TABLETS | Whippany, NJ 07981 |
| 633 | background | +0.295281 |  | 20 EFFERVESCENT TABLETS | PARENTS |
| 291 | subject_nontext | +0.242400 |  |  | Whippany, NJ 07981 |

## 成功对照

本包装没有图像和融合同时 Top-1 正确的照片，缺少同实例成功对照。


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
