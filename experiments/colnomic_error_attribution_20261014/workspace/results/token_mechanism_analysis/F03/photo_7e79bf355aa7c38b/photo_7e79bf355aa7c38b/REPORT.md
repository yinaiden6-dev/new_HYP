# photo_7e79bf355aa7c38b

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000820`，F03 错误参考 `reference_000797`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003367`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 1574 |
| 图像 Top-1 参考 | reference_003367 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 19 |
| 图像正负分差 | -0.041130 |
| BM25 正负分差 | -26.893433 |
| 0.5 × 图像 z 分差 | -0.783949 |
| 0.5 × BM25 z 分差 | -0.308571 |
| 最终融合分差 | -1.092521 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 180 个。正向贡献 +0.035055，负向贡献 -0.076185，相加等于图像分差 -0.041130。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 588 | -0.046107 | -0.061084 |
| nonvisual | 11 | -0.000058 | -0.004140 |
| subject_nontext | 48 | -0.000280 | -0.004543 |
| subject_text | 132 | +0.005315 | +0.031369 |

仅保留主体 query token 的两候选分差：+0.021792。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.046107。主体框内 token 占有效序列 23.1%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 50 | 56.0% | True | +9.219704 | 78.2% |
| 错误 | 5 | 7.1% | False | 不可评估 | 不可评估 |

主体中 132 个 token 与 OCR 相交，77 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 663 | background | -0.572299 |  | Questlons?1-888-823-7837 |  |
| 737 | background | -0.563592 |  |  |  |
| 670 | background | -0.550331 |  |  |  |
| 541 | subject_nontext | -0.550129 |  | Questlons?1-888-823-7837 |  |
| 671 | background | -0.545906 |  |  |  |
| 515 | background | -0.543326 | azole 15 | contact a Polson Control Center Immedlately. |  |
| 294 | background | -0.537149 | www.SimilasanUSA.com |  | www.SimilasanUSA.com |
| 406 | subject_nontext | -0.534465 |  | Neck & Chest Treatment/Traitement du Cou et de la |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 472 | subject_text | +0.437691 | Stop use nd aak a doclof lf rasn co |  |  |
| 276 | subject_text | +0.399227 | Uses - Hetps prevent sun bum. • I used as drected with other sun |  | physicians who were open to homeopathic |
| 497 | subject_text | +0.393377 | rects the appearance of uneven skin tone & sun dannge. |  | decrease the ear's sensitivity to |
| 316 | subject_nontext | +0.376962 |  |  | Kids Ear |
| 210 | subject_nontext | +0.376639 |  |  |  |
| 15 | background | +0.365154 |  |  | Corporation, Highlands Ranch, CO 80129 |
| 239 | subject_text | +0.348286 | This whipped.lightwright moisturizing therapy visibly |  | Kids Ear |
| 220 | subject_nontext | +0.333462 |  |  | Kids Ear |

## 成功对照

`photo_41892aea27698341` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.041130 | +0.118593 |
| 仅主体分差 | +0.021792 | +0.258581 |
| 主体 token 数 | 180 | 238 |

[对照 token 数据](../photo_41892aea27698341/tokens.csv) · [对照摘要](../photo_41892aea27698341/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
