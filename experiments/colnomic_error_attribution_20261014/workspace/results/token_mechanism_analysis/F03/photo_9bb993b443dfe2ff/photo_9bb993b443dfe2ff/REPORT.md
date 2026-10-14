# photo_9bb993b443dfe2ff

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000935`，F03 错误参考 `reference_003797`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_003797 |
| BM25 真值排名 | 3 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.013901 |
| BM25 正负分差 | -13.974976 |
| 0.5 × 图像 z 分差 | -0.285485 |
| 0.5 × BM25 z 分差 | -0.539296 |
| 最终融合分差 | -0.824781 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 156 个。正向贡献 +0.026786，负向贡献 -0.040687，相加等于图像分差 -0.013901。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 564 | -0.011299 | -0.014644 |
| nonvisual | 11 | +0.000055 | +0.003651 |
| subject_nontext | 119 | -0.001003 | -0.006160 |
| subject_text | 37 | -0.001655 | -0.032692 |

仅保留主体 query token 的两候选分差：-0.012453。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.011299。主体框内 token 占有效序列 21.3%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 125 | 22.6% | True | +8.109859 | 71.7% |
| 错误 | 55 | 3.8% | False | 不可评估 | 不可评估 |

主体中 37 个 token 与 OCR 相交，24 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 351 | subject_text | -0.599109 | This product is not manufactured or distributed | Drug Facts (continued) |  |
| 350 | subject_text | -0.549424 | This product is not manufactured or distributed | "This product is not manufactured or distributed |  |
| 231 | background | -0.474546 |  | Drug Facts (continued) |  |
| 230 | background | -0.447269 |  | hypromellose, isopropyl alcoho, microcrystaline |  |
| 498 | subject_nontext | -0.370986 |  | Drug Facts (continued) | 200 WILMOT RD. DEERELD, IL 60015 |
| 392 | subject_nontext | -0.369024 |  | call 1-877-770-3183 Mon-Fri 8:00 AM EST to 5:00 PM PST | Walgreens Pharmacist Survey |
| 171 | background | -0.367987 |  |  |  |
| 612 | background | -0.341108 |  |  | 200 WILMOT RD. DEERELD, IL 60015 |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 399 | subject_text | +0.506291 | walgreens.com ©2020 Walgreen Co. | ITEM 696611W10383-0420-L | temporarilly rellieves minor aches and pains due to: |
| 599 | background | +0.455624 |  | ITEM 696611W10383-0420-L |  |
| 401 | background | +0.444768 |  | ITEM 696611W10383-0420-L |  |
| 600 | background | +0.425222 |  | ITEM 696611W10383-0420-L |  |
| 373 | subject_text | +0.425036 | Tylenol• and Extra Strength Tylenol• PM. | Compare to the active ingredlents |  |
| 659 | background | +0.412877 |  | ITEM 696611W10383-0420-L |  |
| 254 | background | +0.320946 |  | Reliever PM | Overdose warning: In case of overdose, get medical hellp or |
| 589 | background | +0.302481 |  | Exp. | Exp. |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% | 3 | +0.000155 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_77b67ea38a19f6b3` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013901 | +0.079191 |
| 仅主体分差 | -0.012453 | +0.105753 |
| 主体 token 数 | 156 | 143 |

[对照 token 数据](../photo_77b67ea38a19f6b3/tokens.csv) · [对照摘要](../photo_77b67ea38a19f6b3/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
