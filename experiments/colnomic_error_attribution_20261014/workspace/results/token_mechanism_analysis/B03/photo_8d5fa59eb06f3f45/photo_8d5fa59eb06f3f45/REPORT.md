# B03 · photo_8d5fa59eb06f3f45

按当前评测真值分析，未新增身份人工复核。

包装 `package_113`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000157`，**纯图像实际 Top-1 错误参考** `reference_000506`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.640788 |
| 错误参考 MaxSim | +0.655241 |
| 正确减错误的分差 | -0.014453 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 230 个。正向贡献 +0.049157，负向贡献 -0.063609，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 490 | +0.000512 | +0.000763 |
| nonvisual | 11 | +0.000068 | +0.004547 |
| subject_nontext | 74 | -0.006299 | -0.062225 |
| subject_text | 156 | -0.008734 | -0.040926 |

最主要负差额来自 `subject_text`，贡献 -0.008734。只保留主体 query token 后，本对分差为 -0.047779。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 843 | 58.9% | True | +8.291693 | 74.0% |
| 错误 | 8 | 28.8% | True | +8.768950 | 84.6% |

主体中 156 个 token 与 OCR 行相交，127 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 556 | subject_text | -0.557116 | Distributed by: MWI | V1 504020 |  |
| 358 | subject_nontext | -0.514759 |  | injection site (1-2 mL per site in small calves) |  |
| 518 | subject_nontext | -0.497749 |  |  |  |
| 356 | subject_text | -0.494466 | treatment must not be used for food Rapid | injection site (1-2 mL per site in small calves) |  |
| 362 | background | -0.491227 |  |  | Rapid intravenous administration may result in |
| 536 | subject_text | -0.487788 | Distributed by: MWI |  |  |
| 311 | subject_text | -0.485375 | the dressing procedure |  | surrounding tissues during the dressing procedure. |
| 394 | subject_text | -0.470696 | Exceeding |  | Exceeding the highest recommended level of drug per |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 634 | subject_text | +0.553670 | NoCoating | NoCoating | STORAGE CONDITIONS: |
| 593 | subject_text | +0.513769 | EXP. DATE: | EXP. DATE: |  |
| 635 | subject_text | +0.513056 | NoCoating | Coating |  |
| 623 | background | +0.482380 |  | NoCoating |  |
| 613 | subject_text | +0.477014 | EXP. DATE: | EXP. DATE: |  |
| 531 | subject_text | +0.468324 | TAKETIME | TAKE TIME |  |
| 534 | subject_text | +0.439590 | Distributed by: MWI | Distributed by: MWI |  |
| 610 | subject_text | +0.438631 | LOTNO | LOT NO. |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| antibiotic oxytetracycline Each mL contains 200 mg | 16 | +0.000775 |
| and on a w/u basis, 40.0% 2- pyrrolidone, 5.0% | 14 | +0.001990 |
| poridone, 1.8% magnesium oxide, 02% sodium | 10 | +0.001663 |
| exceeding 10 mL subcutaneously per injection site in | 12 | -0.001176 |
| adult beel and dairy cattle, and 5 mL intramuscularly | 9 | +0.000132 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_63103452c5081a78` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.014453 | +0.028276 |
| 只保留主体分差 | -0.047779 | +0.035520 |
| 主体 token 数 | 230 | 230 |

[成功对照 token 表](../photo_63103452c5081a78/tokens.csv) · [对照摘要](../photo_63103452c5081a78/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
