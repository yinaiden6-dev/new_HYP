# B03 · photo_65de2e175e6cbd9d

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_004066`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 5 |
| 正确参考 MaxSim | +0.599447 |
| 错误参考 MaxSim | +0.638841 |
| 正确减错误的分差 | -0.039394 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 119 个。正向贡献 +0.024825，负向贡献 -0.064219，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 601 | -0.034220 | -0.041622 |
| nonvisual | 11 | -0.000407 | -0.027074 |
| subject_nontext | 47 | -0.003328 | -0.051761 |
| subject_text | 72 | -0.001439 | -0.014610 |

最主要负差额来自 `background`，贡献 -0.034220。只保留主体 query token 后，本对分差为 -0.029283。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 168 | 54.1% | True | +4.350083 | 58.8% |
| 错误 | 6 | 9.6% | False | 不可评估 | 不可评估 |

主体中 72 个 token 与 OCR 行相交，53 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 259 | background | -0.369829 |  |  | The Procter & Gamble Company, owner of the |
| 402 | background | -0.358034 |  |  |  |
| 295 | subject_text | -0.345493 | by Johnson & Johnson Consumer Inc., owner of |  | The Procter & Gamble Company, owner of the |
| 393 | subject_text | -0.344846 | MADEIN INDIA |  |  |
| 624 | background | -0.343000 |  | "This product is not manufactured or distributed |  |
| 467 | background | -0.333852 |  | GUARANTEED |  |
| 34 | background | -0.332873 |  | ITEM 696611W10383-0420-L |  |
| 573 | background | -0.324629 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 235 | subject_text | +0.520160 | This product is not manufactured or distributed | "This product is not manufactured or distributed |  |
| 109 | background | +0.447354 |  | "This product is not manufactured or distributed | Sevem Cald & Flu Rellief LiquiCaps |
| 473 | subject_text | +0.440217 | ASPIRIN FREE | ITEM 696611W10383-0420-L | Doxylamire siccinate6.25 mg... |
| 106 | background | +0.401060 |  | "This product is not manufactured or distributed | Sevem Cald & Flu Rellief LiquiCaps |
| 667 | background | +0.368463 |  | ITEM 696611W10383-0420-L | Drug Facts (continued) |
| 483 | background | +0.364168 |  | ITEM 696611W10383-0420-L | Drug Facts (continued) |
| 663 | background | +0.343432 |  | ITEM 696611W10383-0420-L | Drug Facts (continued) |
| 660 | background | +0.324729 |  | ITEM 696611W10383-0420-L | Drug Facts (continued) |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| 100% | 8 | -0.001058 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7f4bc7d166b40bfc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.039394 | +0.225503 |
| 只保留主体分差 | -0.029283 | +0.266018 |
| 主体 token 数 | 119 | 288 |

[成功对照 token 表](../photo_7f4bc7d166b40bfc/tokens.csv) · [对照摘要](../photo_7f4bc7d166b40bfc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
