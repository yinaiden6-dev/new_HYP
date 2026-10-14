# B03 · photo_7c0c3a5cb514ea44

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_111`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000512`，**纯图像实际 Top-1 错误参考** `reference_000507`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.669963 |
| 错误参考 MaxSim | +0.672296 |
| 正确减错误的分差 | -0.002333 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 114 个。正向贡献 +0.019566，负向贡献 -0.021900，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 606 | -0.003639 | -0.004390 |
| nonvisual | 11 | +0.000074 | +0.004900 |
| subject_nontext | 75 | +0.000291 | +0.002833 |
| subject_text | 39 | +0.000941 | +0.017647 |

最主要负差额来自 `background`，贡献 -0.003639。只保留主体 query token 后，本对分差为 +0.007901。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 201 | 65.6% | True | +2.517043 | 46.0% |
| 错误 | 133 | 49.8% | True | +2.240394 | 43.9% |

主体中 39 个 token 与 OCR 行相交，19 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 388 | subject_text | -0.285960 | 7015 College Boulevard, Sulte 525 | Overland Park, KS 66211 USA | Overland Park, KS 66211 USA |
| 78 | background | -0.216679 |  | Approved by FDA under NADA # 141-291 | Overland Park, KS 66211 USA |
| 588 | background | -0.206914 |  | Overland Park, KS 66211 USA | Overland Park, KS 66211 USA |
| 586 | background | -0.194836 |  | Overland Park, KS 66211 USA |  |
| 219 | background | -0.167746 |  | 317033-112-30 | 317033-105-30 |
| 339 | background | -0.167094 |  | 317033-112-30 | 317033-105-30 |
| 585 | background | -0.154359 |  | Overland Park, KS 66211 USA | Overland Park, KS 66211 USA |
| 697 | background | -0.149373 |  |  | blood vo lume and reral perfusion. Potass ium sparing diureti cs (e. g s pirorolactone) s fould rot be us ed with VETORYL |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 660 | background | +0.408553 |  | 317033-112-30 | 317033-105-30 |
| 341 | background | +0.336745 |  | 317033-112-30 |  |
| 460 | subject_text | +0.307832 | 17033-112-30 | 317033-112-30 | 317033-105-30 |
| 395 | subject_nontext | +0.271609 |  | Overland Park, KS 66211 USA |  |
| 406 | subject_text | +0.268991 | (866)933-2472 | Approved by FDA under NADA # 141-291 | 317033-105-30 |
| 407 | subject_text | +0.231978 | (866)933-2472 | Overland Park, KS 66211 USA |  |
| 287 | background | +0.228650 |  | Approved by FDA under NADA # 141-291 | 317033-105-30 |
| 275 | background | +0.222507 |  | Overland Park, KS 66211 USA |  |

## 同包装成功对照

`photo_1b6a28535da238b9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002333 | +0.010292 |
| 只保留主体分差 | +0.007901 | +0.015664 |
| 主体 token 数 | 114 | 66 |

[成功对照 token 表](../photo_1b6a28535da238b9/tokens.csv) · [对照摘要](../photo_1b6a28535da238b9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
