# B03 · photo_158cee75554e9c49

按当前评测真值分析，未新增身份人工复核。

包装 `package_080`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003367`，**纯图像实际 Top-1 错误参考** `reference_004289`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.535541 |
| 错误参考 MaxSim | +0.544405 |
| 正确减错误的分差 | -0.008864 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 300 个。正向贡献 +0.024036，负向贡献 -0.032900，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 468 | -0.009666 | -0.016089 |
| nonvisual | 11 | +0.000145 | +0.010251 |
| subject_nontext | 177 | -0.002762 | -0.012156 |
| subject_text | 123 | +0.003420 | +0.021657 |

最主要负差额来自 `background`，贡献 -0.009666。只保留主体 query token 后，本对分差为 +0.001707。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 168 | 34.9% | True | +9.283423 | 78.6% |
| 错误 | 17 | 14.0% | False | 不可评估 | 不可评估 |

主体中 123 个 token 与 OCR 行相交，70 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 147 | background | -0.274805 |  |  | UNIT IS TORN, BROKEN OR SHOWS ANY SIGNS OF TAMPERING |
| 78 | background | -0.250082 |  |  | alcohol, sedatives, and tranquilizers may increase drowsiness |
| 478 | subject_nontext | -0.246963 |  | ASAL L |  |
| 579 | background | -0.241280 |  |  |  |
| 52 | background | -0.241070 |  |  |  |
| 269 | subject_nontext | -0.238223 |  |  | Diphenhydramine HCI 25 mg |
| 47 | background | -0.227324 |  |  | Diphenhydramine HCI 25 mg |
| 115 | background | -0.226256 | Scan to get |  | UNIT IS TORN, BROKEN OR SHOWS ANY SIGNS OF TAMPERING |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 271 | subject_nontext | +0.396628 |  | povidone, purified water, sodium |  |
| 318 | subject_text | +0.378908 | cvs Quality | Money Back Guarantee | Money Back Guarantee |
| 463 | subject_text | +0.354376 | phosphate dibasic, sodium | povidone, purified water, sodium | itching of the nose or throat (Nighttime only) |
| 494 | subject_text | +0.341280 | and carboxymethylcellulose | flavor, microcrystalline cellulose | sodium carboxymethyfcellulose, sodium citrate dihydrate, |
| 751 | background | +0.341001 |  | povidone, purified water, sodium |  |
| 497 | subject_text | +0.337777 | *CVS/pharmacy® Nasal Mist is not | *CVS/pharmacy® Nasal Mist is not | use caution when driving a motor vehicle or operating |
| 501 | subject_nontext | +0.335198 |  | retain carton for future |  |
| 108 | background | +0.316265 |  | retain carton for future |  |

## 同包装成功对照

`photo_587142eeed6488be` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008864 | +0.081890 |
| 只保留主体分差 | +0.001707 | +0.143552 |
| 主体 token 数 | 300 | 276 |

[成功对照 token 表](../photo_587142eeed6488be/tokens.csv) · [对照摘要](../photo_587142eeed6488be/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
