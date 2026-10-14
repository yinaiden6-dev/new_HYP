# B03 · photo_3fa448ba98849bb7

按当前评测真值分析，未新增身份人工复核。

包装 `package_026`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000995`，**纯图像实际 Top-1 错误参考** `reference_000852`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.629874 |
| 错误参考 MaxSim | +0.640777 |
| 正确减错误的分差 | -0.010903 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 156 个。正向贡献 +0.020851，负向贡献 -0.031754，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 612 | -0.007233 | -0.009207 |
| nonvisual | 11 | +0.000159 | +0.011226 |
| subject_nontext | 86 | -0.001886 | -0.017081 |
| subject_text | 70 | -0.001942 | -0.021615 |

最主要负差额来自 `background`，贡献 -0.007233。只保留主体 query token 后，本对分差为 -0.019116。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 130 | 37.5% | True | +6.302050 | 67.9% |
| 错误 | 78 | 6.4% | False | 不可评估 | 不可评估 |

主体中 70 个 token 与 OCR 行相交，52 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 436 | subject_text | -0.388498 | Do not lie flat or bend over soon after eating | Do not eat late at night or just before bedtime. | Tips for Managing Heartburn |
| 756 | background | -0.352958 |  | Do not eat late at night or just before bedtime. | Tips for Managing Heartburn |
| 371 | subject_text | -0.306128 | Eat slowly and do not eat big meals. |  | Eat slowly and do not eat big |
| 345 | subject_nontext | -0.273341 |  | BOX |  |
| 248 | subject_text | -0.256352 | how2recyclainfo | be caused by a serious condition. | how2recycle,info |
| 153 | background | -0.244219 |  | BOX |  |
| 355 | background | -0.238046 |  | Do not eat late at night or just before bedtime. | chocolate, caffeine, alcohol, even |
| 376 | subject_text | -0.235191 | PAPER | BOX |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 17 | background | +0.300511 |  | Tips for Managing Heartburn | Cimetidine |
| 501 | subject_text | +0.258522 | Wear loose-fitting clothing around | • Raise the head of your bed |  |
| 319 | background | +0.224954 |  |  |  |
| 305 | subject_text | +0.219203 | Avoid foods or drinks that are more likely to |  |  |
| 731 | background | +0.195453 |  | Tips for Managing Heartburn | Do not eat late at night, or just |
| 43 | background | +0.185850 |  | Tips for Managing Heartburn |  |
| 21 | background | +0.182934 |  | • Raise the head of your bed | CIMETIDINE TABLETS USP, 200 mg / |
| 302 | subject_nontext | +0.180730 |  |  |  |

## 同包装成功对照

`photo_cf724bd3f6c03346` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010903 | +0.100736 |
| 只保留主体分差 | -0.019116 | +0.178242 |
| 主体 token 数 | 156 | 144 |

[成功对照 token 表](../photo_cf724bd3f6c03346/tokens.csv) · [对照摘要](../photo_cf724bd3f6c03346/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
