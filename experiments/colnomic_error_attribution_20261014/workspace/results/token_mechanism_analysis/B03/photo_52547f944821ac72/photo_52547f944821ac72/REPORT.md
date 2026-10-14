# B03 · photo_52547f944821ac72

按当前评测真值分析，未新增身份人工复核。

包装 `package_040`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001112`，**纯图像实际 Top-1 错误参考** `reference_003821`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 15 |
| 正确参考 MaxSim | +0.614498 |
| 错误参考 MaxSim | +0.652463 |
| 正确减错误的分差 | -0.037965 |
| F03 的正确参考排名（辅助对照） | 3 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 299 个。正向贡献 +0.017318，负向贡献 -0.055282，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 469 | -0.018045 | -0.029972 |
| nonvisual | 11 | -0.000114 | -0.008064 |
| subject_nontext | 68 | -0.003494 | -0.040022 |
| subject_text | 231 | -0.016312 | -0.055009 |

最主要负差额来自 `background`，贡献 -0.018045。只保留主体 query token 后，本对分差为 -0.051601。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 544 | 66.7% | True | +4.903694 | 59.3% |
| 错误 | 51 | 22.8% | True | +4.401301 | 58.6% |

主体中 231 个 token 与 OCR 行相交，138 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 464 | subject_text | -0.472733 | temporarily relieves these symptoms due to hay fever or | Keep out of reach of children. In case of overdose, get |  |
| 259 | background | -0.446590 |  |  |  |
| 440 | subject_text | -0.423346 | tranquilizers or sedatives. |  | drowsiness may occur |
| 279 | subject_text | -0.413771 | disease. Your doctor should determine if you need a | READ AND KEEP CARTON FOR COMPLETE | disease. Your doctor should determine if you need a |
| 474 | subject_text | -0.383227 | alcohol, sedatives, and tranquilizers may increase |  | drowsiness may occur |
| 139 | background | -0.380600 |  | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |
| 311 | subject_text | -0.373129 | disease. Your doctor should determine if you need a | Ask a doctor or pharmacist before use if you are taking | disease. Your doctor should determine if you need a |
| 378 | subject_text | -0.367356 | drowsiness may occur avoid alcoholic drinks | When using this product |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 235 | subject_text | +0.489189 | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |  |
| 203 | subject_text | +0.447989 | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |  |
| 200 | subject_nontext | +0.418958 |  | READ AND KEEP CARTON FOR COMPLETE |  |
| 331 | subject_text | +0.312570 | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |  |
| 555 | subject_nontext | +0.271898 |  | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |
| 241 | subject_text | +0.271552 | temporarily relieves these symptoms due to hay fever or | Antihistamine | Temporarily relieves these symptoms due to hay fever or |
| 459 | subject_text | +0.267375 | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  | SAFETY SEAL UNDER CAP IS BROKEN OR MISSING. |
| 392 | subject_text | +0.248473 | READ AND KEEP CARTON FOR COMPLETE | READ AND KEEP CARTON FOR COMPLETE |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Cetirizine HCl 10 mg. | 7 | -0.000566 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_2df687d8a2f89f40` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.037965 | +0.023698 |
| 只保留主体分差 | -0.051601 | +0.041633 |
| 主体 token 数 | 299 | 312 |

[成功对照 token 表](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
