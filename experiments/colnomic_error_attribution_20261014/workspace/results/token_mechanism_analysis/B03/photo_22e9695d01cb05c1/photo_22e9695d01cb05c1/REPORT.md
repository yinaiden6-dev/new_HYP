# B03 · photo_22e9695d01cb05c1

按当前评测真值分析，未新增身份人工复核。

包装 `package_040`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_001112`，**纯图像实际 Top-1 错误参考** `reference_003832`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 20 |
| 正确参考 MaxSim | +0.472821 |
| 错误参考 MaxSim | +0.483247 |
| 正确减错误的分差 | -0.010427 |
| F03 的正确参考排名（辅助对照） | 14 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 170 个。正向贡献 +0.022354，负向贡献 -0.032781，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 598 | -0.007017 | -0.009141 |
| nonvisual | 11 | -0.000088 | -0.006264 |
| subject_nontext | 22 | -0.000410 | -0.014508 |
| subject_text | 148 | -0.002911 | -0.015323 |

最主要负差额来自 `background`，贡献 -0.007017。只保留主体 query token 后，本对分差为 -0.015218。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 327 | 66.6% | True | +6.644394 | 59.6% |
| 错误 | 25 | 14.3% | False | 不可评估 | 不可评估 |

主体中 148 个 token 与 OCR 行相交，96 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 305 | subject_text | -0.552179 | temporanly relieves these symptoms due to hay lever or |  |  |
| 533 | subject_text | -0.490683 | alcohol, sedatives, and tranquilizers may increase |  |  |
| 531 | subject_text | -0.417005 | alcohol, sedatives, and tranquilizers may increase |  |  |
| 534 | subject_text | -0.416371 | alcohol, sedatives, and tranquilizers may increase |  |  |
| 500 | subject_text | -0.376280 | When using this product |  |  |
| 594 | subject_text | -0.346223 | Stop use and ask a doctor il an allergic reaction to this | be careful when driving a motor vehicle or operating |  |
| 181 | subject_text | -0.334932 | TAMPER EVIDENT: DO NOT USE THIS PRODUCT IF | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |
| 404 | subject_text | -0.315737 | this product or any of its ingredients, or to an |  | to this product or any of its ingredients or to an |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 147 | subject_text | +0.535283 | READ AND KEEP CARTON FOR COMPLETE | READ AND KEEP CARTON FOR COMPLETE |  |
| 213 | subject_text | +0.486643 | THE BOTTLE IS CUT. TORN, BROKEN OR MISSING | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |
| 177 | subject_text | +0.471885 | TAMPER EVIDENT: DO NOT USE THIS PRODUCT IF | THE IMPRINTED FOIL SEAL OVER THE MOUTH OF |  |
| 183 | subject_text | +0.427828 | TAMPER EVIDENT: DO NOT USE THIS PRODUCT IF |  |  |
| 146 | subject_text | +0.408004 | READ AND KEEP CARTON FOR COMPLETE | READ AND KEEP CARTON FOR COMPLETE |  |
| 215 | subject_text | +0.401775 | THE BOTTLE IS CUT. TORN, BROKEN OR MISSING | THE BOTTLE IS CUT, TORN, BROKEN OR MISSING |  |
| 210 | subject_text | +0.385281 | THE BOTTLE IS CUT. TORN, BROKEN OR MISSING |  | one 10 mg capsule once |
| 145 | subject_text | +0.366991 | READ AND KEEP CARTON FOR COMPLETE | READ AND KEEP CARTON FOR COMPLETE |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Cetrizine HCI 10 mg. | 3 | -0.000130 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_2df687d8a2f89f40` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010427 | +0.033877 |
| 只保留主体分差 | -0.015218 | +0.064922 |
| 主体 token 数 | 170 | 312 |

[成功对照 token 表](../photo_2df687d8a2f89f40/tokens.csv) · [对照摘要](../photo_2df687d8a2f89f40/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
