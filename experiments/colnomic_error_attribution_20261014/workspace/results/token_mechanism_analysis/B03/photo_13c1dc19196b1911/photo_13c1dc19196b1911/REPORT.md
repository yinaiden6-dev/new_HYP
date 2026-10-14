# B03 · photo_13c1dc19196b1911

按当前评测真值分析，未新增身份人工复核。

包装 `package_016`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000935`，**纯图像实际 Top-1 错误参考** `reference_003096`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 34 |
| 正确参考 MaxSim | +0.666203 |
| 错误参考 MaxSim | +0.696398 |
| 正确减错误的分差 | -0.030195 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 342 个。正向贡献 +0.027140，负向贡献 -0.057335，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 378 | -0.013159 | -0.025448 |
| nonvisual | 11 | +0.000239 | +0.015880 |
| subject_nontext | 98 | -0.003087 | -0.023030 |
| subject_text | 244 | -0.014187 | -0.042504 |

最主要负差额来自 `subject_text`，贡献 -0.014187。只保留主体 query token 后，本对分差为 -0.036924。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 1093 | 67.9% | True | +7.730962 | 82.2% |
| 错误 | 4 | 0.7% | False | 不可评估 | 不可评估 |

主体中 244 个 token 与 OCR 行相交，204 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 493 | subject_text | -0.495361 | do not drive a motor vehide or operate machinery |  |  |
| 427 | subject_text | -0.486926 | a breathing problem such as emphysema or | Ask a doctor or pharmacist before use If you are | a breathing problem such as emphysema or chronic bronchitis |
| 320 | subject_text | -0.443825 | blistersrash | Allergy alert acetaminophen may causa severa | Allergy alert: Acetaminophen may cause severe skin reactions. |
| 428 | subject_text | -0.406236 | a breathing problem such as emphysema or | If pragnant or breest-feeding, ask a health | a breathing problem such as emphysema or chronic bronchitis |
| 235 | subject_text | -0.371428 | Liver warning: This product contains |  | Liver warning: This product contains acetaminophen. Severe liver |
| 311 | subject_text | -0.369535 | rash | Allergy alert acetaminophen may causa severa | Allergy alert: Acetaminophen may cause severe skin reactions. |
| 530 | subject_text | -0.348221 | pain gets worse or lasts more than 10 days |  | pain gets worse or lasts more than 10 days |
| 514 | subject_nontext | -0.345427 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 611 | background | +0.331983 |  | a breathing problem such as emphysema or | Ask a doctor before use if you have |
| 250 | subject_text | +0.277517 | acataminophen. Severe lIiver damage may occur If | acetaminophen. Severe liver damage may occur | Liver warning: This product contains acetaminophen. Severe liver |
| 238 | subject_text | +0.258894 | Liver warning: This product contains | acetaminophen. Severe liver damage may occur | Liver warning: This product contains acetaminophen. Severe liver |
| 150 | background | +0.257507 |  | 3 or more alcohollc drinks every day while using | Reliever |
| 279 | subject_text | +0.246921 | 3 or more alcohosic drinks every day while | 3 or more alcohollc drinks every day while using | Reliever |
| 534 | subject_nontext | +0.245960 |  | Reliever PM | Reliever |
| 459 | subject_text | +0.238394 | These could be signs of a serlous condition. |  | a breathing problem such as emphysema or chronic bronchitis |
| 111 | background | +0.234103 |  | a breathing problem such as emphysema or |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| more than 4,000 mg of acetaminophen in 24 | 8 | -0.000120 |
| more than 4,000 mg of acetaminophen in 24 hours | 9 | +0.000178 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_7f4bc7d166b40bfc` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.030195 | +0.062756 |
| 只保留主体分差 | -0.036924 | +0.072080 |
| 主体 token 数 | 342 | 288 |

[成功对照 token 表](../photo_7f4bc7d166b40bfc/tokens.csv) · [对照摘要](../photo_7f4bc7d166b40bfc/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
