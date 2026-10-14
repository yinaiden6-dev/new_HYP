# B03 · photo_6f1045004ede2c6e

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_079`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003333`，**纯图像实际 Top-1 错误参考** `reference_002079`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 11 |
| 正确参考 MaxSim | +0.690709 |
| 错误参考 MaxSim | +0.704634 |
| 正确减错误的分差 | -0.013925 |
| F03 的正确参考排名（辅助对照） | 2 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 300 个。正向贡献 +0.036818，负向贡献 -0.050744，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 420 | -0.006702 | -0.011665 |
| nonvisual | 11 | -0.000371 | -0.024626 |
| subject_nontext | 65 | -0.000704 | -0.007917 |
| subject_text | 235 | -0.006149 | -0.019127 |

最主要负差额来自 `background`，贡献 -0.006702。只保留主体 query token 后，本对分差为 -0.016698。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 619 | 65.9% | True | +4.666683 | 54.3% |
| 错误 | 6 | 6.5% | False | 不可评估 | 不可评估 |

主体中 235 个 token 与 OCR 行相交，161 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 315 | subject_text | -0.552230 | medical help or contact a Poison Control Center right away |  | swallowed, get medical help or |
| 307 | background | -0.524235 |  |  | Keep out of reach of |
| 337 | subject_nontext | -0.522227 |  |  | apply liberally 15 minutes |
| 399 | subject_text | -0.510240 | time in the sun increases your risk of sun cancer and | wear long-sleeved shirt, pants, hats and sunglasses | To decrease this risk, |
| 303 | background | -0.491799 |  |  | Keep out of reach of |
| 98 | subject_nontext | -0.472208 |  |  | To decrease this risk, |
| 378 | subject_text | -0.470771 | use a water resistant sunscreen if swimming or |  | sunscreen if swimming or |
| 440 | subject_text | -0.469572 | limit time in the sun, especially from 10 a.m-2 p.m | before sun exposure • reapply at least every 2 hours | 10 a.m. - 2 p.m. |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 613 | subject_text | +0.554772 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 | children under 6 months: Ask |
| 614 | subject_text | +0.496791 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 |  |
| 37 | subject_nontext | +0.459864 |  |  |  |
| 573 | subject_text | +0.422410 | Glycol Fragrance (Parfum) May Contain: Iron Oxides (C77491. | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (Ci 77491, CI | Carica Papaya Extract, |
| 574 | subject_text | +0.418735 | Glycol Fragrance (Parfum) May Contain: Iron Oxides (C77491. | Glycol, Fragrance (Parfum) May Contain: Iron Oxides (Ci 77491, CI | Di-PPG-3 Myristyl Ether |
| 616 | subject_text | +0.416835 | Questions or comments?1-855-353-7320 | Questions or comments?1-855-353-7320 | swallowed, get medical help or |
| 46 | background | +0.410619 |  |  |  |
| 607 | background | +0.404668 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Octinoxate 6.0% | 4 | -0.000487 |
| Octisalate 4.0% | 4 | +0.000954 |
| Oxybenzone 3.5% | 9 | +0.001430 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_51f7a3bb218030b8` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.013925 | +0.221243 |
| 只保留主体分差 | -0.016698 | +0.262020 |
| 主体 token 数 | 300 | 319 |

[成功对照 token 表](../photo_51f7a3bb218030b8/tokens.csv) · [对照摘要](../photo_51f7a3bb218030b8/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
