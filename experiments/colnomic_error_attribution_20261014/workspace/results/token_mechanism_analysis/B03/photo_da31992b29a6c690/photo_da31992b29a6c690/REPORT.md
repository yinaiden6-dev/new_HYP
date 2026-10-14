# B03 · photo_da31992b29a6c690

按当前评测真值分析，未新增身份人工复核。

包装 `package_023`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000977`，**纯图像实际 Top-1 错误参考** `reference_002548`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.589598 |
| 错误参考 MaxSim | +0.591538 |
| 正确减错误的分差 | -0.001940 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 200 个。正向贡献 +0.023047，负向贡献 -0.024987，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 568 | -0.003983 | -0.005463 |
| nonvisual | 11 | -0.000096 | -0.006799 |
| subject_nontext | 49 | +0.000490 | +0.007794 |
| subject_text | 151 | +0.001649 | +0.008507 |

最主要负差额来自 `background`，贡献 -0.003983。只保留主体 query token 后，本对分差为 +0.008332。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 361 | 68.5% | True | +4.503256 | 58.0% |
| 错误 | 21 | 14.6% | False | 不可评估 | 不可评估 |

主体中 151 个 token 与 OCR 行相交，97 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 297 | background | -0.238653 |  |  |  |
| 283 | subject_text | -0.230687 | put 1 or 2 drops in the affected eye(s) as needed | put 1 or 2 drops in the affected eye(s) as needed | as needed |
| 141 | background | -0.224478 |  |  |  |
| 467 | subject_text | -0.202470 | Do not use |  | For external use only |
| 506 | subject_text | -0.178515 | away. | away. | (1-800-222-1222) right away. |
| 499 | subject_text | -0.174132 | Do not use |  | For external use only |
| 244 | subject_text | -0.169971 | if you are sensitive to any ingredient in this product |  |  |
| 565 | background | -0.165687 |  |  |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 334 | subject_text | +0.497093 | for the temporary relief of burning and irritation |  |  |
| 475 | subject_text | +0.392464 | put 1 or 2 drops in the affected eye(s) as needed | downward to remove residual | one or two drops in the |
| 110 | background | +0.337151 |  |  | TAMPER EVIDENT: DO NOT |
| 667 | background | +0.292782 |  | downward to remove residual | Stop use and ask a doctor if you |
| 445 | subject_text | +0.253378 | store at room temperature | store at room temperature |  |
| 654 | background | +0.216623 |  | for the temporary relief of burning and irritation |  |
| 763 | background | +0.203535 |  |  |  |
| 615 | background | +0.188327 |  |  | if this product changes color or |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| Propylene Glycol 0.6% | 8 | -0.000347 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_f1f661a94082552c` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001940 | +0.046432 |
| 只保留主体分差 | +0.008332 | +0.086422 |
| 主体 token 数 | 200 | 176 |

[成功对照 token 表](../photo_f1f661a94082552c/tokens.csv) · [对照摘要](../photo_f1f661a94082552c/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
