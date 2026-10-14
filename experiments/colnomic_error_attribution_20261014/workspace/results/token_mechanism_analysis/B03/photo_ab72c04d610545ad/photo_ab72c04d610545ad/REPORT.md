# B03 · photo_ab72c04d610545ad

按当前评测真值分析，未新增身份人工复核。

包装 `package_065`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003233`，**纯图像实际 Top-1 错误参考** `reference_004585`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 338 |
| 正确参考 MaxSim | +0.527122 |
| 错误参考 MaxSim | +0.601485 |
| 正确减错误的分差 | -0.074363 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 81 个。正向贡献 +0.038388，负向贡献 -0.112751，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 639 | -0.070913 | -0.081123 |
| nonvisual | 11 | -0.000421 | -0.028004 |
| subject_nontext | 49 | +0.001624 | +0.024231 |
| subject_text | 32 | -0.004653 | -0.106289 |

最主要负差额来自 `background`，贡献 -0.070913。只保留主体 query token 后，本对分差为 -0.027333。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 173 | 47.7% | True | +4.934762 | 69.1% |
| 错误 | 12 | 2.3% | False | 不可评估 | 不可评估 |

主体中 32 个 token 与 OCR 行相交，18 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 31 | background | -0.624488 |  |  |  |
| 151 | background | -0.585245 |  |  |  |
| 220 | background | -0.566090 |  |  |  |
| 414 | subject_text | -0.561905 | No print/No varnish | No print/No varnish | no print / no varnish area |
| 295 | subject_text | -0.542209 | TAMPER EVIDENT: DO NOT USE IF IMPRINTED | product or any of its ingredients | TAMPER EVIDENT: DO NOT USE IF |
| 595 | background | -0.540406 |  |  | no print / no varnish area |
| 415 | subject_text | -0.540334 | No print/No varnish | Best Choice. | no print / no varnish area |
| 693 | background | -0.538873 |  |  | CAP IS BROKEN OR MISSING |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 320 | background | +0.588479 |  |  |  |
| 691 | background | +0.583372 |  |  |  |
| 692 | background | +0.569205 |  | Best Choice. |  |
| 321 | background | +0.556761 |  | Best Choice. |  |
| 341 | background | +0.508618 |  | Choice |  |
| 521 | background | +0.504852 |  | Best Choice. |  |
| 310 | subject_text | +0.487135 | BEST CHOICE | Best |  |
| 311 | subject_text | +0.482878 | BEST CHOICE | Choice |  |

## 同包装成功对照

`photo_f2e65c6d203e45a2` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.074363 | +0.199188 |
| 只保留主体分差 | -0.027333 | +0.236789 |
| 主体 token 数 | 81 | 120 |

[成功对照 token 表](../photo_f2e65c6d203e45a2/tokens.csv) · [对照摘要](../photo_f2e65c6d203e45a2/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
