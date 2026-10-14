# B03 · photo_7d94403c5a56e1a9

按当前评测真值分析，未新增身份人工复核。

包装 `package_065`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_003233`，**纯图像实际 Top-1 错误参考** `reference_004269`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 49 |
| 正确参考 MaxSim | +0.553588 |
| 错误参考 MaxSim | +0.583146 |
| 正确减错误的分差 | -0.029557 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 77 个。正向贡献 +0.046658，负向贡献 -0.076215，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 643 | -0.028051 | -0.031890 |
| nonvisual | 11 | -0.000233 | -0.015505 |
| subject_nontext | 34 | +0.000826 | +0.017758 |
| subject_text | 43 | -0.002099 | -0.035680 |

最主要负差额来自 `background`，贡献 -0.028051。只保留主体 query token 后，本对分差为 -0.012084。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 120 | 20.9% | True | +12.132956 | 77.9% |
| 错误 | 9 | 5.2% | False | 不可评估 | 不可评估 |

主体中 43 个 token 与 OCR 行相交，32 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 40 | background | -0.556978 |  | drink a full glass of water with each dose | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |
| 223 | background | -0.556281 |  | drink a full glass of water with each dose | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |
| 88 | background | -0.549448 |  | drink a full glass of water with each dose | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |
| 100 | background | -0.548886 |  | drink a full glass of water with each dose | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |
| 182 | background | -0.537151 |  |  |  |
| 347 | background | -0.531665 |  | drink a full glass of water with each dose | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |
| 343 | background | -0.515474 |  | drink a full glass of water with each dose | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |
| 333 | background | -0.502152 |  | TAMPER EVIDENT: DO NOT USE IF IMPRINTED | TAMPER EVIDENT: DO NOT USE IF IMPRINTED SAFETY |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 373 | subject_text | +0.612852 | BEST CHOICE |  |  |
| 562 | background | +0.582027 |  | Choice |  |
| 383 | background | +0.573074 |  | Choice |  |
| 390 | background | +0.562379 |  | Choice |  |
| 573 | background | +0.544763 |  |  |  |
| 367 | background | +0.542321 |  | Choice | Sore throat warning: If sore throat is severe, persists for more |
| 374 | subject_text | +0.539096 | BEST CHOICE |  | KEEP OUTER PACKAGE FOR |
| 599 | background | +0.490848 |  |  |  |

## 同包装成功对照

`photo_5149e96e0098a5e9` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.029557 | +0.108195 |
| 只保留主体分差 | -0.012084 | +0.142904 |
| 主体 token 数 | 77 | 60 |

[成功对照 token 表](../photo_5149e96e0098a5e9/tokens.csv) · [对照摘要](../photo_5149e96e0098a5e9/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
