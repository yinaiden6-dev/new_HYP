# B03 · photo_1f48d45127db4129

按当前评测真值分析，未新增身份人工复核。

包装 `package_047`，划分 `train`；用户主题：文字排版结构。正确参考 `reference_001139`，**纯图像实际 Top-1 错误参考** `reference_004489`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 148 |
| 正确参考 MaxSim | +0.590654 |
| 错误参考 MaxSim | +0.710510 |
| 正确减错误的分差 | -0.119856 |
| F03 的正确参考排名（辅助对照） | 10 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 252 个。正向贡献 +0.012130，负向贡献 -0.131986，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 468 | -0.068335 | -0.106737 |
| nonvisual | 11 | -0.000199 | -0.013207 |
| subject_nontext | 70 | -0.010582 | -0.110504 |
| subject_text | 182 | -0.040740 | -0.163633 |

最主要负差额来自 `background`，贡献 -0.068335。只保留主体 query token 后，本对分差为 -0.148875。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 847 | 77.8% | True | +8.620227 | 78.2% |
| 错误 | 24 | 8.4% | False | 不可评估 | 不可评估 |

主体中 182 个 token 与 OCR 行相交，145 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 329 | subject_text | -0.555177 | Warnings | Itcontains important information. |  |
| 439 | background | -0.524504 |  |  | bronchitis, or emphysema |
| 438 | subject_nontext | -0.509344 |  |  | bronchitis, or emphysema |
| 310 | subject_text | -0.509183 | the intensity of coughing | 719283151100 |  |
| 330 | subject_text | -0.508319 | Warnings | Itcontains important information. |  |
| 450 | subject_text | -0.502422 | cough accomparied by too much phlegm(mucus) |  | occurs with smoking, asthma, chronic |
| 495 | subject_text | -0.502070 | cough lasts more than 7 days, comes back, or occurs | Itcontains important information. |  |
| 521 | background | -0.500871 |  | meijer. | health professional before use. |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 293 | subject_text | +0.400216 | cough due to minor throat and bronchial rritation as | the impulse to cough to help you get to sleep | cough due to minor throat and |
| 303 | background | +0.307176 |  |  | Amneal Pharmaceuticals LLC |
| 183 | background | +0.264926 |  |  |  |
| 107 | background | +0.264553 |  |  |  |
| 307 | subject_nontext | +0.219715 |  |  |  |
| 180 | background | +0.190448 |  | Drug Facts | †Compare to the active ingredients of Mucinex® DM |
| 69 | background | +0.187719 |  | Dextromethorphan HBr 60 mg..... Cough suppressant | EXPECTORANT |
| 165 | background | +0.185065 |  |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| De tromethorphan HBr 60 mg.....Cough suppreesant | 10 | -0.001056 |
| Gualeneein 1200 mg | 8 | -0.000407 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_02c11dd14bf4116f` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.119856 | +0.082564 |
| 只保留主体分差 | -0.148875 | +0.119655 |
| 主体 token 数 | 252 | 220 |

[成功对照 token 表](../photo_02c11dd14bf4116f/tokens.csv) · [对照摘要](../photo_02c11dd14bf4116f/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
