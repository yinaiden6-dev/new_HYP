# B03 · photo_dc4d31058a813a38

按当前评测真值分析，未新增身份人工复核。

包装 `package_088`，划分 `val`；用户主题：尚未人工分类。正确参考 `reference_005041`，**纯图像实际 Top-1 错误参考** `reference_005042`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2 |
| 正确参考 MaxSim | +0.570339 |
| 错误参考 MaxSim | +0.571862 |
| 正确减错误的分差 | -0.001523 |
| F03 的正确参考排名（辅助对照） | 1 |
| 基于测量的诊断 | 主体筛选可翻转这两个候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 779 个，主体框内视觉 token 253 个。正向贡献 +0.021200，负向贡献 -0.022724，相加等于原图像分差。与原评分缓存的最大误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 515 | -0.003479 | -0.005262 |
| nonvisual | 11 | -0.000012 | -0.000863 |
| subject_nontext | 147 | +0.001463 | +0.007752 |
| subject_text | 106 | +0.000505 | +0.003709 |

最主要负差额来自 `background`，贡献 -0.003479。只保留主体 query token 后，本对分差为 +0.006059。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 326 | 42.1% | True | +8.311821 | 71.8% |
| 错误 | 15 | 7.2% | False | 不可评估 | 不可评估 |

主体中 106 个 token 与 OCR 行相交，44 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 7 | background | -0.265955 |  |  |  |
| 540 | subject_text | -0.258552 | Vienna Spring Water | One pre-filled syringe with diluent contains 1 mL Sterile Water | One vial contains: 0.26-0.27 mg cetrorelix acetate (equivalent to 0.25 mg cetrorelix) and 54.8 mg mannitol. |
| 26 | background | -0.209311 |  | Cetrorelix Acetate | Cetrorelix Acetate for Injection |
| 571 | subject_text | -0.203083 | Vienna Spring Water | One pre-filled syringe with diluent contains 1 mL Sterile Water | One vial contains: 0.26-0.27 mg cetrorelix acetate (equivalent to 0.25 mg cetrorelix) and 54.8 mg mannitol. |
| 453 | background | -0.198918 | Water |  | One vial contains: 0.26-0.27 mg cetrorelix acetate (equivalent to 0.25 mg cetrorelix) and 54.8 mg mannitol. |
| 205 | background | -0.182140 |  | Sterile - for subcutaneous use only |  |
| 10 | background | -0.159199 |  |  | Cetrorelix Acetate for Injection |
| 542 | background | -0.148548 | Vienna is famo | Made in Spain |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 497 | subject_nontext | +0.340169 |  |  | 1312001441-00 |
| 248 | subject_text | +0.315958 | BO |  |  |
| 247 | subject_text | +0.289329 | BO |  |  |
| 128 | background | +0.271335 | IEEE Robotics & Automation Society |  |  |
| 646 | background | +0.246195 |  |  |  |
| 341 | subject_nontext | +0.228530 |  |  |  |
| 662 | background | +0.218314 |  |  |  |
| 465 | subject_text | +0.197374 | One carton contains one packaged tray which contains: |  |  |

## 数字单位／末尾词区域

| OCR 行 | 交叠 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| for Injection 0.25 mg | 16 | -0.000062 |

区域可能重叠、包含相邻文字，不能相加当作独立词语贡献。[焦点行及 token 索引](focal_lines.csv)。


## 同包装成功对照

`photo_39880628576a7ea5` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.001523 | +0.012927 |
| 只保留主体分差 | +0.006059 | +0.023256 |
| 主体 token 数 | 253 | 220 |

[成功对照 token 表](../photo_39880628576a7ea5/tokens.csv) · [对照摘要](../photo_39880628576a7ea5/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
