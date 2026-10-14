# B03 · photo_38ab72e39bb67515

按当前评测真值分析，未新增身份人工复核。

包装 `package_013`，划分 `train`；用户主题：尚未人工分类。正确参考 `reference_000929`，**纯图像实际 Top-1 错误参考** `reference_001896`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 2167 |
| 正确参考 MaxSim | +0.526851 |
| 错误参考 MaxSim | +0.587577 |
| 正确减错误的分差 | -0.060726 |
| F03 的正确参考排名（辅助对照） | 5 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 144 个。正向贡献 +0.015997，负向贡献 -0.076723，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 576 | -0.048045 | -0.060974 |
| nonvisual | 11 | -0.000326 | -0.021673 |
| subject_nontext | 113 | -0.008844 | -0.057214 |
| subject_text | 31 | -0.003511 | -0.082784 |

最主要负差额来自 `background`，贡献 -0.048045。只保留主体 query token 后，本对分差为 -0.062719。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 17 | 9.5% | False | 不可评估 | 不可评估 |
| 错误 | 4 | 2.3% | False | 不可评估 | 不可评估 |

主体中 31 个 token 与 OCR 行相交，10 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 660 | background | -0.305408 |  |  |  |
| 175 | background | -0.288546 |  |  | TAMPER EVIDENT |
| 481 | background | -0.282436 |  |  |  |
| 121 | background | -0.279942 |  |  |  |
| 476 | subject_nontext | -0.278455 |  |  | TAMPER EVIDENT |
| 454 | subject_nontext | -0.278301 |  |  | TAMPER EVIDENT |
| 251 | background | -0.275788 |  | See below for important information about use in children. Children 4-11: do not use | TAMPER EVIDENT |
| 714 | background | -0.272072 |  |  | Acetaminophen 500 mg |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 87 | background | +0.532948 |  |  |  |
| 503 | background | +0.387974 |  |  |  |
| 290 | subject_nontext | +0.378412 |  |  |  |
| 310 | subject_nontext | +0.370797 |  |  |  |
| 316 | subject_nontext | +0.355633 |  |  |  |
| 314 | subject_nontext | +0.347167 |  |  | Cramps/Cólicos·Bloating/Hinchazón |
| 340 | subject_nontext | +0.321434 |  |  |  |
| 320 | subject_nontext | +0.319968 |  |  |  |

## 同包装成功对照

`photo_cdd1aa422a4f6dc0` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.060726 | +0.362146 |
| 只保留主体分差 | -0.062719 | +0.433593 |
| 主体 token 数 | 144 | 150 |

[成功对照 token 表](../photo_cdd1aa422a4f6dc0/tokens.csv) · [对照摘要](../photo_cdd1aa422a4f6dc0/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
