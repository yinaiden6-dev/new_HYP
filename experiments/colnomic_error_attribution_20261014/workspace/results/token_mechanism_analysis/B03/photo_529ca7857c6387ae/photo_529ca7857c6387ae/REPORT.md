# B03 · photo_529ca7857c6387ae

此照片对其他候选曾有身份歧义标记；本例负候选不同，未把那个标签直接转移到本对。

包装 `package_025`，划分 `test`；用户主题：尚未人工分类。正确参考 `reference_000999`，**纯图像实际 Top-1 错误参考** `reference_001001`。

## 评分与检索

| 检查 | 结果 |
| --- | --- |
| 正确参考排名 | 4 |
| 正确参考 MaxSim | +0.615013 |
| 错误参考 MaxSim | +0.623723 |
| 正确减错误的分差 | -0.008709 |
| F03 的正确参考排名（辅助对照） | 4 |
| 基于测量的诊断 | 主体匹配仍偏错误候选 |

B03 只用冻结图像特征的 MaxSim。未将 BM25 或融合造成的错误归入本例；编码器与匹配函数的责任仍需进一步实验区分。

## Token 分差守恒

有效 token 731 个，主体框内视觉 token 90 个。正向贡献 +0.013826，负向贡献 -0.022536，相加等于原图像分差。与原评分缓存的最大误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 区域内均值 |
| --- | --- | --- | --- |
| background | 630 | -0.006928 | -0.008038 |
| nonvisual | 11 | -0.000215 | -0.014291 |
| subject_nontext | 82 | -0.001086 | -0.009679 |
| subject_text | 8 | -0.000481 | -0.043911 |

最主要负差额来自 `background`，贡献 -0.006928。只保留主体 query token 后，本对分差为 -0.012722。这仅是两个候选的固定特征筛选诊断，没有重新编码，也没有重排完整图库。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（参考网格单元） | 超过3单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 8 | 14.6% | False | 不可评估 | 不可评估 |
| 错误 | 11 | 15.2% | True | +7.511333 | 83.3% |

主体中 8 个 token 与 OCR 行相交，2 个与多行相交。视觉 token 是网格区域，不是单词；自动对齐失败时不报告空间错配结论，通过阈值仍非人工几何真值。


## 最不利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 683 | background | -0.173675 |  |  |  |
| 54 | background | -0.170880 |  |  |  |
| 614 | background | -0.170145 |  |  |  |
| 87 | background | -0.161962 |  |  |  |
| 94 | background | -0.152845 |  |  |  |
| 549 | background | -0.148702 |  |  |  |
| 155 | background | -0.142502 |  |  |  |
| 651 | background | -0.129031 |  | CE |  |

## 最有利的8个 token

| 缓存索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 201 | background | +0.159693 |  |  |  |
| 585 | background | +0.144466 |  | Silica, Microcrystalline Wax, Ceresin, Ozokerite, DiglycerylSebacate/Isopalmitate, Octyldodecanol, Alumina, Red 28 Lake (Cl 45410), Polyglyceryl-2 |  |
| 367 | background | +0.120831 |  |  |  |
| 204 | background | +0.116733 |  |  |  |
| 33 | background | +0.113337 |  | Silica, Microcrystalline Wax, Ceresin, Ozokerite, DiglycerylSebacate/Isopalmitate, Octyldodecanol, Alumina, Red 28 Lake (Cl 45410), Polyglyceryl-2 |  |
| 212 | background | +0.110637 |  |  | 3CE lip Color #906 BITE IT FUNCTION: Lipstick INSTRUCTIONS: A rich color lipstick that moistly adheres to lips |
| 220 | background | +0.106885 |  |  |  |
| 53 | background | +0.106778 |  | Silica, Microcrystalline Wax, Ceresin, Ozokerite, DiglycerylSebacate/Isopalmitate, Octyldodecanol, Alumina, Red 28 Lake (Cl 45410), Polyglyceryl-2 |  |

## 同包装成功对照

`photo_16f724e075d1e714` 为纯图像 Top-1 正确的照片，不要求融合也正确。按拍摄标签和主体面积接近程度选择，可见包装面未全部人工确认。

| 诊断 | 错误照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.008709 | +0.010775 |
| 只保留主体分差 | -0.012722 | +0.006052 |
| 主体 token 数 | 90 | 112 |

[成功对照 token 表](../photo_16f724e075d1e714/tokens.csv) · [对照摘要](../photo_16f724e075d1e714/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![贡献分解](contribution.png)

[完整 token 对应与分差](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [摘要与几何内点](summary.json)

## 结论边界

本轮完成匹配对应与差额核算。Token 特征有上下文，框外贡献不等价于背景像素的因果作用。未运行特征 probe、像素遮挡／替换及重编码，不能断言编码器缺少某种布局、颜色或细字信息。
