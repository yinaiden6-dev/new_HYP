# photo_e6160ef7ce3f2f34

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000147`，F03 错误参考 `reference_000146`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 2 |
| 图像 Top-1 参考 | reference_000146 |
| BM25 真值排名 | 2 |
| 融合真值排名 | 2 |
| 图像正负分差 | -0.002477 |
| BM25 正负分差 | -2.732849 |
| 0.5 × 图像 z 分差 | -0.043811 |
| 0.5 × BM25 z 分差 | -0.033712 |
| 最终融合分差 | -0.077523 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 731 个，主体框内 72 个。正向贡献 +0.021118，负向贡献 -0.023595，相加等于图像分差 -0.002477。缓存复现误差 0。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 648 | -0.003791 | -0.004277 |
| nonvisual | 11 | -0.000246 | -0.016336 |
| subject_nontext | 30 | +0.000007 | +0.000160 |
| subject_text | 42 | +0.001553 | +0.027029 |

仅保留主体 query token 的两候选分差：+0.015833。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.003791。主体框内 token 占有效序列 9.8%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 490 | 56.2% | True | +6.340516 | 69.6% |
| 错误 | 65 | 29.3% | True | +6.982824 | 83.3% |

主体中 42 个 token 与 OCR 相交，33 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 443 | background | -0.493309 |  |  |  |
| 634 | background | -0.384399 |  | 75 |  |
| 36 | background | -0.326927 |  |  |  |
| 131 | background | -0.296052 |  |  |  |
| 466 | background | -0.208639 |  |  |  |
| 506 | background | -0.206956 |  |  |  |
| 576 | background | -0.193522 |  |  |  |
| 462 | background | -0.181722 |  |  |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 473 | subject_text | +0.369432 | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acd and | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acid and |  |
| 183 | background | +0.314770 |  | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acid and |  |
| 108 | background | +0.285644 |  |  | Inject subcutaneously as a single dose in the neck at a dosage |
| 477 | subject_text | +0.275341 | free base in a 50% propylene glycol vehicle, monothioglycerol |  | added to adjust pH. |
| 492 | subject_nontext | +0.245491 |  |  |  |
| 474 | subject_text | +0.231462 | sodium hydroxide added to adjust pH. | (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acid and | citric acid (4.8 mg/mL) with hydrochloric acid and sodium hydroxide |
| 481 | background | +0.220458 |  |  |  |
| 231 | background | +0.195871 |  |  | Inject subcutaneously as a single dose in the neck at a dosage |

## 末尾词／数字单位行的区域检查

按 patch 与 OCR 多边形的交叠重新选取 token，不要求该行在多行竞争中取得最大交叠。这些是粗粒度区域分差，包含邻近文字，多个行区域可能重复使用同一个 token，不能相加当作完整分差。

| OCR 行 | 区域 token 数 | 对完整均值分差的区域贡献 |
| --- | --- | --- |
| of 2.5 mg/kg (1 mL/22 Ib) Body Weight (BW). Do not inject | 5 | +0.000182 |
| more than 4 mL per injection site. | 3 | +0.000187 |
| Table 1. Increxxa 25 Swine Dosing Guide (25 mg/mL) | 5 | +0.000182 |
| of 2.5 mg/kg (1 mL/22 Ib) body weight (BW). Do not inject | 5 | -0.000103 |
| more than 11.5 mL per injection site. | 5 | +0.000108 |
| Table 2. Increxxa 25 Calf Dosing Guide (25 mg/mL) | 5 | +0.000154 |
| Each mL of Increoxa 25 contains 25 mg of tulathromycin as the | 9 | +0.001476 |
| free base in a 50% propylene glycol vehicle, monothioglycerol | 6 | +0.001302 |
| (5 mg/mL), citric acid (4.8 mg/mL) with hydrochloric acd and | 5 | +0.000926 |

[完整焦点行与 token 索引](focal_lines.csv)


## 成功对照

`photo_72ad5f2e7435bc37` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.002477 | +0.015164 |
| 仅主体分差 | +0.015833 | +0.034226 |
| 主体 token 数 | 72 | 210 |

[对照 token 数据](../photo_72ad5f2e7435bc37/tokens.csv) · [对照摘要](../photo_72ad5f2e7435bc37/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
