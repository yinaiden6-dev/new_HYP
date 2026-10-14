# photo_83941d6fe68296bd

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000995`，F03 错误参考 `reference_000395`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。 纯图像 Top-1 为 `reference_003684`，与 F03 的错误候选不同；本例 token 分差针对 F03 候选，不能单独解释另一个图像赢家。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 600 |
| 图像 Top-1 参考 | reference_003684 |
| BM25 真值排名 | 82 |
| 融合真值排名 | 110 |
| 图像正负分差 | -0.033490 |
| BM25 正负分差 | -20.064320 |
| 0.5 × 图像 z 分差 | -0.980422 |
| 0.5 × BM25 z 分差 | -1.126245 |
| 最终融合分差 | -2.106667 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 169 个。正向贡献 +0.036554，负向贡献 -0.070044，相加等于图像分差 -0.033490。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 599 | -0.045433 | -0.059086 |
| nonvisual | 11 | +0.000007 | +0.000519 |
| subject_nontext | 124 | +0.003442 | +0.021623 |
| subject_text | 45 | +0.008494 | +0.147038 |

仅保留主体 query token 的两候选分差：+0.055018。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.045433。主体框内 token 占有效序列 21.7%。固定特征、只筛选主体 token 后两候选分差翻正，值得进一步检验聚合范围。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 179 | 52.2% | True | +7.206017 | 70.2% |
| 错误 | 53 | 83.0% | True | +13.420223 | 98.1% |

主体中 45 个 token 与 OCR 相交，9 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 444 | background | -0.639843 |  |  |  |
| 252 | background | -0.627160 |  | PLASTIC | INTERCEPTOR |
| 349 | background | -0.617719 |  |  |  |
| 288 | background | -0.605449 |  |  |  |
| 321 | background | -0.600256 |  | Reducer |  |
| 187 | background | -0.598424 |  | Reducer |  |
| 127 | background | -0.592125 | INTERCEPTOR | Treats Frequent Heartburn | INTERCEPTOR |
| 157 | background | -0.590642 | (p/ | Reducer | INTERCEPTOR |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 443 | subject_text | +0.542277 | 3T3D5 BN C4 | 3T3D5 BN C4 | Product of France |
| 251 | subject_nontext | +0.523500 |  |  | been evaluated in dogs used for breeding or in lactating females. |
| 250 | subject_nontext | +0.518151 |  |  | been evaluated in dogs used for breeding or in lactating females. |
| 442 | subject_text | +0.509707 | 3T3D5 BN C4 | 3T3D5 BN C4 | 25.1–50 Ibs |
| 411 | subject_text | +0.469752 | 3T3D5 BN C4 | 3T3D5 BN C4 | 6 weeks |
| 507 | subject_text | +0.433156 | 3T3D5 BN C4 | 3T3D5 BN C4 |  |
| 315 | subject_nontext | +0.416243 |  |  |  |
| 506 | subject_text | +0.389775 | 3T3D5 BN C4 |  | Approved by FDA under NADA # 141-338 |

## 成功对照

`photo_ab6fd7a33337ccee` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.033490 | +0.101082 |
| 仅主体分差 | +0.055018 | +0.282803 |
| 主体 token 数 | 169 | 209 |

[对照 token 数据](../photo_ab6fd7a33337ccee/tokens.csv) · [对照摘要](../photo_ab6fd7a33337ccee/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
