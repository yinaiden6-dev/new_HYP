# photo_3fa448ba98849bb7

按当前真值分析，未新增身份复核。 用户指定主题：尚未人工分类。

正确参考 `reference_000995`，F03 错误参考 `reference_000852`。图像评分中已失败；编码器与 MaxSim 的责任尚未区分。

## 评分阶段

| 检查 | 结果 |
| --- | --- |
| 图像真值排名 | 4 |
| 图像 Top-1 参考 | reference_000852 |
| BM25 真值排名 | 4 |
| 融合真值排名 | 3 |
| 图像正负分差 | -0.010903 |
| BM25 正负分差 | -70.896622 |
| 0.5 × 图像 z 分差 | -0.240372 |
| 0.5 × BM25 z 分差 | -2.002964 |
| 最终融合分差 | -2.243336 |

分差为正确参考减错误参考；归一化使用完整 5,405 行图库。

固定这套完整图库标准化尺度，改变融合权重使正确参考超过本例错误参考的条件：**两个信号均不支持正确候选；凸组合权重不能使此分差为正**。此条件仅描述双候选，不保证完整图库 Top-1；没有按这些阈值调整 baseline。

## Token 贡献

有效 token 779 个，主体框内 156 个。正向贡献 +0.020851，负向贡献 -0.031754，相加等于图像分差 -0.010903。缓存复现误差 6e-08。

| 区域 | Token 数 | 对完整均值分差的贡献 | 组内分差均值 |
| --- | --- | --- | --- |
| background | 612 | -0.007233 | -0.009207 |
| nonvisual | 11 | +0.000159 | +0.011226 |
| subject_nontext | 86 | -0.001886 | -0.017081 |
| subject_text | 70 | -0.001942 | -0.021615 |

仅保留主体 query token 的两候选分差：-0.019116。这是固定特征的双候选匹配筛选，没有重新编码，也没有重排完整图库。

对完整图像分差最不利的区域是 `background`，贡献 -0.007233。主体框内 token 占有效序列 20.0%。主体内分差仍为负，简单筛选主体不足以让正确候选超过这个错误候选。

## 对应与独立几何

| 候选 | SIFT 内点 | 主体覆盖率 | 对齐通过阈值 | 残差中位数（网格单元） | 超过 3 单元比例 |
| --- | --- | --- | --- | --- | --- |
| 正确 | 130 | 37.5% | True | +6.302050 | 67.9% |
| 错误 | 78 | 6.4% | False | 不可评估 | 不可评估 |

主体中 70 个 token 与 OCR 相交，52 个与多行相交。视觉 token 对应网格区域，不能直接当作单词。对齐失败时不填写空间错配结论；通过自动阈值仍需人工核对。


## 最不利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 436 | subject_text | -0.388498 | Do not lie flat or bend over soon after eating | Do not eat late at night or just before bedtime. | Tips for Managing Heartburn |
| 756 | background | -0.352958 |  | Do not eat late at night or just before bedtime. | Tips for Managing Heartburn |
| 371 | subject_text | -0.306128 | Eat slowly and do not eat big meals. |  | Eat slowly and do not eat big |
| 345 | subject_nontext | -0.273341 |  | BOX |  |
| 248 | subject_text | -0.256352 | how2recyclainfo | be caused by a serious condition. | how2recycle,info |
| 153 | background | -0.244219 |  | BOX |  |
| 355 | background | -0.238046 |  | Do not eat late at night or just before bedtime. | chocolate, caffeine, alcohol, even |
| 376 | subject_text | -0.235191 | PAPER | BOX |  |

## 最有利的 8 个 token

| 索引 | 区域 | Δi | Query OCR | 正确参考 OCR | 错误参考 OCR |
| --- | --- | --- | --- | --- | --- |
| 17 | background | +0.300511 |  | Tips for Managing Heartburn | Cimetidine |
| 501 | subject_text | +0.258522 | Wear loose-fitting clothing around | • Raise the head of your bed |  |
| 319 | background | +0.224954 |  |  |  |
| 305 | subject_text | +0.219203 | Avoid foods or drinks that are more likely to |  |  |
| 731 | background | +0.195453 |  | Tips for Managing Heartburn | Do not eat late at night, or just |
| 43 | background | +0.185850 |  | Tips for Managing Heartburn |  |
| 21 | background | +0.182934 |  | • Raise the head of your bed | CIMETIDINE TABLETS USP, 200 mg / |
| 302 | subject_nontext | +0.180730 |  |  |  |

## 成功对照

`photo_cf724bd3f6c03346` 为同包装、图像和融合均正确的照片。按拍摄标签和主体面积匹配；可见包装面尚未逐对人工核实。

| 诊断 | 失败照片 | 成功照片 |
| --- | --- | --- |
| 图像分差 | -0.010903 | +0.100736 |
| 仅主体分差 | -0.019116 | +0.178242 |
| 主体 token 数 | 156 | 144 |

[对照 token 数据](../photo_cf724bd3f6c03346/tokens.csv) · [对照摘要](../photo_cf724bd3f6c03346/summary.json)


## 图与原始数据

![对应连线](correspondence.png)

![分差贡献](contribution.png)

[完整 token 对应](tokens.csv) · [区域汇总](regions.csv) · [OCR 行汇总](ocr_lines.csv) · [几何与摘要](summary.json)

## 解释边界

本例定位了评分阶段和正负差额来源。Token 特征包含上下文，框外 token 贡献不能直接解释为背景像素的因果作用。没有运行 feature probe 或像素遮挡、替换与重编码，不能断言编码器缺少布局、颜色或细字信息。
