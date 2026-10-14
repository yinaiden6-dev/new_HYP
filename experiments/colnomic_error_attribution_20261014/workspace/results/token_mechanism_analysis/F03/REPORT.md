# F03 全部 145 个错误：Token 对应与贡献

逐案复用冻结 embedding 与评分缓存，分析正确／错误候选的全部有效 token；没有修改 baseline。每例均附独立报告、完整对应 CSV、贡献分解、匹配连线和可用成功对照。

## 已验证的失败阶段

| 范围 | 图像已错 | 图像原本正确、融合变错 | 总计 |
| --- | --- | --- | --- |
| 全部 F03 错误 | 104 | 41 | 145 |
| 排除已有身份歧义 | 86 | 40 | 126 |

19 例已有身份歧义标签，保留分析但不直接作为模型失败证据。现有人工分类文件含 0 条记录；四个主题来自用户，其余案例未编造语义类别。

## 根据测量得到的逐案诊断分组

| 诊断 | 案例数 |
| --- | --- |
| 身份歧义待复核 | 19 |
| 主体筛选可翻转本对候选 | 18 |
| 主体匹配仍偏错误候选 | 56 |
| 融合覆盖正确图像赢家 | 40 |
| 纯图像另有错误赢家，当前负候选来自融合 | 12 |

这些是评分与筛选行为的分组，不是已经验证的编码器因果机制。F03 的错误候选可能与纯图像 Top-1 不同；逐案报告分别记录两者。

## 指定的四个案例

| 案例 | 图像排名 | 融合排名 | 图像分差 | 0.5图像z差 | 0.5 BM25 z差 | 仅主体分差 |
| --- | --- | --- | --- | --- | --- | --- |
| [换行与末尾词位置](photo_41de3eb383934ec3/photo_41de3eb383934ec3/REPORT.md) | 2 | 2 | -0.004094 | -0.105170 | -0.168726 | -0.015571 |
| [颜色差异](photo_21bd7df9ed6795e7/photo_21bd7df9ed6795e7/REPORT.md) | 1 | 2 | +0.018803 | +0.338296 | -0.933426 | +0.022199 |
| [文字排版结构](photo_1f48d45127db4129/photo_1f48d45127db4129/REPORT.md) | 148 | 10 | -0.119856 | -1.239275 | +0.460947 | -0.148875 |
| [细小文字差异](photo_0e87e0717ab1c215/photo_0e87e0717ab1c215/REPORT.md) | 2 | 2 | -0.005469 | -0.079247 | -0.136362 | -0.007399 |

颜色案例的图像结果正确，应首先追查融合。文字结构案例图像排名 148，融合升至第 10，仍失败。20/50 mL 和 100/250 mL 案例的图像分差为负。容量、NDC、其他细节也有变化，不能当作只改变排版的受控实验。

## 覆盖与对照

| 检查 | 结果 |
| --- | --- |
| 完成案例 | 145/145 |
| 同包装成功对照 | 113 |
| 不同成功照片 | 71 |
| 缺少对照 | 32 |
| 正负几何均通过阈值 | 68 |
| 主体 token 筛选使两候选由负翻正 | 19/90 |
| 同样筛选损害成功对照的两候选分差 | 6/113 |
| 最大缓存分数复现误差 | 1.7881393432617188e-07 |

对照可重复使用，不能视为独立样本。可见包装面没有全部人工确认。主体筛选翻转仅指这两个候选，不是完整图库 Recall@1 改善，也不是裁剪图片重新编码的效果。

## 方法与结论边界

- MaxSim 使用有效 query token 均值，保留有效非图像 token；去掉零向量，FP32 点积，不再次归一化。每例验证分差守恒。
- 网格从原始解码方向按 EXIF 转至显示坐标，再关联主体框与 OCR；patch 面积交叠至少 5% 才分配行，同时记录多行歧义。
- 独立 SIFT 对齐：query 主体框、完整参考图、比率 .75、最大边 1400、RANSAC 4px、至少 8 内点且覆盖主体 15%。自动通过仍非人工几何真值。
- 匹配图固定展示主体中最有利／最不利各 8 个 token；3 网格单元空间残差仅用于描述，没有被验证为异常阈值。
- 贡献是差额核算。上下文特征、框内遮挡、展开图重复面和透视限制语义归因。编码信息可分性需要 feature probe，像素级因果作用需要受控干预。

## 数据与复现

[错误评分阶段清单](F03_error_inventory.csv) · [全部摘要](cohort_summary.csv) · [融合权重双候选诊断](fusion_weight_diagnostics.csv) · [覆盖和数值验证](verification.json) · [完整 JSON 与缓存签名](summary.json)

[逐案诊断分组与图像赢家](diagnostics.csv)

在已装 ML 依赖的环境运行：

```bash
python -B scripts/analyze_colnomic_tokens.py --all-errors --resume
python -B scripts/report_colnomic_tokens.py
```

## 全部逐案报告

| 案例 | 用户主题 | 图像排名 | 融合排名 | 图像分差 | 仅主体分差 | 成功对照 | 身份状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [photo_01baf6cd7fca88bb](photo_01baf6cd7fca88bb/photo_01baf6cd7fca88bb/REPORT.md) | 尚未人工分类 | 390 | 8 | -0.016694 | -0.007344 | 无 | 身份歧义 |
| [photo_02fb3c06e1fb70e5](photo_02fb3c06e1fb70e5/photo_02fb3c06e1fb70e5/REPORT.md) | 尚未人工分类 | 189 | 48 | -0.087715 | +0.060815 | photo_80eb2d50ed90738e | 待复核 |
| [photo_04a4b44735fe9ba8](photo_04a4b44735fe9ba8/photo_04a4b44735fe9ba8/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.013976 | -0.017099 | 无 | 待复核 |
| [photo_062fafe5f481faca](photo_062fafe5f481faca/photo_062fafe5f481faca/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.002851 | -0.008744 | photo_ca454f8a9d866abb | 待复核 |
| [photo_079284acc905a73f](photo_079284acc905a73f/photo_079284acc905a73f/REPORT.md) | 尚未人工分类 | 1 | 3 | +0.005185 | +0.015135 | 无 | 待复核 |
| [photo_0913bcc35e9beb5a](photo_0913bcc35e9beb5a/photo_0913bcc35e9beb5a/REPORT.md) | 尚未人工分类 | 2216 | 2 | -0.106005 | -0.125506 | photo_e82154bdf3d3c846 | 待复核 |
| [photo_0a0e75a3b7797609](photo_0a0e75a3b7797609/photo_0a0e75a3b7797609/REPORT.md) | 尚未人工分类 | 13 | 29 | -0.018798 | +0.011385 | photo_12f7c257ae1f8f3c | 待复核 |
| [photo_0a612f53747b2554](photo_0a612f53747b2554/photo_0a612f53747b2554/REPORT.md) | 尚未人工分类 | 3 | 2 | -0.002653 | -0.017042 | photo_cdd1aa422a4f6dc0 | 待复核 |
| [photo_0be06814ec4048e3](photo_0be06814ec4048e3/photo_0be06814ec4048e3/REPORT.md) | 尚未人工分类 | 9 | 7 | -0.006459 | -0.014566 | photo_47430c79b3238e4a | 待复核 |
| [photo_0e87e0717ab1c215](photo_0e87e0717ab1c215/photo_0e87e0717ab1c215/REPORT.md) | 细小文字差异 | 2 | 2 | -0.005469 | -0.007399 | photo_72ad5f2e7435bc37 | 待复核 |
| [photo_0f45e3b0cc440b0a](photo_0f45e3b0cc440b0a/photo_0f45e3b0cc440b0a/REPORT.md) | 尚未人工分类 | 678 | 3 | -0.044576 | -0.049802 | photo_38d1d33620ab1ea1 | 待复核 |
| [photo_10fccaade3338c73](photo_10fccaade3338c73/photo_10fccaade3338c73/REPORT.md) | 尚未人工分类 | 2 | 3 | -0.003373 | +0.015225 | photo_80eb2d50ed90738e | 待复核 |
| [photo_12b37e2d68711d62](photo_12b37e2d68711d62/photo_12b37e2d68711d62/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.010871 | +0.007021 | 无 | 待复核 |
| [photo_16ff2e9baedfa1da](photo_16ff2e9baedfa1da/photo_16ff2e9baedfa1da/REPORT.md) | 尚未人工分类 | 10 | 3 | -0.000822 | -0.000944 | photo_51f7a3bb218030b8 | 身份歧义 |
| [photo_1db26288dca18777](photo_1db26288dca18777/photo_1db26288dca18777/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.001554 | -0.003463 | photo_ca454f8a9d866abb | 待复核 |
| [photo_1f48d45127db4129](photo_1f48d45127db4129/photo_1f48d45127db4129/REPORT.md) | 文字排版结构 | 148 | 10 | -0.119856 | -0.148875 | photo_02c11dd14bf4116f | 待复核 |
| [photo_206a9b7d61884264](photo_206a9b7d61884264/photo_206a9b7d61884264/REPORT.md) | 尚未人工分类 | 10 | 10 | -0.008121 | -0.010261 | 无 | 身份歧义 |
| [photo_21bd7df9ed6795e7](photo_21bd7df9ed6795e7/photo_21bd7df9ed6795e7/REPORT.md) | 颜色差异 | 1 | 2 | +0.018803 | +0.022199 | photo_badd72818d4cd9d8 | 待复核 |
| [photo_22e9695d01cb05c1](photo_22e9695d01cb05c1/photo_22e9695d01cb05c1/REPORT.md) | 尚未人工分类 | 20 | 14 | +0.039225 | +0.173404 | photo_2df687d8a2f89f40 | 待复核 |
| [photo_264a39405899b54d](photo_264a39405899b54d/photo_264a39405899b54d/REPORT.md) | 尚未人工分类 | 4 | 2 | -0.002339 | -0.001920 | photo_3bf9d27254c310cf | 待复核 |
| [photo_28d813f7c9451126](photo_28d813f7c9451126/photo_28d813f7c9451126/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.007260 | +0.018893 | 无 | 身份歧义 |
| [photo_2bb7a4894380d509](photo_2bb7a4894380d509/photo_2bb7a4894380d509/REPORT.md) | 尚未人工分类 | 10 | 9 | -0.013508 | -0.020073 | 无 | 身份歧义 |
| [photo_2bdbf879ee1d7dd8](photo_2bdbf879ee1d7dd8/photo_2bdbf879ee1d7dd8/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.008512 | -0.021425 | 无 | 待复核 |
| [photo_2d06e466a2c40697](photo_2d06e466a2c40697/photo_2d06e466a2c40697/REPORT.md) | 尚未人工分类 | 3 | 3 | +0.002445 | +0.005997 | photo_613ba5f27d0c5cfe | 待复核 |
| [photo_2e80aa2f6651a382](photo_2e80aa2f6651a382/photo_2e80aa2f6651a382/REPORT.md) | 尚未人工分类 | 1 | 3 | +0.082449 | +0.232488 | photo_b881c4736ab7cef0 | 待复核 |
| [photo_312d9f59eef6563d](photo_312d9f59eef6563d/photo_312d9f59eef6563d/REPORT.md) | 尚未人工分类 | 690 | 2 | -0.004400 | +0.006386 | 无 | 身份歧义 |
| [photo_31ff4c6250aab6d5](photo_31ff4c6250aab6d5/photo_31ff4c6250aab6d5/REPORT.md) | 尚未人工分类 | 23 | 4 | -0.011962 | -0.047620 | photo_6b934fda731eb0bb | 待复核 |
| [photo_324805558f8eab41](photo_324805558f8eab41/photo_324805558f8eab41/REPORT.md) | 尚未人工分类 | 1711 | 1711 | -0.057280 | -0.013147 | photo_71a0dab1e3a6a114 | 待复核 |
| [photo_32a5170c4382b4e9](photo_32a5170c4382b4e9/photo_32a5170c4382b4e9/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.008667 | +0.012234 | 无 | 待复核 |
| [photo_3484b3a02c37703c](photo_3484b3a02c37703c/photo_3484b3a02c37703c/REPORT.md) | 尚未人工分类 | 71 | 2 | -0.004534 | +0.014934 | photo_12f7c257ae1f8f3c | 待复核 |
| [photo_35d5f9aaea72e94c](photo_35d5f9aaea72e94c/photo_35d5f9aaea72e94c/REPORT.md) | 尚未人工分类 | 3 | 66 | +0.007256 | +0.201351 | photo_ca454f8a9d866abb | 待复核 |
| [photo_3657bb9dffa08975](photo_3657bb9dffa08975/photo_3657bb9dffa08975/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.011478 | +0.022290 | photo_42f6f613cad79d1c | 待复核 |
| [photo_37c9ab8958ab73bc](photo_37c9ab8958ab73bc/photo_37c9ab8958ab73bc/REPORT.md) | 尚未人工分类 | 355 | 2 | -0.059276 | -0.075006 | photo_375b67daeeeeff5f | 待复核 |
| [photo_38ab72e39bb67515](photo_38ab72e39bb67515/photo_38ab72e39bb67515/REPORT.md) | 尚未人工分类 | 2167 | 5 | -0.008881 | -0.015382 | photo_cdd1aa422a4f6dc0 | 待复核 |
| [photo_39034fd7db4f883a](photo_39034fd7db4f883a/photo_39034fd7db4f883a/REPORT.md) | 尚未人工分类 | 4399 | 360 | -0.023352 | +0.003950 | photo_2df687d8a2f89f40 | 待复核 |
| [photo_3ae9218d84c7a2d5](photo_3ae9218d84c7a2d5/photo_3ae9218d84c7a2d5/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.006099 | +0.003515 | photo_72ad5f2e7435bc37 | 待复核 |
| [photo_3ea432cd519827df](photo_3ea432cd519827df/photo_3ea432cd519827df/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.001768 | +0.008179 | photo_20594074f229bd8f | 待复核 |
| [photo_3fa0e51f2f450f84](photo_3fa0e51f2f450f84/photo_3fa0e51f2f450f84/REPORT.md) | 尚未人工分类 | 1 | 3 | +0.001735 | -0.000031 | photo_51f740065ab2be47 | 待复核 |
| [photo_3fa448ba98849bb7](photo_3fa448ba98849bb7/photo_3fa448ba98849bb7/REPORT.md) | 尚未人工分类 | 4 | 3 | -0.010903 | -0.019116 | photo_cf724bd3f6c03346 | 待复核 |
| [photo_3ff837bd4edbded5](photo_3ff837bd4edbded5/photo_3ff837bd4edbded5/REPORT.md) | 尚未人工分类 | 2 | 4 | +0.005321 | +0.035693 | photo_80eb2d50ed90738e | 待复核 |
| [photo_4060facca326e13d](photo_4060facca326e13d/photo_4060facca326e13d/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.000895 | +0.021030 | 无 | 待复核 |
| [photo_409c0caca9a1be53](photo_409c0caca9a1be53/photo_409c0caca9a1be53/REPORT.md) | 尚未人工分类 | 4 | 4 | +0.024483 | +0.105230 | 无 | 待复核 |
| [photo_40d4735eb62e06cf](photo_40d4735eb62e06cf/photo_40d4735eb62e06cf/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.010135 | -0.009820 | photo_7247f008f2067eb2 | 待复核 |
| [photo_41de3eb383934ec3](photo_41de3eb383934ec3/photo_41de3eb383934ec3/REPORT.md) | 换行与末尾词位置 | 2 | 2 | -0.004094 | -0.015571 | photo_20594074f229bd8f | 待复核 |
| [photo_423dd2b26155876d](photo_423dd2b26155876d/photo_423dd2b26155876d/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.012200 | +0.030711 | photo_7e82db4d268e6916 | 待复核 |
| [photo_438026ead38bb48f](photo_438026ead38bb48f/photo_438026ead38bb48f/REPORT.md) | 尚未人工分类 | 3 | 6 | -0.034037 | -0.031887 | 无 | 待复核 |
| [photo_441809a846ca8a24](photo_441809a846ca8a24/photo_441809a846ca8a24/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.013820 | -0.019864 | photo_12144e4f9cd970e7 | 待复核 |
| [photo_449ade96538a6b03](photo_449ade96538a6b03/photo_449ade96538a6b03/REPORT.md) | 尚未人工分类 | 9 | 9 | -0.008797 | -0.002584 | 无 | 身份歧义 |
| [photo_45a3da8e4498856f](photo_45a3da8e4498856f/photo_45a3da8e4498856f/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.047164 | +0.013660 | photo_47430c79b3238e4a | 待复核 |
| [photo_45b5970f1ebe4f19](photo_45b5970f1ebe4f19/photo_45b5970f1ebe4f19/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.001765 | +0.004502 | photo_bbbf0bcaee0cc855 | 待复核 |
| [photo_46e147c8cea37fda](photo_46e147c8cea37fda/photo_46e147c8cea37fda/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.008891 | -0.023575 | 无 | 待复核 |
| [photo_487b4a6b018c220d](photo_487b4a6b018c220d/photo_487b4a6b018c220d/REPORT.md) | 尚未人工分类 | 524 | 269 | +0.016816 | +0.017715 | photo_1089688627444cf6 | 待复核 |
| [photo_4920adb7ad190a5f](photo_4920adb7ad190a5f/photo_4920adb7ad190a5f/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.018254 | +0.028812 | photo_2d3809ffa08d826a | 待复核 |
| [photo_4b09c283ed00faca](photo_4b09c283ed00faca/photo_4b09c283ed00faca/REPORT.md) | 尚未人工分类 | 290 | 6 | -0.014228 | -0.030329 | photo_80eb2d50ed90738e | 待复核 |
| [photo_50614555bf86aaf4](photo_50614555bf86aaf4/photo_50614555bf86aaf4/REPORT.md) | 尚未人工分类 | 64 | 12 | -0.052627 | -0.073011 | photo_1089688627444cf6 | 待复核 |
| [photo_51b91a8efbda2ee9](photo_51b91a8efbda2ee9/photo_51b91a8efbda2ee9/REPORT.md) | 尚未人工分类 | 117 | 4 | -0.038539 | -0.053530 | photo_792997f8dbfba097 | 待复核 |
| [photo_52547f944821ac72](photo_52547f944821ac72/photo_52547f944821ac72/REPORT.md) | 尚未人工分类 | 15 | 3 | -0.037965 | -0.051601 | photo_2df687d8a2f89f40 | 待复核 |
| [photo_529ca7857c6387ae](photo_529ca7857c6387ae/photo_529ca7857c6387ae/REPORT.md) | 尚未人工分类 | 4 | 4 | -0.008346 | -0.011006 | photo_16f724e075d1e714 | 身份歧义 |
| [photo_55b61213a0c8f1a9](photo_55b61213a0c8f1a9/photo_55b61213a0c8f1a9/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.000750 | -0.000919 | photo_184d6f956513c5d8 | 身份歧义 |
| [photo_578a5c367aa237fe](photo_578a5c367aa237fe/photo_578a5c367aa237fe/REPORT.md) | 尚未人工分类 | 16 | 2 | -0.025228 | -0.027934 | photo_b111a943691a2aa7 | 身份歧义 |
| [photo_58545c26f42820eb](photo_58545c26f42820eb/photo_58545c26f42820eb/REPORT.md) | 尚未人工分类 | 2 | 3 | -0.004665 | -0.013990 | photo_1d2280603a42552a | 待复核 |
| [photo_5964b5e1bd6ef958](photo_5964b5e1bd6ef958/photo_5964b5e1bd6ef958/REPORT.md) | 尚未人工分类 | 657 | 18 | -0.069397 | -0.005519 | photo_9872ed71c7af7565 | 待复核 |
| [photo_59d5e8c638df29f0](photo_59d5e8c638df29f0/photo_59d5e8c638df29f0/REPORT.md) | 尚未人工分类 | 114 | 23 | -0.067503 | -0.078963 | photo_4f2addadc5937695 | 待复核 |
| [photo_5b21aa76cc4c1a37](photo_5b21aa76cc4c1a37/photo_5b21aa76cc4c1a37/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.006172 | -0.004491 | photo_72ad5f2e7435bc37 | 待复核 |
| [photo_5ec236299f8266c9](photo_5ec236299f8266c9/photo_5ec236299f8266c9/REPORT.md) | 尚未人工分类 | 1 | 6 | +0.011187 | +0.116277 | 无 | 待复核 |
| [photo_5f0ddbb68434bc39](photo_5f0ddbb68434bc39/photo_5f0ddbb68434bc39/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.058574 | +0.207384 | photo_53000d6acda74399 | 待复核 |
| [photo_60f0c3cd66c76c52](photo_60f0c3cd66c76c52/photo_60f0c3cd66c76c52/REPORT.md) | 尚未人工分类 | 1 | 3 | +0.000638 | -0.014943 | photo_9a8d63f5711209a8 | 待复核 |
| [photo_6172bb3bbbb3a2d6](photo_6172bb3bbbb3a2d6/photo_6172bb3bbbb3a2d6/REPORT.md) | 尚未人工分类 | 10 | 8 | -0.010434 | -0.028988 | 无 | 身份歧义 |
| [photo_63d8b363d3a72f0f](photo_63d8b363d3a72f0f/photo_63d8b363d3a72f0f/REPORT.md) | 尚未人工分类 | 4 | 2 | -0.001206 | -0.006386 | photo_ab6fd7a33337ccee | 待复核 |
| [photo_64625e0a749be4a5](photo_64625e0a749be4a5/photo_64625e0a749be4a5/REPORT.md) | 尚未人工分类 | 19 | 4 | -0.010822 | -0.011775 | photo_80eb2d50ed90738e | 待复核 |
| [photo_65de2e175e6cbd9d](photo_65de2e175e6cbd9d/photo_65de2e175e6cbd9d/REPORT.md) | 尚未人工分类 | 5 | 2 | -0.000846 | -0.001078 | photo_7f4bc7d166b40bfc | 待复核 |
| [photo_65fa10852c465cb1](photo_65fa10852c465cb1/photo_65fa10852c465cb1/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.011496 | +0.016909 | 无 | 待复核 |
| [photo_665b96b3fb0b01c7](photo_665b96b3fb0b01c7/photo_665b96b3fb0b01c7/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.009928 | -0.004483 | photo_20594074f229bd8f | 待复核 |
| [photo_6f1045004ede2c6e](photo_6f1045004ede2c6e/photo_6f1045004ede2c6e/REPORT.md) | 尚未人工分类 | 11 | 2 | -0.001505 | -0.000904 | photo_51f7a3bb218030b8 | 身份歧义 |
| [photo_740025f501c6528b](photo_740025f501c6528b/photo_740025f501c6528b/REPORT.md) | 尚未人工分类 | 160 | 21 | -0.064999 | -0.062543 | 无 | 待复核 |
| [photo_7b5ffe8d495ed220](photo_7b5ffe8d495ed220/photo_7b5ffe8d495ed220/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.001358 | +0.001526 | photo_20594074f229bd8f | 待复核 |
| [photo_7c0c3a5cb514ea44](photo_7c0c3a5cb514ea44/photo_7c0c3a5cb514ea44/REPORT.md) | 尚未人工分类 | 2 | 2 | +0.001054 | +0.010509 | photo_1b6a28535da238b9 | 身份歧义 |
| [photo_7e0b721a9c25b555](photo_7e0b721a9c25b555/photo_7e0b721a9c25b555/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.008533 | +0.001896 | 无 | 待复核 |
| [photo_7e10c5317a5e74d8](photo_7e10c5317a5e74d8/photo_7e10c5317a5e74d8/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.001255 | -0.001308 | photo_ca454f8a9d866abb | 待复核 |
| [photo_7e79bf355aa7c38b](photo_7e79bf355aa7c38b/photo_7e79bf355aa7c38b/REPORT.md) | 尚未人工分类 | 1574 | 19 | -0.041130 | +0.021792 | photo_41892aea27698341 | 待复核 |
| [photo_7eb4a41f278c6e96](photo_7eb4a41f278c6e96/photo_7eb4a41f278c6e96/REPORT.md) | 尚未人工分类 | 4 | 4 | -0.032804 | -0.042508 | 无 | 待复核 |
| [photo_7ee8dcc5521c64c9](photo_7ee8dcc5521c64c9/photo_7ee8dcc5521c64c9/REPORT.md) | 尚未人工分类 | 166 | 9 | -0.085926 | -0.088677 | photo_441c254874dcdd8d | 待复核 |
| [photo_7f0add741b1b243c](photo_7f0add741b1b243c/photo_7f0add741b1b243c/REPORT.md) | 尚未人工分类 | 2 | 4 | +0.013277 | +0.251554 | photo_709e8a3ee42539bc | 待复核 |
| [photo_7f3f0a27f2d5c4c3](photo_7f3f0a27f2d5c4c3/photo_7f3f0a27f2d5c4c3/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.007253 | -0.005829 | photo_ab6fd7a33337ccee | 待复核 |
| [photo_80f2fdc29081fedd](photo_80f2fdc29081fedd/photo_80f2fdc29081fedd/REPORT.md) | 尚未人工分类 | 297 | 22 | -0.031719 | -0.038055 | photo_59498b6acfbfbf4f | 待复核 |
| [photo_813fb072c0f3b380](photo_813fb072c0f3b380/photo_813fb072c0f3b380/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.017577 | -0.015735 | 无 | 待复核 |
| [photo_817005ee11887c2a](photo_817005ee11887c2a/photo_817005ee11887c2a/REPORT.md) | 尚未人工分类 | 9 | 8 | -0.034319 | -0.072718 | photo_4917ad5f3fac9f55 | 待复核 |
| [photo_831ae64dd7297736](photo_831ae64dd7297736/photo_831ae64dd7297736/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.006583 | +0.026738 | photo_74c95fac5a37018f | 待复核 |
| [photo_83941d6fe68296bd](photo_83941d6fe68296bd/photo_83941d6fe68296bd/REPORT.md) | 尚未人工分类 | 600 | 110 | -0.033490 | +0.055018 | photo_ab6fd7a33337ccee | 待复核 |
| [photo_85e10feb92319a7b](photo_85e10feb92319a7b/photo_85e10feb92319a7b/REPORT.md) | 尚未人工分类 | 16 | 2 | -0.021370 | -0.003606 | photo_1b6a28535da238b9 | 身份歧义 |
| [photo_8788bcb16ad7b804](photo_8788bcb16ad7b804/photo_8788bcb16ad7b804/REPORT.md) | 尚未人工分类 | 6 | 2 | -0.002848 | -0.007152 | photo_184d6f956513c5d8 | 身份歧义 |
| [photo_8a62e5cd6902b0ae](photo_8a62e5cd6902b0ae/photo_8a62e5cd6902b0ae/REPORT.md) | 尚未人工分类 | 4 | 5 | -0.015964 | -0.026304 | 无 | 待复核 |
| [photo_8bc3a7a3577518f0](photo_8bc3a7a3577518f0/photo_8bc3a7a3577518f0/REPORT.md) | 尚未人工分类 | 1004 | 16 | -0.157122 | -0.205972 | photo_441c254874dcdd8d | 待复核 |
| [photo_8c54b7644f2e05b2](photo_8c54b7644f2e05b2/photo_8c54b7644f2e05b2/REPORT.md) | 尚未人工分类 | 10 | 2 | -0.014849 | -0.026494 | photo_709e8a3ee42539bc | 待复核 |
| [photo_900c769d86302c63](photo_900c769d86302c63/photo_900c769d86302c63/REPORT.md) | 尚未人工分类 | 19 | 19 | -0.021619 | -0.006776 | photo_7247f008f2067eb2 | 待复核 |
| [photo_92b44955f8a40dbb](photo_92b44955f8a40dbb/photo_92b44955f8a40dbb/REPORT.md) | 尚未人工分类 | 1 | 4 | +0.040558 | +0.304604 | photo_420054ef52bb8e29 | 待复核 |
| [photo_97b88c7af1923448](photo_97b88c7af1923448/photo_97b88c7af1923448/REPORT.md) | 尚未人工分类 | 1 | 4 | +0.000771 | +0.020436 | photo_d1348f6b31b826fa | 待复核 |
| [photo_9bb993b443dfe2ff](photo_9bb993b443dfe2ff/photo_9bb993b443dfe2ff/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.013901 | -0.012453 | photo_77b67ea38a19f6b3 | 待复核 |
| [photo_9e0c7038dd6543a7](photo_9e0c7038dd6543a7/photo_9e0c7038dd6543a7/REPORT.md) | 尚未人工分类 | 13 | 9 | -0.004728 | -0.001329 | photo_1ba61c7d4c7edb4b | 待复核 |
| [photo_9e6d02a08ae58eae](photo_9e6d02a08ae58eae/photo_9e6d02a08ae58eae/REPORT.md) | 尚未人工分类 | 3 | 2 | -0.011336 | -0.013793 | photo_16f724e075d1e714 | 身份歧义 |
| [photo_9f799473a3f5aacc](photo_9f799473a3f5aacc/photo_9f799473a3f5aacc/REPORT.md) | 尚未人工分类 | 2 | 2 | +0.032642 | +0.036467 | photo_9fbc291d90193703 | 身份歧义 |
| [photo_a0dda9c59f0160fe](photo_a0dda9c59f0160fe/photo_a0dda9c59f0160fe/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.018815 | -0.021652 | photo_80eb2d50ed90738e | 待复核 |
| [photo_a1dba542121735f2](photo_a1dba542121735f2/photo_a1dba542121735f2/REPORT.md) | 尚未人工分类 | 14 | 14 | -0.010177 | +0.010464 | photo_176abc65485427c3 | 待复核 |
| [photo_a2020c21aa08ba7b](photo_a2020c21aa08ba7b/photo_a2020c21aa08ba7b/REPORT.md) | 尚未人工分类 | 9 | 3 | -0.021175 | -0.034928 | photo_47430c79b3238e4a | 待复核 |
| [photo_a7b92b4cabf1396a](photo_a7b92b4cabf1396a/photo_a7b92b4cabf1396a/REPORT.md) | 尚未人工分类 | 8 | 36 | +0.013100 | +0.016090 | photo_a018168fa8c88d4c | 待复核 |
| [photo_aabee66f53cb53e8](photo_aabee66f53cb53e8/photo_aabee66f53cb53e8/REPORT.md) | 尚未人工分类 | 3 | 4 | -0.010339 | +0.041143 | photo_80eb2d50ed90738e | 待复核 |
| [photo_acbddc48df8f77fd](photo_acbddc48df8f77fd/photo_acbddc48df8f77fd/REPORT.md) | 尚未人工分类 | 112 | 4 | -0.004048 | +0.011402 | photo_6e3ed6cfb297f29c | 待复核 |
| [photo_ace6fcbc6d37331a](photo_ace6fcbc6d37331a/photo_ace6fcbc6d37331a/REPORT.md) | 尚未人工分类 | 855 | 2 | -0.030474 | -0.008933 | photo_60b643cb7fb93b4e | 待复核 |
| [photo_adfc445d03f6aab6](photo_adfc445d03f6aab6/photo_adfc445d03f6aab6/REPORT.md) | 尚未人工分类 | 1 | 4 | +0.007559 | +0.018811 | 无 | 待复核 |
| [photo_b080c24ec8e8e236](photo_b080c24ec8e8e236/photo_b080c24ec8e8e236/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.003187 | +0.016929 | photo_2d3809ffa08d826a | 待复核 |
| [photo_b35bafd0e0a8761b](photo_b35bafd0e0a8761b/photo_b35bafd0e0a8761b/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.012761 | +0.049062 | photo_b5df0dce47884a73 | 待复核 |
| [photo_bc13fbd48065accd](photo_bc13fbd48065accd/photo_bc13fbd48065accd/REPORT.md) | 尚未人工分类 | 1 | 4 | +0.016410 | +0.020892 | 无 | 待复核 |
| [photo_bc5564b0e01eeb18](photo_bc5564b0e01eeb18/photo_bc5564b0e01eeb18/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.001996 | -0.005083 | photo_20594074f229bd8f | 待复核 |
| [photo_bfe7ecdd06599257](photo_bfe7ecdd06599257/photo_bfe7ecdd06599257/REPORT.md) | 尚未人工分类 | 38 | 4 | -0.056442 | -0.066048 | photo_38d1d33620ab1ea1 | 待复核 |
| [photo_c0babaac8a7306cd](photo_c0babaac8a7306cd/photo_c0babaac8a7306cd/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.022234 | +0.032386 | photo_6d830d08fb9cad94 | 待复核 |
| [photo_c49cbdcb940ee7ee](photo_c49cbdcb940ee7ee/photo_c49cbdcb940ee7ee/REPORT.md) | 尚未人工分类 | 74 | 49 | +0.019054 | +0.243454 | photo_2df687d8a2f89f40 | 待复核 |
| [photo_c576c0c30f74cd4f](photo_c576c0c30f74cd4f/photo_c576c0c30f74cd4f/REPORT.md) | 尚未人工分类 | 10 | 10 | -0.008658 | -0.018250 | 无 | 身份歧义 |
| [photo_c798e089f34805e9](photo_c798e089f34805e9/photo_c798e089f34805e9/REPORT.md) | 尚未人工分类 | 2 | 2 | +0.017477 | +0.044709 | photo_b5df0dce47884a73 | 待复核 |
| [photo_c81d2a7abb5af160](photo_c81d2a7abb5af160/photo_c81d2a7abb5af160/REPORT.md) | 尚未人工分类 | 5 | 7 | -0.000335 | -0.004631 | photo_0eb729e9f7b895d9 | 待复核 |
| [photo_c86676a1232aefc7](photo_c86676a1232aefc7/photo_c86676a1232aefc7/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.015224 | +0.012364 | photo_cccdde216b52603a | 待复核 |
| [photo_c8920e14924534af](photo_c8920e14924534af/photo_c8920e14924534af/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.007082 | -0.022312 | photo_bbbf0bcaee0cc855 | 待复核 |
| [photo_c8a330b5025d96f5](photo_c8a330b5025d96f5/photo_c8a330b5025d96f5/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.006633 | +0.021936 | photo_0d77b404d327bb1b | 待复核 |
| [photo_d2377339b5c178d4](photo_d2377339b5c178d4/photo_d2377339b5c178d4/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.006482 | +0.030441 | photo_80eb2d50ed90738e | 待复核 |
| [photo_d5bb794066061e60](photo_d5bb794066061e60/photo_d5bb794066061e60/REPORT.md) | 尚未人工分类 | 1 | 8 | +0.063360 | +0.207173 | photo_7d401182f8389bc4 | 待复核 |
| [photo_d61e8e0610298d5e](photo_d61e8e0610298d5e/photo_d61e8e0610298d5e/REPORT.md) | 尚未人工分类 | 1 | 8 | +0.049529 | +0.096841 | photo_eba7bd6c17741f92 | 待复核 |
| [photo_d626691df6c29796](photo_d626691df6c29796/photo_d626691df6c29796/REPORT.md) | 尚未人工分类 | 1 | 4 | +0.041698 | +0.044791 | 无 | 待复核 |
| [photo_d637c6c785f0063a](photo_d637c6c785f0063a/photo_d637c6c785f0063a/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.011556 | +0.021544 | 无 | 待复核 |
| [photo_d65410305cff7c7c](photo_d65410305cff7c7c/photo_d65410305cff7c7c/REPORT.md) | 尚未人工分类 | 1 | 4 | +0.005182 | +0.054837 | 无 | 待复核 |
| [photo_da9d9887ba9c878f](photo_da9d9887ba9c878f/photo_da9d9887ba9c878f/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.010933 | -0.018496 | photo_f0d980d139f16444 | 待复核 |
| [photo_daac4b2576b9e9f5](photo_daac4b2576b9e9f5/photo_daac4b2576b9e9f5/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.013929 | +0.027074 | photo_ca454f8a9d866abb | 待复核 |
| [photo_dc89013e88c49cda](photo_dc89013e88c49cda/photo_dc89013e88c49cda/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.109170 | +0.242125 | photo_8ae269507838f198 | 待复核 |
| [photo_e0cb00f622ec06e8](photo_e0cb00f622ec06e8/photo_e0cb00f622ec06e8/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.013033 | +0.021315 | 无 | 待复核 |
| [photo_e32b999979ddc21b](photo_e32b999979ddc21b/photo_e32b999979ddc21b/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.004168 | +0.005068 | photo_20594074f229bd8f | 待复核 |
| [photo_e5b4dba01622bfaf](photo_e5b4dba01622bfaf/photo_e5b4dba01622bfaf/REPORT.md) | 尚未人工分类 | 3 | 2 | -0.007887 | -0.003945 | photo_3cfb7f3d355174a8 | 待复核 |
| [photo_e6160ef7ce3f2f34](photo_e6160ef7ce3f2f34/photo_e6160ef7ce3f2f34/REPORT.md) | 尚未人工分类 | 2 | 2 | -0.002477 | +0.015833 | photo_72ad5f2e7435bc37 | 待复核 |
| [photo_e806395db591b23e](photo_e806395db591b23e/photo_e806395db591b23e/REPORT.md) | 尚未人工分类 | 1 | 7 | +0.037090 | +0.285134 | photo_eb65c801ffc82c76 | 待复核 |
| [photo_ee67048e97d12344](photo_ee67048e97d12344/photo_ee67048e97d12344/REPORT.md) | 尚未人工分类 | 2 | 21 | +0.034440 | +0.048908 | photo_43eca55aded800ef | 待复核 |
| [photo_f3f437a6e3a67796](photo_f3f437a6e3a67796/photo_f3f437a6e3a67796/REPORT.md) | 尚未人工分类 | 7 | 6 | -0.012294 | +0.000092 | photo_4917ad5f3fac9f55 | 待复核 |
| [photo_f58ecf22c1eb6891](photo_f58ecf22c1eb6891/photo_f58ecf22c1eb6891/REPORT.md) | 尚未人工分类 | 1 | 2 | +0.002102 | +0.011250 | 无 | 待复核 |
| [photo_f8f533ac85b05854](photo_f8f533ac85b05854/photo_f8f533ac85b05854/REPORT.md) | 尚未人工分类 | 27 | 6 | -0.007754 | -0.016739 | 无 | 身份歧义 |
| [photo_f9a136cbb2bdc194](photo_f9a136cbb2bdc194/photo_f9a136cbb2bdc194/REPORT.md) | 尚未人工分类 | 3 | 2 | -0.033401 | +0.002579 | photo_1a510aba547e1d64 | 待复核 |
| [photo_fbc7b6d852b1125b](photo_fbc7b6d852b1125b/photo_fbc7b6d852b1125b/REPORT.md) | 尚未人工分类 | 4 | 2 | +0.076381 | +0.118587 | photo_184d6f956513c5d8 | 待复核 |
| [photo_fc0ee05b6b185ffa](photo_fc0ee05b6b185ffa/photo_fc0ee05b6b185ffa/REPORT.md) | 尚未人工分类 | 6 | 9 | -0.032082 | -0.056089 | photo_618866896f180a73 | 待复核 |
| [photo_fd3bb2cdcb3bd3a7](photo_fd3bb2cdcb3bd3a7/photo_fd3bb2cdcb3bd3a7/REPORT.md) | 尚未人工分类 | 24 | 34 | -0.040275 | +0.129251 | photo_d68d74e6a16e45bf | 待复核 |
| [photo_fde389ec0d74a099](photo_fde389ec0d74a099/photo_fde389ec0d74a099/REPORT.md) | 尚未人工分类 | 2181 | 2 | -0.035787 | -0.028185 | photo_4dfba78901c38dc4 | 待复核 |

### 原始特征与复现补充

已补充原始特征、完整候选对相似度矩阵和冻结编码代码。入口：[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
