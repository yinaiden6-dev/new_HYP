# B03：ColNomic 7B 纯图像检索全部 247 个错误

分析的是冻结图像 encoder 与原始 MaxSim 的检索错误。负候选均为纯图像实际 Top-1，逐案核对原错误清单和缓存分数。每例含完整 token 对应、贡献分解、独立几何诊断、焦点文字检查、匹配连线与单独报告。

## 分析范围与对照

| 检查 | 结果 |
| --- | --- |
| 错误案例完成 | 247/247 |
| 目标 token 记录数 | 183629 |
| 同包装成功对照 | 238 |
| 不同成功照片 | 139 |
| 缺少同包装成功对照 | 9 |
| 相同候选对已有身份歧义 | 4 |
| 其他候选曾有身份歧义，未直接转移标签 | 14 |
| 正负几何均通过自动阈值 | 89 |
| 最大缓存评分复现误差 | 1.7881393432617188e-07 |

对照只要求纯图像 Top-1 正确，F03 成功与否不参与选择。重复对照不是独立样本，可见包装面没有逐对确认。

## 根据测量得到的诊断分组

| 诊断 | 数量 |
| --- | --- |
| 主体匹配仍偏错误候选 | 152 |
| 主体筛选可翻转这两个候选 | 91 |
| 同一候选对的身份歧义待复核 | 4 |

固定 token 的主体筛选使 91/247 组正负候选由非正翻正，但也使 3/238 组成功对照的分差变负。这是双候选诊断，不是完整图库 Recall@1 的提升，也不是裁剪图重新编码的效果。

## 划分与 F03 辅助比较

| 划分 | B03 错误数 | 主体筛选双候选翻正 | 在 F03 中正确 |
| --- | --- | --- | --- |
| train | 174 | 66 | 105 |
| val | 37 | 13 | 20 |
| test | 36 | 12 | 18 |

B03 的 247 例中，143 例在 F03 中检索正确，104 例仍错误。本文 token 匹配始终针对 B03 负候选；F03 仅提供已存在的结果对照。

## 先前指定案例

| 案例 | B03 真值排名 | 图像分差 | 仅主体分差 |
| --- | --- | --- | --- |
| [换行与末尾词位置](photo_41de3eb383934ec3/photo_41de3eb383934ec3/REPORT.md) | 2 | -0.004094 | -0.015571 |
| [文字排版结构](photo_1f48d45127db4129/photo_1f48d45127db4129/REPORT.md) | 148 | -0.119856 | -0.148875 |
| [细小文字差异](photo_0e87e0717ab1c215/photo_0e87e0717ab1c215/REPORT.md) | 2 | -0.005469 | -0.007399 |

此前的颜色案例 photo_21bd7df9ed6795e7 在 B03 中正确，不属于这247例；它的 F03 错误不能计入纯图像失败。

## 方法与证据边界

- FP32 点积、有效 query token 的 MaxSim 均值，保留有效模板 token，去掉零向量，不再次归一化；每例验证贡献守恒。
- 原始缓存网格按 EXIF 转到显示坐标再关联人工主体框与 OCR。OCR 交叠阈值为 patch 面积5%，同时记录多行歧义。
- SIFT 独立估计几何：比率 .75、最大边1400、RANSAC 4px、至少8个内点且覆盖主体15%。未通过时不填空间错配结论。
- 匹配图展示主体内最有利／最不利各8个 token，完整数据都在CSV。散点色阶按绝对分差的95%分位裁剪，曲线和表格保留完整数值。
- 数字单位与末尾词区域可能共享同一 patch、包含邻近文字；上下文 token 和矩形框内遮挡也限制语义归因。
- 完成本轮对应与贡献分析并不证明编码器丢失身份信息。编码可分性需要冻结特征 Probe，因果归因需要受控干预。

## 原始数据与复现

[错误候选核查](B03_error_inventory.csv) · [逐案诊断](diagnostics.csv) · [全部对照摘要](cohort_summary.csv) · [焦点文字汇总](focal_lines_all.csv) · [数值与覆盖验证](verification.json) · [完整JSON及缓存签名](summary.json)

```bash
python -B scripts/analyze_colnomic_tokens.py --configuration B03 --all-errors --resume --reuse-from results/token_mechanism_analysis/F03
python -B scripts/report_colnomic_b03_tokens.py
```

## 全部逐案报告

| 案例 | 划分 | 排名 | 图像分差 | 仅主体分差 | 诊断 | 成功对照 |
| --- | --- | --- | --- | --- | --- | --- |
| [photo_007eccaeb41273df](photo_007eccaeb41273df/photo_007eccaeb41273df/REPORT.md) | train | 80 | -0.033210 | -0.021607 | 主体匹配仍偏错误候选 | photo_4917ad5f3fac9f55 |
| [photo_016cf1f638d1cb66](photo_016cf1f638d1cb66/photo_016cf1f638d1cb66/REPORT.md) | val | 8 | -0.005180 | +0.134744 | 主体筛选可翻转这两个候选 | photo_be1805e5ca1f87a1 |
| [photo_01a2f5c5bef357c2](photo_01a2f5c5bef357c2/photo_01a2f5c5bef357c2/REPORT.md) | train | 3 | -0.012711 | -0.017117 | 主体匹配仍偏错误候选 | photo_557f7ff982ed4de2 |
| [photo_01baf6cd7fca88bb](photo_01baf6cd7fca88bb/photo_01baf6cd7fca88bb/REPORT.md) | val | 390 | -0.076721 | -0.097999 | 主体匹配仍偏错误候选 | 无 |
| [photo_0295d53bc16d3fa3](photo_0295d53bc16d3fa3/photo_0295d53bc16d3fa3/REPORT.md) | val | 3 | -0.014952 | -0.028158 | 主体匹配仍偏错误候选 | photo_7bc115b787ac85b4 |
| [photo_02fb3c06e1fb70e5](photo_02fb3c06e1fb70e5/photo_02fb3c06e1fb70e5/REPORT.md) | train | 189 | -0.087715 | +0.060815 | 主体筛选可翻转这两个候选 | photo_80eb2d50ed90738e |
| [photo_04a4b44735fe9ba8](photo_04a4b44735fe9ba8/photo_04a4b44735fe9ba8/REPORT.md) | test | 2 | -0.013976 | -0.017099 | 主体匹配仍偏错误候选 | photo_7e0b721a9c25b555 |
| [photo_0500abf430ca34bc](photo_0500abf430ca34bc/photo_0500abf430ca34bc/REPORT.md) | test | 665 | -0.125407 | -0.138599 | 主体匹配仍偏错误候选 | photo_0a8643c0c2c88704 |
| [photo_062fafe5f481faca](photo_062fafe5f481faca/photo_062fafe5f481faca/REPORT.md) | train | 2 | -0.002851 | -0.008744 | 主体匹配仍偏错误候选 | photo_daac4b2576b9e9f5 |
| [photo_07e45f3cfc816f44](photo_07e45f3cfc816f44/photo_07e45f3cfc816f44/REPORT.md) | test | 48 | -0.031166 | -0.042891 | 主体匹配仍偏错误候选 | photo_77b67ea38a19f6b3 |
| [photo_08e485d58f7a4484](photo_08e485d58f7a4484/photo_08e485d58f7a4484/REPORT.md) | train | 2 | -0.005517 | +0.003155 | 主体筛选可翻转这两个候选 | photo_efcce5600c8aa4cf |
| [photo_0913bcc35e9beb5a](photo_0913bcc35e9beb5a/photo_0913bcc35e9beb5a/REPORT.md) | val | 2216 | -0.107944 | -0.128942 | 主体匹配仍偏错误候选 | photo_e82154bdf3d3c846 |
| [photo_0956810f30cd4324](photo_0956810f30cd4324/photo_0956810f30cd4324/REPORT.md) | test | 3 | -0.002065 | +0.003450 | 主体筛选可翻转这两个候选 | photo_faec4f7b9134de7d |
| [photo_0a0e75a3b7797609](photo_0a0e75a3b7797609/photo_0a0e75a3b7797609/REPORT.md) | train | 13 | -0.018798 | +0.011385 | 主体筛选可翻转这两个候选 | photo_12f7c257ae1f8f3c |
| [photo_0a612f53747b2554](photo_0a612f53747b2554/photo_0a612f53747b2554/REPORT.md) | train | 3 | -0.002902 | -0.019816 | 主体匹配仍偏错误候选 | photo_cdd1aa422a4f6dc0 |
| [photo_0be06814ec4048e3](photo_0be06814ec4048e3/photo_0be06814ec4048e3/REPORT.md) | train | 9 | -0.016813 | -0.024331 | 主体匹配仍偏错误候选 | photo_47430c79b3238e4a |
| [photo_0c63c2a740b336ea](photo_0c63c2a740b336ea/photo_0c63c2a740b336ea/REPORT.md) | val | 7 | -0.011493 | -0.010726 | 主体匹配仍偏错误候选 | photo_2df687d8a2f89f40 |
| [photo_0cfa49485997e1d4](photo_0cfa49485997e1d4/photo_0cfa49485997e1d4/REPORT.md) | test | 296 | -0.075236 | -0.088084 | 主体匹配仍偏错误候选 | photo_7f4bc7d166b40bfc |
| [photo_0e87e0717ab1c215](photo_0e87e0717ab1c215/photo_0e87e0717ab1c215/REPORT.md) | train | 2 | -0.005469 | -0.007399 | 主体匹配仍偏错误候选 | photo_72ad5f2e7435bc37 |
| [photo_0f45e3b0cc440b0a](photo_0f45e3b0cc440b0a/photo_0f45e3b0cc440b0a/REPORT.md) | val | 678 | -0.072381 | -0.084957 | 主体匹配仍偏错误候选 | photo_38d1d33620ab1ea1 |
| [photo_10fccaade3338c73](photo_10fccaade3338c73/photo_10fccaade3338c73/REPORT.md) | train | 2 | -0.003373 | +0.015225 | 主体筛选可翻转这两个候选 | photo_d2377339b5c178d4 |
| [photo_1315223466f5e465](photo_1315223466f5e465/photo_1315223466f5e465/REPORT.md) | train | 3 | -0.003090 | +0.001329 | 主体筛选可翻转这两个候选 | photo_15676d7248e6a493 |
| [photo_1328677cd2f33881](photo_1328677cd2f33881/photo_1328677cd2f33881/REPORT.md) | train | 2 | -0.000308 | +0.009413 | 主体筛选可翻转这两个候选 | photo_1d2280603a42552a |
| [photo_13c1dc19196b1911](photo_13c1dc19196b1911/photo_13c1dc19196b1911/REPORT.md) | test | 34 | -0.030195 | -0.036924 | 主体匹配仍偏错误候选 | photo_7f4bc7d166b40bfc |
| [photo_140ec98de5514fc3](photo_140ec98de5514fc3/photo_140ec98de5514fc3/REPORT.md) | train | 190 | -0.027919 | -0.023796 | 主体匹配仍偏错误候选 | photo_0cf112e57c488bf5 |
| [photo_158cee75554e9c49](photo_158cee75554e9c49/photo_158cee75554e9c49/REPORT.md) | train | 4 | -0.008864 | +0.001707 | 主体筛选可翻转这两个候选 | photo_587142eeed6488be |
| [photo_158d2a9a0ee0d554](photo_158d2a9a0ee0d554/photo_158d2a9a0ee0d554/REPORT.md) | train | 7 | -0.009051 | -0.014640 | 主体匹配仍偏错误候选 | photo_e96b3f54227d2e31 |
| [photo_15ce8c42c548e678](photo_15ce8c42c548e678/photo_15ce8c42c548e678/REPORT.md) | train | 14 | -0.022750 | +0.005124 | 主体筛选可翻转这两个候选 | photo_12ddea4ffa36f7a8 |
| [photo_16ff2e9baedfa1da](photo_16ff2e9baedfa1da/photo_16ff2e9baedfa1da/REPORT.md) | train | 10 | -0.011895 | -0.013991 | 主体匹配仍偏错误候选 | photo_51f7a3bb218030b8 |
| [photo_19aa4d08f51deacc](photo_19aa4d08f51deacc/photo_19aa4d08f51deacc/REPORT.md) | val | 8 | -0.024982 | -0.007058 | 主体匹配仍偏错误候选 | photo_049efdffe7a60987 |
| [photo_1a3a7ee95f631a62](photo_1a3a7ee95f631a62/photo_1a3a7ee95f631a62/REPORT.md) | train | 38 | -0.016956 | +0.029205 | 主体筛选可翻转这两个候选 | photo_bc13fbd48065accd |
| [photo_1c020a9f93e2a55c](photo_1c020a9f93e2a55c/photo_1c020a9f93e2a55c/REPORT.md) | train | 2 | -0.006349 | +0.012814 | 主体筛选可翻转这两个候选 | photo_8c6978ca27f16e82 |
| [photo_1caebee07e99d064](photo_1caebee07e99d064/photo_1caebee07e99d064/REPORT.md) | train | 2 | -0.001077 | +0.036282 | 主体筛选可翻转这两个候选 | photo_174967c2577c6dc9 |
| [photo_1f48d45127db4129](photo_1f48d45127db4129/photo_1f48d45127db4129/REPORT.md) | train | 148 | -0.119856 | -0.148875 | 主体匹配仍偏错误候选 | photo_02c11dd14bf4116f |
| [photo_206a9b7d61884264](photo_206a9b7d61884264/photo_206a9b7d61884264/REPORT.md) | val | 10 | -0.011764 | -0.020640 | 主体匹配仍偏错误候选 | 无 |
| [photo_226e28640d388e04](photo_226e28640d388e04/photo_226e28640d388e04/REPORT.md) | train | 2 | -0.010319 | -0.007135 | 主体匹配仍偏错误候选 | photo_72ad5f2e7435bc37 |
| [photo_22e9695d01cb05c1](photo_22e9695d01cb05c1/photo_22e9695d01cb05c1/REPORT.md) | val | 20 | -0.010427 | -0.015218 | 主体匹配仍偏错误候选 | photo_2df687d8a2f89f40 |
| [photo_24aba52659f86e92](photo_24aba52659f86e92/photo_24aba52659f86e92/REPORT.md) | train | 4 | -0.006859 | -0.011880 | 主体匹配仍偏错误候选 | photo_73d46bde9ac4b4c5 |
| [photo_264a39405899b54d](photo_264a39405899b54d/photo_264a39405899b54d/REPORT.md) | train | 4 | -0.002339 | -0.001920 | 主体匹配仍偏错误候选 | photo_3bf9d27254c310cf |
| [photo_268216a578a8bed1](photo_268216a578a8bed1/photo_268216a578a8bed1/REPORT.md) | train | 2 | -0.009466 | +0.001810 | 主体筛选可翻转这两个候选 | photo_1d2280603a42552a |
| [photo_298abe6eb119b64d](photo_298abe6eb119b64d/photo_298abe6eb119b64d/REPORT.md) | test | 14 | -0.006939 | +0.228069 | 主体筛选可翻转这两个候选 | photo_709e8a3ee42539bc |
| [photo_2b13d05bc35513ee](photo_2b13d05bc35513ee/photo_2b13d05bc35513ee/REPORT.md) | val | 2 | -0.000044 | +0.016404 | 主体筛选可翻转这两个候选 | photo_7bc115b787ac85b4 |
| [photo_2bb7a4894380d509](photo_2bb7a4894380d509/photo_2bb7a4894380d509/REPORT.md) | val | 10 | -0.017177 | -0.010091 | 主体匹配仍偏错误候选 | 无 |
| [photo_2bdbf879ee1d7dd8](photo_2bdbf879ee1d7dd8/photo_2bdbf879ee1d7dd8/REPORT.md) | test | 2 | -0.008512 | -0.021425 | 主体匹配仍偏错误候选 | photo_7e0b721a9c25b555 |
| [photo_2d06e466a2c40697](photo_2d06e466a2c40697/photo_2d06e466a2c40697/REPORT.md) | train | 3 | -0.006731 | -0.012616 | 主体匹配仍偏错误候选 | photo_613ba5f27d0c5cfe |
| [photo_2d1d66f5ae3b249e](photo_2d1d66f5ae3b249e/photo_2d1d66f5ae3b249e/REPORT.md) | train | 2 | -0.013258 | +0.003664 | 主体筛选可翻转这两个候选 | photo_72ad5f2e7435bc37 |
| [photo_2db3d27da013553c](photo_2db3d27da013553c/photo_2db3d27da013553c/REPORT.md) | train | 29 | -0.010673 | -0.002710 | 主体匹配仍偏错误候选 | photo_8bd2c2ece5146d67 |
| [photo_2eef3175a4bb80d2](photo_2eef3175a4bb80d2/photo_2eef3175a4bb80d2/REPORT.md) | train | 2 | -0.012020 | -0.007098 | 主体匹配仍偏错误候选 | photo_4f3036c68b1b3b4e |
| [photo_2f4051894f10c596](photo_2f4051894f10c596/photo_2f4051894f10c596/REPORT.md) | train | 3 | -0.000829 | -0.002686 | 主体匹配仍偏错误候选 | photo_8ae269507838f198 |
| [photo_3035bdfaed71d4db](photo_3035bdfaed71d4db/photo_3035bdfaed71d4db/REPORT.md) | test | 44 | -0.025797 | -0.029153 | 主体匹配仍偏错误候选 | photo_7f4bc7d166b40bfc |
| [photo_312d9f59eef6563d](photo_312d9f59eef6563d/photo_312d9f59eef6563d/REPORT.md) | val | 690 | -0.066428 | -0.071955 | 主体匹配仍偏错误候选 | 无 |
| [photo_31521050e001b9b4](photo_31521050e001b9b4/photo_31521050e001b9b4/REPORT.md) | val | 23 | -0.030820 | +0.100287 | 主体筛选可翻转这两个候选 | photo_53000d6acda74399 |
| [photo_31ef5fd43cac8585](photo_31ef5fd43cac8585/photo_31ef5fd43cac8585/REPORT.md) | val | 2 | -0.003256 | +0.006950 | 主体筛选可翻转这两个候选 | photo_831ae64dd7297736 |
| [photo_31ff4c6250aab6d5](photo_31ff4c6250aab6d5/photo_31ff4c6250aab6d5/REPORT.md) | train | 23 | -0.024041 | -0.003965 | 主体匹配仍偏错误候选 | photo_6b934fda731eb0bb |
| [photo_324805558f8eab41](photo_324805558f8eab41/photo_324805558f8eab41/REPORT.md) | train | 1711 | -0.057280 | -0.013147 | 主体匹配仍偏错误候选 | photo_71a0dab1e3a6a114 |
| [photo_3409192d1c80c20f](photo_3409192d1c80c20f/photo_3409192d1c80c20f/REPORT.md) | train | 2 | -0.001937 | -0.011555 | 主体匹配仍偏错误候选 | photo_e564c98bb10f3824 |
| [photo_3484b3a02c37703c](photo_3484b3a02c37703c/photo_3484b3a02c37703c/REPORT.md) | train | 71 | -0.046630 | -0.014815 | 主体匹配仍偏错误候选 | photo_12f7c257ae1f8f3c |
| [photo_3487dd8994f438f3](photo_3487dd8994f438f3/photo_3487dd8994f438f3/REPORT.md) | train | 5 | -0.006567 | +0.010597 | 主体筛选可翻转这两个候选 | photo_c0ed1211516e6abd |
| [photo_34a99820aa8a556f](photo_34a99820aa8a556f/photo_34a99820aa8a556f/REPORT.md) | train | 2 | -0.008523 | +0.004703 | 主体筛选可翻转这两个候选 | photo_ee5f8169ea7ffcd2 |
| [photo_35d5f9aaea72e94c](photo_35d5f9aaea72e94c/photo_35d5f9aaea72e94c/REPORT.md) | train | 3 | -0.011568 | -0.010320 | 主体匹配仍偏错误候选 | photo_ca454f8a9d866abb |
| [photo_368eda8ed55ea25d](photo_368eda8ed55ea25d/photo_368eda8ed55ea25d/REPORT.md) | train | 3 | -0.006430 | -0.008646 | 主体匹配仍偏错误候选 | photo_f0d980d139f16444 |
| [photo_37c9ab8958ab73bc](photo_37c9ab8958ab73bc/photo_37c9ab8958ab73bc/REPORT.md) | train | 355 | -0.059276 | -0.075006 | 主体匹配仍偏错误候选 | photo_375b67daeeeeff5f |
| [photo_38ab72e39bb67515](photo_38ab72e39bb67515/photo_38ab72e39bb67515/REPORT.md) | train | 2167 | -0.060726 | -0.062719 | 主体匹配仍偏错误候选 | photo_cdd1aa422a4f6dc0 |
| [photo_39034fd7db4f883a](photo_39034fd7db4f883a/photo_39034fd7db4f883a/REPORT.md) | val | 4399 | -0.044470 | -0.046554 | 主体匹配仍偏错误候选 | photo_2df687d8a2f89f40 |
| [photo_390d64929b6faaa9](photo_390d64929b6faaa9/photo_390d64929b6faaa9/REPORT.md) | train | 18 | -0.015624 | -0.004694 | 主体匹配仍偏错误候选 | photo_60a129173d7c7794 |
| [photo_398328fbee9cdb14](photo_398328fbee9cdb14/photo_398328fbee9cdb14/REPORT.md) | test | 2 | -0.001074 | +0.010887 | 主体筛选可翻转这两个候选 | photo_fc3e4bf8506bc27b |
| [photo_3ae9218d84c7a2d5](photo_3ae9218d84c7a2d5/photo_3ae9218d84c7a2d5/REPORT.md) | train | 2 | -0.006099 | +0.003515 | 主体筛选可翻转这两个候选 | photo_72ad5f2e7435bc37 |
| [photo_3b065aea3dabd5cd](photo_3b065aea3dabd5cd/photo_3b065aea3dabd5cd/REPORT.md) | train | 11 | -0.012551 | -0.015775 | 主体匹配仍偏错误候选 | photo_26a6aec8d98e13fd |
| [photo_3ea432cd519827df](photo_3ea432cd519827df/photo_3ea432cd519827df/REPORT.md) | test | 2 | -0.001768 | +0.008179 | 主体筛选可翻转这两个候选 | photo_20594074f229bd8f |
| [photo_3ed4da0dca991608](photo_3ed4da0dca991608/photo_3ed4da0dca991608/REPORT.md) | train | 3 | -0.007663 | +0.031957 | 主体筛选可翻转这两个候选 | photo_567000412151b626 |
| [photo_3fa448ba98849bb7](photo_3fa448ba98849bb7/photo_3fa448ba98849bb7/REPORT.md) | train | 4 | -0.010903 | -0.019116 | 主体匹配仍偏错误候选 | photo_cf724bd3f6c03346 |
| [photo_3ff837bd4edbded5](photo_3ff837bd4edbded5/photo_3ff837bd4edbded5/REPORT.md) | train | 2 | -0.000154 | +0.046528 | 主体筛选可翻转这两个候选 | photo_d2377339b5c178d4 |
| [photo_409c0caca9a1be53](photo_409c0caca9a1be53/photo_409c0caca9a1be53/REPORT.md) | train | 4 | -0.010410 | -0.005710 | 主体匹配仍偏错误候选 | photo_d65410305cff7c7c |
| [photo_40d4735eb62e06cf](photo_40d4735eb62e06cf/photo_40d4735eb62e06cf/REPORT.md) | train | 2 | -0.010135 | -0.009820 | 主体匹配仍偏错误候选 | photo_7247f008f2067eb2 |
| [photo_41de3eb383934ec3](photo_41de3eb383934ec3/photo_41de3eb383934ec3/REPORT.md) | test | 2 | -0.004094 | -0.015571 | 主体匹配仍偏错误候选 | photo_20594074f229bd8f |
| [photo_438026ead38bb48f](photo_438026ead38bb48f/photo_438026ead38bb48f/REPORT.md) | train | 3 | -0.034037 | -0.031887 | 主体匹配仍偏错误候选 | photo_079284acc905a73f |
| [photo_441809a846ca8a24](photo_441809a846ca8a24/photo_441809a846ca8a24/REPORT.md) | train | 2 | -0.013820 | -0.019864 | 主体匹配仍偏错误候选 | photo_12144e4f9cd970e7 |
| [photo_449ade96538a6b03](photo_449ade96538a6b03/photo_449ade96538a6b03/REPORT.md) | val | 9 | -0.008797 | -0.002584 | 同一候选对的身份歧义待复核 | 无 |
| [photo_46e147c8cea37fda](photo_46e147c8cea37fda/photo_46e147c8cea37fda/REPORT.md) | test | 2 | -0.008891 | -0.023575 | 主体匹配仍偏错误候选 | photo_7e0b721a9c25b555 |
| [photo_487b4a6b018c220d](photo_487b4a6b018c220d/photo_487b4a6b018c220d/REPORT.md) | train | 524 | -0.057249 | -0.046210 | 主体匹配仍偏错误候选 | photo_1089688627444cf6 |
| [photo_48eba161f8426074](photo_48eba161f8426074/photo_48eba161f8426074/REPORT.md) | train | 8 | -0.015293 | +0.007646 | 主体筛选可翻转这两个候选 | photo_01340abaa6725c78 |
| [photo_49940e68a23b618f](photo_49940e68a23b618f/photo_49940e68a23b618f/REPORT.md) | train | 29 | -0.029812 | -0.033105 | 主体匹配仍偏错误候选 | photo_ca454f8a9d866abb |
| [photo_4a578988d89f5973](photo_4a578988d89f5973/photo_4a578988d89f5973/REPORT.md) | test | 2 | -0.001344 | -0.007521 | 主体匹配仍偏错误候选 | photo_85d3b99db68ac04f |
| [photo_4b09c283ed00faca](photo_4b09c283ed00faca/photo_4b09c283ed00faca/REPORT.md) | train | 290 | -0.090015 | +0.029573 | 主体筛选可翻转这两个候选 | photo_80eb2d50ed90738e |
| [photo_4bb2cadff5e2a61c](photo_4bb2cadff5e2a61c/photo_4bb2cadff5e2a61c/REPORT.md) | train | 2 | -0.003571 | +0.051123 | 主体筛选可翻转这两个候选 | photo_60258955de39a64a |
| [photo_4bb5f19e9b60fe8c](photo_4bb5f19e9b60fe8c/photo_4bb5f19e9b60fe8c/REPORT.md) | train | 1253 | -0.053035 | -0.041437 | 主体匹配仍偏错误候选 | photo_dd1b9056edc18f04 |
| [photo_4cc2cdefcf733a54](photo_4cc2cdefcf733a54/photo_4cc2cdefcf733a54/REPORT.md) | val | 3 | -0.004294 | +0.003146 | 主体筛选可翻转这两个候选 | photo_49600b91aa952055 |
| [photo_4d9a6959fb508a57](photo_4d9a6959fb508a57/photo_4d9a6959fb508a57/REPORT.md) | train | 2 | -0.001556 | +0.000909 | 主体筛选可翻转这两个候选 | photo_f0d980d139f16444 |
| [photo_50614555bf86aaf4](photo_50614555bf86aaf4/photo_50614555bf86aaf4/REPORT.md) | train | 64 | -0.074206 | -0.091625 | 主体匹配仍偏错误候选 | photo_1089688627444cf6 |
| [photo_518888b98e9ac193](photo_518888b98e9ac193/photo_518888b98e9ac193/REPORT.md) | train | 2 | -0.002932 | +0.021234 | 主体筛选可翻转这两个候选 | photo_27cce13b62f1216a |
| [photo_51aaba9b1676b797](photo_51aaba9b1676b797/photo_51aaba9b1676b797/REPORT.md) | train | 8 | -0.016155 | -0.006913 | 主体匹配仍偏错误候选 | photo_5d0a29a6c57b4010 |
| [photo_51b91a8efbda2ee9](photo_51b91a8efbda2ee9/photo_51b91a8efbda2ee9/REPORT.md) | train | 117 | -0.039639 | -0.053587 | 主体匹配仍偏错误候选 | photo_792997f8dbfba097 |
| [photo_52547f944821ac72](photo_52547f944821ac72/photo_52547f944821ac72/REPORT.md) | val | 15 | -0.037965 | -0.051601 | 主体匹配仍偏错误候选 | photo_2df687d8a2f89f40 |
| [photo_527d8bc5deabf857](photo_527d8bc5deabf857/photo_527d8bc5deabf857/REPORT.md) | train | 2 | -0.004240 | -0.001443 | 主体匹配仍偏错误候选 | photo_26a54b4b825eefbb |
| [photo_529ca7857c6387ae](photo_529ca7857c6387ae/photo_529ca7857c6387ae/REPORT.md) | test | 4 | -0.008709 | -0.012722 | 主体匹配仍偏错误候选 | photo_16f724e075d1e714 |
| [photo_554f71fba7e6ec2d](photo_554f71fba7e6ec2d/photo_554f71fba7e6ec2d/REPORT.md) | train | 2 | -0.003739 | +0.004826 | 主体筛选可翻转这两个候选 | photo_bdfa0b3f68166b32 |
| [photo_55b61213a0c8f1a9](photo_55b61213a0c8f1a9/photo_55b61213a0c8f1a9/REPORT.md) | train | 2 | -0.000750 | -0.000919 | 同一候选对的身份歧义待复核 | photo_184d6f956513c5d8 |
| [photo_5623815155f4d189](photo_5623815155f4d189/photo_5623815155f4d189/REPORT.md) | train | 2 | -0.017783 | -0.049229 | 主体匹配仍偏错误候选 | photo_63103452c5081a78 |
| [photo_578a5c367aa237fe](photo_578a5c367aa237fe/photo_578a5c367aa237fe/REPORT.md) | train | 16 | -0.050393 | -0.080847 | 主体匹配仍偏错误候选 | photo_b111a943691a2aa7 |
| [photo_58545c26f42820eb](photo_58545c26f42820eb/photo_58545c26f42820eb/REPORT.md) | train | 2 | -0.004665 | -0.013990 | 主体匹配仍偏错误候选 | photo_1d2280603a42552a |
| [photo_5885af0890eb6878](photo_5885af0890eb6878/photo_5885af0890eb6878/REPORT.md) | train | 2 | -0.000595 | +0.005805 | 主体筛选可翻转这两个候选 | photo_1d2280603a42552a |
| [photo_5901c8902f9ff4f1](photo_5901c8902f9ff4f1/photo_5901c8902f9ff4f1/REPORT.md) | train | 5 | -0.008717 | -0.003983 | 主体匹配仍偏错误候选 | photo_1b6a28535da238b9 |
| [photo_591ada8ecfa071bc](photo_591ada8ecfa071bc/photo_591ada8ecfa071bc/REPORT.md) | train | 2 | -0.013615 | -0.009621 | 主体匹配仍偏错误候选 | photo_f0d980d139f16444 |
| [photo_5956b454b359f3e7](photo_5956b454b359f3e7/photo_5956b454b359f3e7/REPORT.md) | train | 2 | -0.004844 | -0.000451 | 主体匹配仍偏错误候选 | photo_3c8abd32bcea6cfa |
| [photo_5964b5e1bd6ef958](photo_5964b5e1bd6ef958/photo_5964b5e1bd6ef958/REPORT.md) | train | 657 | -0.069397 | -0.005519 | 主体匹配仍偏错误候选 | photo_9872ed71c7af7565 |
| [photo_59d5e8c638df29f0](photo_59d5e8c638df29f0/photo_59d5e8c638df29f0/REPORT.md) | train | 114 | -0.074698 | -0.076696 | 主体匹配仍偏错误候选 | photo_4f2addadc5937695 |
| [photo_5b21aa76cc4c1a37](photo_5b21aa76cc4c1a37/photo_5b21aa76cc4c1a37/REPORT.md) | train | 2 | -0.006172 | -0.004491 | 主体匹配仍偏错误候选 | photo_72ad5f2e7435bc37 |
| [photo_5e312e599a65ec79](photo_5e312e599a65ec79/photo_5e312e599a65ec79/REPORT.md) | train | 3 | -0.016313 | +0.019282 | 主体筛选可翻转这两个候选 | photo_a018168fa8c88d4c |
| [photo_5e415d5f852b3694](photo_5e415d5f852b3694/photo_5e415d5f852b3694/REPORT.md) | train | 4 | -0.009876 | -0.013339 | 主体匹配仍偏错误候选 | photo_1089688627444cf6 |
| [photo_5ebb86bf9ef59cc4](photo_5ebb86bf9ef59cc4/photo_5ebb86bf9ef59cc4/REPORT.md) | train | 2 | -0.005054 | +0.007063 | 主体筛选可翻转这两个候选 | photo_f29d766322bd9fc2 |
| [photo_5eee185cb7b7326a](photo_5eee185cb7b7326a/photo_5eee185cb7b7326a/REPORT.md) | train | 3 | -0.013115 | -0.013301 | 主体匹配仍偏错误候选 | photo_8a094cfd1e9fa61e |
| [photo_5f2ae9df66c280c5](photo_5f2ae9df66c280c5/photo_5f2ae9df66c280c5/REPORT.md) | train | 1685 | -0.103869 | -0.119373 | 主体匹配仍偏错误候选 | photo_1d7f3a9eaf659041 |
| [photo_6063de8dd48b934a](photo_6063de8dd48b934a/photo_6063de8dd48b934a/REPORT.md) | train | 2 | -0.005201 | +0.000534 | 主体筛选可翻转这两个候选 | photo_8e714c29254155f0 |
| [photo_614a8f00f4c9b153](photo_614a8f00f4c9b153/photo_614a8f00f4c9b153/REPORT.md) | train | 7 | -0.019451 | -0.035446 | 主体匹配仍偏错误候选 | photo_4917ad5f3fac9f55 |
| [photo_6172bb3bbbb3a2d6](photo_6172bb3bbbb3a2d6/photo_6172bb3bbbb3a2d6/REPORT.md) | val | 10 | -0.010434 | -0.028988 | 同一候选对的身份歧义待复核 | 无 |
| [photo_63d8b363d3a72f0f](photo_63d8b363d3a72f0f/photo_63d8b363d3a72f0f/REPORT.md) | train | 4 | -0.001383 | +0.011904 | 主体筛选可翻转这两个候选 | photo_ab6fd7a33337ccee |
| [photo_64625e0a749be4a5](photo_64625e0a749be4a5/photo_64625e0a749be4a5/REPORT.md) | train | 19 | -0.010822 | -0.011775 | 主体匹配仍偏错误候选 | photo_d2377339b5c178d4 |
| [photo_65dbff1a3b69197f](photo_65dbff1a3b69197f/photo_65dbff1a3b69197f/REPORT.md) | test | 2 | -0.000051 | +0.052357 | 主体筛选可翻转这两个候选 | photo_e589720459db3cc8 |
| [photo_65de2e175e6cbd9d](photo_65de2e175e6cbd9d/photo_65de2e175e6cbd9d/REPORT.md) | test | 5 | -0.039394 | -0.029283 | 主体匹配仍偏错误候选 | photo_7f4bc7d166b40bfc |
| [photo_6634ecb6a03f1b3a](photo_6634ecb6a03f1b3a/photo_6634ecb6a03f1b3a/REPORT.md) | val | 2 | -0.004619 | +0.007700 | 主体筛选可翻转这两个候选 | photo_49600b91aa952055 |
| [photo_665b96b3fb0b01c7](photo_665b96b3fb0b01c7/photo_665b96b3fb0b01c7/REPORT.md) | test | 2 | -0.009928 | -0.004483 | 主体匹配仍偏错误候选 | photo_20594074f229bd8f |
| [photo_6c1be6707e17adb6](photo_6c1be6707e17adb6/photo_6c1be6707e17adb6/REPORT.md) | train | 5 | -0.030213 | -0.065118 | 主体匹配仍偏错误候选 | photo_72ad5f2e7435bc37 |
| [photo_6df2442272ecdb72](photo_6df2442272ecdb72/photo_6df2442272ecdb72/REPORT.md) | train | 599 | -0.046599 | -0.043370 | 主体匹配仍偏错误候选 | photo_9fbc291d90193703 |
| [photo_6e23145f8417a076](photo_6e23145f8417a076/photo_6e23145f8417a076/REPORT.md) | val | 80 | -0.019627 | -0.009496 | 主体匹配仍偏错误候选 | photo_38d1d33620ab1ea1 |
| [photo_6f1045004ede2c6e](photo_6f1045004ede2c6e/photo_6f1045004ede2c6e/REPORT.md) | train | 11 | -0.013925 | -0.016698 | 主体匹配仍偏错误候选 | photo_51f7a3bb218030b8 |
| [photo_6f5a3b64b5809ad5](photo_6f5a3b64b5809ad5/photo_6f5a3b64b5809ad5/REPORT.md) | test | 54 | -0.030302 | -0.015162 | 主体匹配仍偏错误候选 | photo_db0f3171b2e53170 |
| [photo_740025f501c6528b](photo_740025f501c6528b/photo_740025f501c6528b/REPORT.md) | train | 160 | -0.072386 | -0.063903 | 主体匹配仍偏错误候选 | photo_079284acc905a73f |
| [photo_759614539794e4a6](photo_759614539794e4a6/photo_759614539794e4a6/REPORT.md) | train | 215 | -0.071462 | -0.073892 | 主体匹配仍偏错误候选 | photo_02c11dd14bf4116f |
| [photo_76125f3f5a5762a5](photo_76125f3f5a5762a5/photo_76125f3f5a5762a5/REPORT.md) | train | 120 | -0.034852 | -0.009177 | 主体匹配仍偏错误候选 | photo_792997f8dbfba097 |
| [photo_7b5ffe8d495ed220](photo_7b5ffe8d495ed220/photo_7b5ffe8d495ed220/REPORT.md) | test | 2 | -0.001358 | +0.001526 | 主体筛选可翻转这两个候选 | photo_20594074f229bd8f |
| [photo_7c0c3a5cb514ea44](photo_7c0c3a5cb514ea44/photo_7c0c3a5cb514ea44/REPORT.md) | train | 2 | -0.002333 | +0.007901 | 主体筛选可翻转这两个候选 | photo_1b6a28535da238b9 |
| [photo_7c65ac75f6293f15](photo_7c65ac75f6293f15/photo_7c65ac75f6293f15/REPORT.md) | train | 6 | -0.008393 | +0.051279 | 主体筛选可翻转这两个候选 | photo_d4b8067813323602 |
| [photo_7d94403c5a56e1a9](photo_7d94403c5a56e1a9/photo_7d94403c5a56e1a9/REPORT.md) | train | 49 | -0.029557 | -0.012084 | 主体匹配仍偏错误候选 | photo_5149e96e0098a5e9 |
| [photo_7e79bf355aa7c38b](photo_7e79bf355aa7c38b/photo_7e79bf355aa7c38b/REPORT.md) | val | 1574 | -0.076171 | +0.022085 | 主体筛选可翻转这两个候选 | photo_41892aea27698341 |
| [photo_7eb4a41f278c6e96](photo_7eb4a41f278c6e96/photo_7eb4a41f278c6e96/REPORT.md) | train | 4 | -0.032804 | -0.042508 | 主体匹配仍偏错误候选 | photo_079284acc905a73f |
| [photo_7ee8dcc5521c64c9](photo_7ee8dcc5521c64c9/photo_7ee8dcc5521c64c9/REPORT.md) | train | 166 | -0.085926 | -0.088677 | 主体匹配仍偏错误候选 | photo_441c254874dcdd8d |
| [photo_7f0add741b1b243c](photo_7f0add741b1b243c/photo_7f0add741b1b243c/REPORT.md) | test | 2 | -0.002320 | +0.033536 | 主体筛选可翻转这两个候选 | photo_709e8a3ee42539bc |
| [photo_7f3f0a27f2d5c4c3](photo_7f3f0a27f2d5c4c3/photo_7f3f0a27f2d5c4c3/REPORT.md) | train | 2 | -0.007253 | -0.005829 | 主体匹配仍偏错误候选 | photo_ab6fd7a33337ccee |
| [photo_80f2fdc29081fedd](photo_80f2fdc29081fedd/photo_80f2fdc29081fedd/REPORT.md) | train | 297 | -0.052390 | -0.059184 | 主体匹配仍偏错误候选 | photo_59498b6acfbfbf4f |
| [photo_813fb072c0f3b380](photo_813fb072c0f3b380/photo_813fb072c0f3b380/REPORT.md) | test | 2 | -0.017577 | -0.015735 | 主体匹配仍偏错误候选 | photo_7e0b721a9c25b555 |
| [photo_817005ee11887c2a](photo_817005ee11887c2a/photo_817005ee11887c2a/REPORT.md) | train | 9 | -0.039617 | -0.050700 | 主体匹配仍偏错误候选 | photo_4917ad5f3fac9f55 |
| [photo_83941d6fe68296bd](photo_83941d6fe68296bd/photo_83941d6fe68296bd/REPORT.md) | train | 600 | -0.033663 | +0.032158 | 主体筛选可翻转这两个候选 | photo_ab6fd7a33337ccee |
| [photo_85e10feb92319a7b](photo_85e10feb92319a7b/photo_85e10feb92319a7b/REPORT.md) | train | 16 | -0.034692 | -0.001424 | 主体匹配仍偏错误候选 | photo_1b6a28535da238b9 |
| [photo_85e17d28180026cd](photo_85e17d28180026cd/photo_85e17d28180026cd/REPORT.md) | train | 2 | -0.003624 | +0.022277 | 主体筛选可翻转这两个候选 | photo_f0ed0b2bc867b9df |
| [photo_86a37b0da0dd5ece](photo_86a37b0da0dd5ece/photo_86a37b0da0dd5ece/REPORT.md) | train | 12 | -0.010572 | -0.008701 | 主体匹配仍偏错误候选 | photo_e156cd0eb0edd234 |
| [photo_8788bcb16ad7b804](photo_8788bcb16ad7b804/photo_8788bcb16ad7b804/REPORT.md) | train | 6 | -0.019810 | -0.038902 | 主体匹配仍偏错误候选 | photo_184d6f956513c5d8 |
| [photo_89af21ff4b0209d2](photo_89af21ff4b0209d2/photo_89af21ff4b0209d2/REPORT.md) | val | 3 | -0.005041 | +0.028944 | 主体筛选可翻转这两个候选 | photo_0380e0954452b0a2 |
| [photo_8a0c0b4d48545276](photo_8a0c0b4d48545276/photo_8a0c0b4d48545276/REPORT.md) | train | 2 | -0.006671 | +0.032151 | 主体筛选可翻转这两个候选 | photo_3f63c9d3d60be6a0 |
| [photo_8a62e5cd6902b0ae](photo_8a62e5cd6902b0ae/photo_8a62e5cd6902b0ae/REPORT.md) | train | 4 | -0.015964 | -0.026304 | 主体匹配仍偏错误候选 | photo_079284acc905a73f |
| [photo_8bc3a7a3577518f0](photo_8bc3a7a3577518f0/photo_8bc3a7a3577518f0/REPORT.md) | train | 1004 | -0.166201 | -0.252806 | 主体匹配仍偏错误候选 | photo_441c254874dcdd8d |
| [photo_8c185778643e720d](photo_8c185778643e720d/photo_8c185778643e720d/REPORT.md) | train | 2 | -0.001907 | +0.000217 | 主体筛选可翻转这两个候选 | photo_841307b9e72b2c9d |
| [photo_8c54b7644f2e05b2](photo_8c54b7644f2e05b2/photo_8c54b7644f2e05b2/REPORT.md) | test | 10 | -0.014849 | -0.026494 | 主体匹配仍偏错误候选 | photo_709e8a3ee42539bc |
| [photo_8cb11807f4bb7510](photo_8cb11807f4bb7510/photo_8cb11807f4bb7510/REPORT.md) | val | 3 | -0.008323 | +0.104044 | 主体筛选可翻转这两个候选 | photo_ae99ad8e6f14e09e |
| [photo_8d5fa59eb06f3f45](photo_8d5fa59eb06f3f45/photo_8d5fa59eb06f3f45/REPORT.md) | train | 2 | -0.014453 | -0.047779 | 主体匹配仍偏错误候选 | photo_63103452c5081a78 |
| [photo_900c769d86302c63](photo_900c769d86302c63/photo_900c769d86302c63/REPORT.md) | train | 19 | -0.027469 | -0.018400 | 主体匹配仍偏错误候选 | photo_7247f008f2067eb2 |
| [photo_907a25ccce027528](photo_907a25ccce027528/photo_907a25ccce027528/REPORT.md) | train | 2 | -0.001330 | +0.014299 | 主体筛选可翻转这两个候选 | photo_b987ba3061c9f389 |
| [photo_93ed25f14350d103](photo_93ed25f14350d103/photo_93ed25f14350d103/REPORT.md) | val | 109 | -0.048947 | -0.063063 | 主体匹配仍偏错误候选 | photo_38d1d33620ab1ea1 |
| [photo_96d22a840ba148c4](photo_96d22a840ba148c4/photo_96d22a840ba148c4/REPORT.md) | test | 20 | -0.022829 | +0.013316 | 主体筛选可翻转这两个候选 | photo_5d0b3972b2685095 |
| [photo_9ae73e36628c0cd8](photo_9ae73e36628c0cd8/photo_9ae73e36628c0cd8/REPORT.md) | train | 2 | -0.000004 | +0.004062 | 主体筛选可翻转这两个候选 | photo_5d0a29a6c57b4010 |
| [photo_9b57495c8cae1f4b](photo_9b57495c8cae1f4b/photo_9b57495c8cae1f4b/REPORT.md) | train | 22 | -0.015937 | -0.010918 | 主体匹配仍偏错误候选 | photo_08acb7e0f1a5f5b9 |
| [photo_9bb993b443dfe2ff](photo_9bb993b443dfe2ff/photo_9bb993b443dfe2ff/REPORT.md) | test | 2 | -0.013901 | -0.012453 | 主体匹配仍偏错误候选 | photo_77b67ea38a19f6b3 |
| [photo_9d9c1aba85ca10cc](photo_9d9c1aba85ca10cc/photo_9d9c1aba85ca10cc/REPORT.md) | test | 2 | -0.006517 | +0.018944 | 主体筛选可翻转这两个候选 | photo_7f4bc7d166b40bfc |
| [photo_9e0c7038dd6543a7](photo_9e0c7038dd6543a7/photo_9e0c7038dd6543a7/REPORT.md) | train | 13 | -0.018413 | +0.004483 | 主体筛选可翻转这两个候选 | photo_1ba61c7d4c7edb4b |
| [photo_9e6d02a08ae58eae](photo_9e6d02a08ae58eae/photo_9e6d02a08ae58eae/REPORT.md) | test | 3 | -0.011336 | -0.013793 | 同一候选对的身份歧义待复核 | photo_16f724e075d1e714 |
| [photo_9f799473a3f5aacc](photo_9f799473a3f5aacc/photo_9f799473a3f5aacc/REPORT.md) | train | 2 | -0.004054 | -0.006030 | 主体匹配仍偏错误候选 | photo_9fbc291d90193703 |
| [photo_a0a282b24735a4d5](photo_a0a282b24735a4d5/photo_a0a282b24735a4d5/REPORT.md) | train | 2 | -0.005642 | -0.001720 | 主体匹配仍偏错误候选 | photo_add4d3fb17dace03 |
| [photo_a0c829b7849000b2](photo_a0c829b7849000b2/photo_a0c829b7849000b2/REPORT.md) | train | 40 | -0.012753 | -0.015701 | 主体匹配仍偏错误候选 | photo_4917ad5f3fac9f55 |
| [photo_a0dda9c59f0160fe](photo_a0dda9c59f0160fe/photo_a0dda9c59f0160fe/REPORT.md) | train | 2 | -0.018815 | -0.021652 | 主体匹配仍偏错误候选 | photo_80eb2d50ed90738e |
| [photo_a1dba542121735f2](photo_a1dba542121735f2/photo_a1dba542121735f2/REPORT.md) | test | 14 | -0.011856 | -0.015630 | 主体匹配仍偏错误候选 | photo_176abc65485427c3 |
| [photo_a2020c21aa08ba7b](photo_a2020c21aa08ba7b/photo_a2020c21aa08ba7b/REPORT.md) | train | 9 | -0.023870 | -0.025089 | 主体匹配仍偏错误候选 | photo_47430c79b3238e4a |
| [photo_a2635073f4550a08](photo_a2635073f4550a08/photo_a2635073f4550a08/REPORT.md) | train | 352 | -0.032170 | +0.025152 | 主体筛选可翻转这两个候选 | photo_375b67daeeeeff5f |
| [photo_a4ccf96dbf0308aa](photo_a4ccf96dbf0308aa/photo_a4ccf96dbf0308aa/REPORT.md) | test | 4 | -0.005317 | +0.005285 | 主体筛选可翻转这两个候选 | photo_5c30d65aff48a1e8 |
| [photo_a5e63c90602a29fd](photo_a5e63c90602a29fd/photo_a5e63c90602a29fd/REPORT.md) | train | 3 | -0.010180 | -0.002645 | 主体匹配仍偏错误候选 | photo_1ba61c7d4c7edb4b |
| [photo_a6097a19f3058ac0](photo_a6097a19f3058ac0/photo_a6097a19f3058ac0/REPORT.md) | train | 2 | -0.001993 | +0.024458 | 主体筛选可翻转这两个候选 | photo_d2377339b5c178d4 |
| [photo_a67c57011d4945e8](photo_a67c57011d4945e8/photo_a67c57011d4945e8/REPORT.md) | train | 2 | -0.003149 | +0.006091 | 主体筛选可翻转这两个候选 | photo_c86676a1232aefc7 |
| [photo_a7b92b4cabf1396a](photo_a7b92b4cabf1396a/photo_a7b92b4cabf1396a/REPORT.md) | train | 8 | -0.019599 | -0.001010 | 主体匹配仍偏错误候选 | photo_a018168fa8c88d4c |
| [photo_aabee66f53cb53e8](photo_aabee66f53cb53e8/photo_aabee66f53cb53e8/REPORT.md) | train | 3 | -0.021734 | +0.228919 | 主体筛选可翻转这两个候选 | photo_80eb2d50ed90738e |
| [photo_ab435a0418e86576](photo_ab435a0418e86576/photo_ab435a0418e86576/REPORT.md) | train | 2 | -0.014899 | -0.011069 | 主体匹配仍偏错误候选 | photo_519a92da367dee34 |
| [photo_ab72c04d610545ad](photo_ab72c04d610545ad/photo_ab72c04d610545ad/REPORT.md) | train | 338 | -0.074363 | -0.027333 | 主体匹配仍偏错误候选 | photo_f2e65c6d203e45a2 |
| [photo_acbddc48df8f77fd](photo_acbddc48df8f77fd/photo_acbddc48df8f77fd/REPORT.md) | train | 112 | -0.036756 | -0.051999 | 主体匹配仍偏错误候选 | photo_6e3ed6cfb297f29c |
| [photo_ace6fcbc6d37331a](photo_ace6fcbc6d37331a/photo_ace6fcbc6d37331a/REPORT.md) | val | 855 | -0.055174 | -0.028912 | 主体匹配仍偏错误候选 | photo_60b643cb7fb93b4e |
| [photo_ad661f7018b71f1f](photo_ad661f7018b71f1f/photo_ad661f7018b71f1f/REPORT.md) | val | 4 | -0.004517 | +0.017724 | 主体筛选可翻转这两个候选 | photo_2df687d8a2f89f40 |
| [photo_ae7153c51817b839](photo_ae7153c51817b839/photo_ae7153c51817b839/REPORT.md) | train | 9 | -0.016630 | +0.045869 | 主体筛选可翻转这两个候选 | photo_41c0b43ce4b3b4f0 |
| [photo_b05f3285c039c3be](photo_b05f3285c039c3be/photo_b05f3285c039c3be/REPORT.md) | train | 2135 | -0.044875 | -0.019485 | 主体匹配仍偏错误候选 | photo_0cf112e57c488bf5 |
| [photo_b43c83ea1488ebc7](photo_b43c83ea1488ebc7/photo_b43c83ea1488ebc7/REPORT.md) | train | 2 | -0.001275 | -0.014699 | 主体匹配仍偏错误候选 | photo_cf724bd3f6c03346 |
| [photo_b64001331fe6182d](photo_b64001331fe6182d/photo_b64001331fe6182d/REPORT.md) | train | 3 | -0.018119 | -0.015209 | 主体匹配仍偏错误候选 | photo_3431544b3be0914a |
| [photo_b81d1f98eea3452f](photo_b81d1f98eea3452f/photo_b81d1f98eea3452f/REPORT.md) | train | 2 | -0.001209 | +0.018172 | 主体筛选可翻转这两个候选 | photo_7247f008f2067eb2 |
| [photo_b8aaaa8b5f65ec7d](photo_b8aaaa8b5f65ec7d/photo_b8aaaa8b5f65ec7d/REPORT.md) | train | 5 | -0.003737 | +0.075622 | 主体筛选可翻转这两个候选 | photo_ab6fd7a33337ccee |
| [photo_b8f2292878596880](photo_b8f2292878596880/photo_b8f2292878596880/REPORT.md) | train | 2 | -0.005203 | +0.027915 | 主体筛选可翻转这两个候选 | photo_43eca55aded800ef |
| [photo_b91498eae9c3ff76](photo_b91498eae9c3ff76/photo_b91498eae9c3ff76/REPORT.md) | train | 2 | -0.001876 | +0.021967 | 主体筛选可翻转这两个候选 | photo_51f7a3bb218030b8 |
| [photo_b96e040e473298cb](photo_b96e040e473298cb/photo_b96e040e473298cb/REPORT.md) | train | 3 | -0.004209 | +0.001506 | 主体筛选可翻转这两个候选 | photo_8bd2c2ece5146d67 |
| [photo_b9713874b6c6dd20](photo_b9713874b6c6dd20/photo_b9713874b6c6dd20/REPORT.md) | train | 1172 | -0.049857 | +0.006333 | 主体筛选可翻转这两个候选 | photo_2228b5a9ef039869 |
| [photo_ba60dbabd4d50f54](photo_ba60dbabd4d50f54/photo_ba60dbabd4d50f54/REPORT.md) | train | 2 | -0.001803 | -0.000007 | 主体匹配仍偏错误候选 | photo_f0d980d139f16444 |
| [photo_bc5564b0e01eeb18](photo_bc5564b0e01eeb18/photo_bc5564b0e01eeb18/REPORT.md) | test | 2 | -0.001996 | -0.005083 | 主体匹配仍偏错误候选 | photo_20594074f229bd8f |
| [photo_bd7ba38f8275e85a](photo_bd7ba38f8275e85a/photo_bd7ba38f8275e85a/REPORT.md) | train | 7 | -0.016572 | -0.039692 | 主体匹配仍偏错误候选 | photo_4917ad5f3fac9f55 |
| [photo_bfe7ecdd06599257](photo_bfe7ecdd06599257/photo_bfe7ecdd06599257/REPORT.md) | val | 38 | -0.059029 | -0.070653 | 主体匹配仍偏错误候选 | photo_38d1d33620ab1ea1 |
| [photo_c396fe59281b6048](photo_c396fe59281b6048/photo_c396fe59281b6048/REPORT.md) | train | 2 | -0.009642 | -0.003420 | 主体匹配仍偏错误候选 | photo_bdfa0b3f68166b32 |
| [photo_c49cbdcb940ee7ee](photo_c49cbdcb940ee7ee/photo_c49cbdcb940ee7ee/REPORT.md) | val | 74 | -0.016819 | +0.008528 | 主体筛选可翻转这两个候选 | photo_2df687d8a2f89f40 |
| [photo_c4aed057612870fa](photo_c4aed057612870fa/photo_c4aed057612870fa/REPORT.md) | test | 222 | -0.029935 | -0.023146 | 主体匹配仍偏错误候选 | photo_fa7137de993a1c70 |
| [photo_c4b827d07e5c4d36](photo_c4b827d07e5c4d36/photo_c4b827d07e5c4d36/REPORT.md) | test | 673 | -0.042728 | -0.069072 | 主体匹配仍偏错误候选 | photo_140b3ac1eebfbb14 |
| [photo_c576c0c30f74cd4f](photo_c576c0c30f74cd4f/photo_c576c0c30f74cd4f/REPORT.md) | val | 10 | -0.013783 | -0.024536 | 主体匹配仍偏错误候选 | 无 |
| [photo_c750468e3ae50d48](photo_c750468e3ae50d48/photo_c750468e3ae50d48/REPORT.md) | val | 3 | -0.014576 | -0.023176 | 主体匹配仍偏错误候选 | photo_7bc115b787ac85b4 |
| [photo_c793f93fe5a60b87](photo_c793f93fe5a60b87/photo_c793f93fe5a60b87/REPORT.md) | train | 112 | -0.036867 | -0.053436 | 主体匹配仍偏错误候选 | photo_375b67daeeeeff5f |
| [photo_c798e089f34805e9](photo_c798e089f34805e9/photo_c798e089f34805e9/REPORT.md) | train | 2 | -0.002161 | +0.017907 | 主体筛选可翻转这两个候选 | photo_b5df0dce47884a73 |
| [photo_c81d2a7abb5af160](photo_c81d2a7abb5af160/photo_c81d2a7abb5af160/REPORT.md) | train | 5 | -0.002286 | -0.017534 | 主体匹配仍偏错误候选 | photo_0eb729e9f7b895d9 |
| [photo_c88cc2a30a61a0bd](photo_c88cc2a30a61a0bd/photo_c88cc2a30a61a0bd/REPORT.md) | train | 13 | -0.012898 | +0.059870 | 主体筛选可翻转这两个候选 | photo_c8d48440bc924f6a |
| [photo_c8920e14924534af](photo_c8920e14924534af/photo_c8920e14924534af/REPORT.md) | test | 2 | -0.007082 | -0.022312 | 主体匹配仍偏错误候选 | photo_45b5970f1ebe4f19 |
| [photo_c926b33d2d03db03](photo_c926b33d2d03db03/photo_c926b33d2d03db03/REPORT.md) | train | 3 | -0.001608 | -0.007376 | 主体匹配仍偏错误候选 | photo_af9240a1c073e28a |
| [photo_cc01a8280f094648](photo_cc01a8280f094648/photo_cc01a8280f094648/REPORT.md) | train | 2 | -0.003820 | -0.007883 | 主体匹配仍偏错误候选 | photo_ab6fd7a33337ccee |
| [photo_cc46823d6b68b813](photo_cc46823d6b68b813/photo_cc46823d6b68b813/REPORT.md) | test | 643 | -0.067835 | -0.075132 | 主体匹配仍偏错误候选 | photo_fd6c39a0ef3bd92a |
| [photo_cd9c6c61f36c9560](photo_cd9c6c61f36c9560/photo_cd9c6c61f36c9560/REPORT.md) | train | 6 | -0.024171 | +0.014330 | 主体筛选可翻转这两个候选 | photo_4917ad5f3fac9f55 |
| [photo_cdd216116fa6260a](photo_cdd216116fa6260a/photo_cdd216116fa6260a/REPORT.md) | train | 7 | -0.004996 | -0.002531 | 主体匹配仍偏错误候选 | photo_5d0a29a6c57b4010 |
| [photo_d6f42eda7b050220](photo_d6f42eda7b050220/photo_d6f42eda7b050220/REPORT.md) | train | 745 | -0.047499 | -0.019877 | 主体匹配仍偏错误候选 | photo_134131912f9105af |
| [photo_d7a6037b4baf35d2](photo_d7a6037b4baf35d2/photo_d7a6037b4baf35d2/REPORT.md) | val | 190 | -0.075387 | -0.068638 | 主体匹配仍偏错误候选 | photo_894fd37ea662af76 |
| [photo_d82d30bab9d77a25](photo_d82d30bab9d77a25/photo_d82d30bab9d77a25/REPORT.md) | train | 5 | -0.020021 | +0.035671 | 主体筛选可翻转这两个候选 | photo_3f61e3e16c427164 |
| [photo_da31992b29a6c690](photo_da31992b29a6c690/photo_da31992b29a6c690/REPORT.md) | test | 2 | -0.001940 | +0.008332 | 主体筛选可翻转这两个候选 | photo_f1f661a94082552c |
| [photo_da9d9887ba9c878f](photo_da9d9887ba9c878f/photo_da9d9887ba9c878f/REPORT.md) | train | 2 | -0.010933 | -0.018496 | 主体匹配仍偏错误候选 | photo_f0d980d139f16444 |
| [photo_dc1a56b14e14c401](photo_dc1a56b14e14c401/photo_dc1a56b14e14c401/REPORT.md) | val | 34 | -0.042660 | -0.041339 | 主体匹配仍偏错误候选 | photo_38d1d33620ab1ea1 |
| [photo_dc4d31058a813a38](photo_dc4d31058a813a38/photo_dc4d31058a813a38/REPORT.md) | val | 2 | -0.001523 | +0.006059 | 主体筛选可翻转这两个候选 | photo_39880628576a7ea5 |
| [photo_dce2ba688bc88c07](photo_dce2ba688bc88c07/photo_dce2ba688bc88c07/REPORT.md) | train | 2 | -0.002567 | +0.008397 | 主体筛选可翻转这两个候选 | photo_174967c2577c6dc9 |
| [photo_df49e627bf523b65](photo_df49e627bf523b65/photo_df49e627bf523b65/REPORT.md) | train | 2 | -0.002577 | -0.007608 | 主体匹配仍偏错误候选 | photo_145709484be1d522 |
| [photo_e1cecb781eaf3b03](photo_e1cecb781eaf3b03/photo_e1cecb781eaf3b03/REPORT.md) | train | 2 | -0.002296 | +0.006282 | 主体筛选可翻转这两个候选 | photo_587142eeed6488be |
| [photo_e248fec4d6254ebf](photo_e248fec4d6254ebf/photo_e248fec4d6254ebf/REPORT.md) | val | 11 | -0.029957 | -0.017227 | 主体匹配仍偏错误候选 | 无 |
| [photo_e32b999979ddc21b](photo_e32b999979ddc21b/photo_e32b999979ddc21b/REPORT.md) | test | 2 | -0.004168 | +0.005068 | 主体筛选可翻转这两个候选 | photo_20594074f229bd8f |
| [photo_e5b4dba01622bfaf](photo_e5b4dba01622bfaf/photo_e5b4dba01622bfaf/REPORT.md) | train | 3 | -0.007887 | -0.003945 | 主体匹配仍偏错误候选 | photo_3cfb7f3d355174a8 |
| [photo_e6160ef7ce3f2f34](photo_e6160ef7ce3f2f34/photo_e6160ef7ce3f2f34/REPORT.md) | train | 2 | -0.002477 | +0.015833 | 主体筛选可翻转这两个候选 | photo_72ad5f2e7435bc37 |
| [photo_e8489caa72c02532](photo_e8489caa72c02532/photo_e8489caa72c02532/REPORT.md) | train | 7 | -0.006113 | +0.002289 | 主体筛选可翻转这两个候选 | photo_9d71019e60419497 |
| [photo_ea963143596a1a31](photo_ea963143596a1a31/photo_ea963143596a1a31/REPORT.md) | train | 13 | -0.011348 | +0.061497 | 主体筛选可翻转这两个候选 | photo_d65410305cff7c7c |
| [photo_eb95305031934df9](photo_eb95305031934df9/photo_eb95305031934df9/REPORT.md) | train | 2 | -0.000059 | +0.002155 | 主体筛选可翻转这两个候选 | photo_01340abaa6725c78 |
| [photo_ec1e6c42595e479e](photo_ec1e6c42595e479e/photo_ec1e6c42595e479e/REPORT.md) | train | 19 | -0.010503 | +0.021351 | 主体筛选可翻转这两个候选 | photo_03df1f5abf1d6a3d |
| [photo_eda8149b7c7961fa](photo_eda8149b7c7961fa/photo_eda8149b7c7961fa/REPORT.md) | train | 1258 | -0.063846 | -0.074872 | 主体匹配仍偏错误候选 | photo_0cf112e57c488bf5 |
| [photo_ee67048e97d12344](photo_ee67048e97d12344/photo_ee67048e97d12344/REPORT.md) | train | 2 | -0.008121 | +0.003320 | 主体筛选可翻转这两个候选 | photo_43eca55aded800ef |
| [photo_eecf79f118c50de9](photo_eecf79f118c50de9/photo_eecf79f118c50de9/REPORT.md) | train | 3976 | -0.085189 | -0.097630 | 主体匹配仍偏错误候选 | photo_f74baa8e17b7087b |
| [photo_efb1beec956e46ec](photo_efb1beec956e46ec/photo_efb1beec956e46ec/REPORT.md) | train | 2 | -0.000436 | +0.007226 | 主体筛选可翻转这两个候选 | photo_0be78636d1b29c9f |
| [photo_f0d5d4b854af8b85](photo_f0d5d4b854af8b85/photo_f0d5d4b854af8b85/REPORT.md) | val | 2565 | -0.073106 | -0.051891 | 主体匹配仍偏错误候选 | photo_894fd37ea662af76 |
| [photo_f2049cd8617674da](photo_f2049cd8617674da/photo_f2049cd8617674da/REPORT.md) | train | 2 | -0.000550 | +0.000026 | 主体筛选可翻转这两个候选 | photo_dd1b9056edc18f04 |
| [photo_f23f2de9b2d68b17](photo_f23f2de9b2d68b17/photo_f23f2de9b2d68b17/REPORT.md) | train | 29 | -0.027562 | -0.024533 | 主体匹配仍偏错误候选 | photo_f0d980d139f16444 |
| [photo_f30f6769bf236e52](photo_f30f6769bf236e52/photo_f30f6769bf236e52/REPORT.md) | train | 591 | -0.091565 | -0.089847 | 主体匹配仍偏错误候选 | photo_441c254874dcdd8d |
| [photo_f3f437a6e3a67796](photo_f3f437a6e3a67796/photo_f3f437a6e3a67796/REPORT.md) | train | 7 | -0.019311 | -0.004185 | 主体匹配仍偏错误候选 | photo_4917ad5f3fac9f55 |
| [photo_f8f533ac85b05854](photo_f8f533ac85b05854/photo_f8f533ac85b05854/REPORT.md) | val | 27 | -0.010022 | +0.000504 | 主体筛选可翻转这两个候选 | 无 |
| [photo_f9a136cbb2bdc194](photo_f9a136cbb2bdc194/photo_f9a136cbb2bdc194/REPORT.md) | train | 3 | -0.033401 | +0.002579 | 主体筛选可翻转这两个候选 | photo_1a510aba547e1d64 |
| [photo_fb6b4915947444a2](photo_fb6b4915947444a2/photo_fb6b4915947444a2/REPORT.md) | train | 2 | -0.001126 | -0.002722 | 主体匹配仍偏错误候选 | photo_09c56f438b411fb1 |
| [photo_fbc7b6d852b1125b](photo_fbc7b6d852b1125b/photo_fbc7b6d852b1125b/REPORT.md) | train | 4 | -0.010805 | -0.025040 | 主体匹配仍偏错误候选 | photo_184d6f956513c5d8 |
| [photo_fc0ee05b6b185ffa](photo_fc0ee05b6b185ffa/photo_fc0ee05b6b185ffa/REPORT.md) | train | 6 | -0.032082 | -0.056089 | 主体匹配仍偏错误候选 | photo_618866896f180a73 |
| [photo_fd3bb2cdcb3bd3a7](photo_fd3bb2cdcb3bd3a7/photo_fd3bb2cdcb3bd3a7/REPORT.md) | train | 24 | -0.040275 | +0.129251 | 主体筛选可翻转这两个候选 | photo_d68d74e6a16e45bf |
| [photo_fd5974ccf5d29b5a](photo_fd5974ccf5d29b5a/photo_fd5974ccf5d29b5a/REPORT.md) | train | 2 | -0.002464 | +0.085235 | 主体筛选可翻转这两个候选 | photo_b35bafd0e0a8761b |
| [photo_fde389ec0d74a099](photo_fde389ec0d74a099/photo_fde389ec0d74a099/REPORT.md) | train | 2181 | -0.054308 | -0.055429 | 主体匹配仍偏错误候选 | photo_4dfba78901c38dc4 |

### 原始特征与复现补充

已补充原始特征、完整候选对相似度矩阵和冻结编码代码。入口：[REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
