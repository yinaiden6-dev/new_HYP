# 全量冻结Token导出复现

只读使用已有 cache 与 dataset；CPU 2线程，不加载模型，不编码图像。

继续（持有进程锁时第二实例会拒绝；每Query校验complete.json所有SHA后复用）：

```bash
/home/jwei/miniconda3/envs/rebut/bin/python -u -B scripts/colnomic_mechanism/full_tokens.py --threads 2
```

只验证（全部查询档案SHA、完整图库排名、逐Token/区域守恒，并从冻结向量逐元素重新计算全部已导出FP32矩阵）：

```bash
/home/jwei/miniconda3/envs/rebut/bin/python -u -B scripts/colnomic_mechanism/full_tokens.py --threads 2 --verify-only
```

断流后先读runtime.json、progress.json和run_full.log尾部，并检查PID及fcntl锁；不要重复启动。

查询档案与矩阵采用有效Token索引（非零模板保留）；完整图库标量分数轴包括全部5405记录，排名先稳定排序再按SHA去重为5330设计。正确记录由positives.csv核验，并扩展全部SHA别名；主候选按原image分数最大、同分银行行序。

自动OCR行区域、对应、分差和材料生成不是视觉审核；视觉审核和机制证据覆盖必须由单独的实际浏览记录确定。按4Q/页×5列的visual_material含原图、已确认主体框crop和关键候选；初始2Q部分页保留，完整4Q版另存batch_001_complete.jpg。

目录与源代码快照为独立补充结果；既有B03、F03、v1、v2不改写。
