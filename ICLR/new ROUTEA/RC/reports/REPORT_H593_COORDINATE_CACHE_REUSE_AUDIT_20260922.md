# 坐标精度实验缓存复用核对

后续更新：用户已授权改造。缓存实现与自动切换已提交，逐位资格任务5157565、依赖切换5157566。实时结果以`REPORT_H593_COORDINATE_EXACT_CACHE_EXECUTION_20260922.md`及validation为准；以下“尚未实现”是本核对完成时的历史状态。


2026-09-22。用户询问5157033_[59-92]是否确实没有数据可复用。结论：已有输入已复用；当前前向内部还有明确重复计算；旧最终结果不足以生成新坐标干预结果。

## 已核实的输入

- 原H593 workers包含593个不同query图像SHA。原RAW ColNomic tokens、自然C128、原RoMa两侧token可见性和四统计由load_input加载；没有重新编码ColNomic。
- 对实际待执行ordinal59读取原RoMa缓存结构：候选只有token可见性、旧四统计、几何/来源与校验信息，没有原始密集DINO/VGG特征、refiner内部状态或完整warp。已有旧得分不能推出坐标干预后的overlap。
- 新融合缓存ready为FUSION_ALL593_FEATURE_CACHE_PASS，5324个图像/几何条目。实际payload保存的是已按ColNomic单元汇聚的components和128维projected_inputs；full_resolution_activations_saved=False。例如抽样图coarse为744×1024、fine为744×64/128/256，而非RoMa原始空间特征网格。不可无损恢复空间采样所需的密集特征。
- 已有合格query直接跳过；每16候选的part可恢复。当前0–58完成不会因本次核查重算。

## 明确未利用的复用机会

run_rc_h593_roma_coordinate_precision_v2.py:170–173对NATIVE/GRID256/GRID64/GRID16四臂逐个调用完整model.match。RoMaV2.forward先执行两侧self.f、matcher、LR/HR细特征，再进refiners。量化hook仅改变refiner的prev_warp。图像预处理、单图粗细特征、两图coarse matcher都不依赖量化级别；程序甚至逐位要求四臂coarse matcher输出相同，却每臂重算。

每query128候选×4臂=512次完整match。可将相同精度、相同预处理下的图像编码缓存一次，粗matcher每对一次，随后独立运行四路refiner和相应评分/中间数据。仅比较调用次数：单图粗编码从1024次降至最多129次（同query复用；128 reference）；LR/HR细特征从2048次降至最多258次；coarse matcher从512次降至128次。refiner仍需保留四臂，且CPU中间数据统计也占成本；这些不是已测墙钟加速比。

已有query058日志：128候选四臂worker约410.956秒，Slurm总墙钟7分36秒；不能据此假称所有时间均花在可缓存模块。

## 最小实现和验证边界

优先做进程内、query常驻＋reference逐对缓存，避免全量高分辨率特征落盘占用大量空间。保持旧match/refiner计算顺序、tensor dtype、scale、图像字节、EXIF、C128、量化位置和FP64评分；cache key区分图像、LR/HR及冻结模型版本。粗matcher返回值可能被下游引用，缓存重放要隔离原地写入。

另开版本，先在固定、不看标签的已有完整C128样本上，比对未缓存/缓存四臂所有原始输出、两侧token权重、坐标统计、匹配索引、逐token贡献、四统计、六特征和完整127分数；同时记录峰值显存及墙钟。逐位一致且资源合格之后，才替换待运行分片并接续依赖，已经合格的59个query保留。尚未实现/提交该缓存加速验证，本次没有修改5157033或当前融合训练任务。
