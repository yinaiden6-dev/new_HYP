> 更新：五折与汇总现已完成，最终RAW227 / MASS5 COST1 322 / CE333。见REPORT_COLQWEN_BASE_NATIVE_FINAL_20260924.md。以下保留检查时的恢复记录。

# ColQwen base H593：完成采集，恢复小头训练

2026-09-24当前检查：不是已训练ColQwen检索适配器的对照。使用本地colqwen2.5-base，Qwen2.5-VL-3B-Instruct骨干及未检索训练的固定投影，不加载检索LoRA。图像编码器本身已有预训练，不称整个网络未经训练。

- 图库5413张：使用与query一致的新编码配置完成；旧缓存兼容性未过的记录保留。
- query593张：全图库检索完成，自有自然C128，不使用ColNomic候选替代。
- RoMa质量：593×128=75904配对完成，复用30074，新算45830；50片全部验收。
- 5161250在完成quality汇总和head输入后，自动提交五折失败；因此此前无最终小头结果，不能误报为实验完成。
- 已补交5161390_[0-4%5]（cpuonly，8CPU/32GB/10分钟）、5161391（afterok五折汇总）。复用全部现成输入，不重复GPU采集，不改变训练协议。

原五折：CONTENT7、MASS5各比较COST1与CE；MASS5保留候选绑定错配对照。结果在results/rc_colqwen_base_native_v2/head/result.json，须有validation.json才报告正式计数。

范围：已打开H593分组OOF、base检索能力控制；不是训练版ColQwen性能，不是独立外部数据确认，也不是原七参数零样本迁移。
