# ColPali H593 query tokens 补采集

已提交 5160106，dev_accelerated / accelerated 共同排队，1 GPU、8 CPU、48GB、15分钟。实际提交脚本已字节核对。

只编码当前 H593 的 593 张 query；图像路径、SHA 和原坐标框架来自已有 feature-fusion catalog，输入清单已冻结。旧 outcome/difficult 的检索结果并不代替 query tokens，因此本次补齐范围不是仅 new difficult 的二十几张。reference 使用已有5413行缓存，RoMa 不重算。

开始时额外编码3张固定 reference，检查与历史缓存的数值兼容；不通过即停止。通过后逐张保存 FP16 tokens、input IDs、attention mask、image token mask、32×32网格、原图尺寸、RGB像素SHA、源文件SHA及模型/processor版本。两侧使用相同本地 ColPali v1.3快照，BF16前向。

每张文件写入后回读并验证，再发布SHA收据。预算用尽时保留完成项，至多自动requeue三次；运行异常不冒充完成。最终需593/593收据及validation.json通过，才能使用。

输出：`results/rc_colpali_h593_query_tokens_v1/`。详细执行记录：`submission_5160106.json`。本阶段无标签读取、无头训练、无检索准确率结果。

## 执行更新：50分片

按用户新指示，尚未运行的5160106已取消。替代数组5160112为accelerated，0–49%50，每片11–12张、1GPU/8CPU/48GB/15分钟。每片独立兼容性检查、写锁、进度和验收文件；按ordinal模50分片，593张无交叉无遗漏。超时预警或可恢复超时只requeue当前分片，最多8次；已验收图片跳过。实际脚本已字节核对，续提分支已用隔离mock验证。汇总5160114在cpuonly等待afterok:5160112，逐片及逐图SHA通过后才发布总validation.json。当前仅提交，无新科学结果。
