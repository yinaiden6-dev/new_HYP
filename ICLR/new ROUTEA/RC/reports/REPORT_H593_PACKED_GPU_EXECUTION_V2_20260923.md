# H593 四 GPU 统一执行池

用户提出：同一批 593 张图、提取不同信息，可把排队任务包装成一个多 GPU/CPU 作业并行执行。已完成执行器、调度器、回退和衔接检查，并提交真实四卡试运行。当前尚未通过实际四卡资格验证，不能宣称达到四倍加速或已有新科学结果。

## 已提交

|项目|值|
|---|---|
|试运行 Job ID|5158723|
|分区|dev_accelerated|
|资源|单节点，4 GPU，32 CPU，256GB|
|时限|试运行 30 分钟；后续合并批次 55 分钟|
|真实调度状态|2026-09-23 03:52 Europe/Berlin：RUNNING，hkn0402；原前置 5158700_32 已完成|
|四项试运行|inside query033、visual query047、coordinate query063、fusion config093|
|调度器|hkn1993.localdomain，PID 2386443；已在独立调用中验证存活及心跳|

Slurm 的 dev QOS 限制为每用户同时运行 1 个作业、最多提交 4 个作业；未设置每作业仅 1 GPU 的限制。单作业 2 GPU / 4 GPU 的 test-only 请求都被接受。5158723 的实际 ReqTRES 已核查为 gpu=4、cpu=32、mem=256G。现已实际获得资源，四个 step 的设备记录确认分别绑定 GPU 0、1、2、3，四个 UUID 不同，均为 A100-SXM4-40GB。

03:52 进度：inside query033 已通过完整 128 候选检查（QUERY_PASS）；visual query047 已保存 88/128 对；coordinate query063 已保存 128/128 对，尚未在本次检查中确认最终验证；fusion config093 已记录到 step100。四类工作实际并行已确认，全套试运行资格仍须等待四项退出和原验证器检查。

## 怎样执行

四个 Slurm step 各分配 1 GPU、8 CPU、64G。每项工作启动原冻结程序的新进程，完成后动态领取下一项；三类图像分析分片与融合训练分片进入统一队列。每配置每 allocation 最多执行一次，避免覆写原融合 chunk receipt。真实 Job ID、原 checkpoint、原验证器和中间数据都保留。

固定原完整 C128、FP64 评分、五折划分、训练更新顺序与干预定义。共享已有磁盘缓存和页缓存；不同干预仍执行各自原算法。这里合并的是资源和调度，不是改写实验公式。

## 切换与回退

1. 保留原 139 个待运行子任务/回调为 held；运行中的 5158700_32 自然结束，试运行依赖其成功完成。旧结果未删除，缓存和 checkpoint 原位续跑。
2. 四项试运行须分别正常退出。三类图像分析必须通过原完整 C128 验证；融合须保存原可续跑 checkpoint 并生成正常 chunk receipt；四个 step 须绑定不同 GPU。
3. 通过后自动接管三类图像分析的旧队列，直接接续合并批次，消除逐批 CPU 续提排队。M-path readout 保留到 inside 全量完成后释放，依赖先重绑再取消旧前置。
4. 旧融合调度器位于另一登录节点。融合旧数组继续保持 held，避免它提交重复训练波次；新池负责计算，旧调度器根据原验证文件负责最终 join。融合汇总通过后清理 held 占位任务。
5. 试运行失败则自动释放被本次暂停的旧队列。后续正式池遇到计算失败则保留证据并停止自动扩展。

## 已完成的工程检查

- 实际启动脚本主体的参数传递检查，含带空格路径；提交 spool 与冻结脚本字节一致。
- 四类原程序入口、唯一/公平任务选择、运行中任务排除。
- 重绑 readout 依赖先于取消旧前置；融合 held 占位防止旧调度器重复提交。
- 不在剩余不足 15 分钟时启动原程序；给原 850 秒保护与最终封存保留余量。
- 静态/模拟检查通过，真实四卡独立分配和四类工作并行已确认；尚待全部原验证器结果和实际吞吐验证。

## 证据位置

- 实时状态：`results/rc_h593_packed_gpu_v2/status.json`
- 旧队列快照：`results/rc_h593_packed_gpu_v2/legacy.json`
- 试运行清单、提交与 spool：`results/rc_h593_packed_gpu_v2/batches/000/`
- 调度日志：`results/rc_h593_packed_gpu_v2/supervisor.out`、`supervisor.err`
- GPU 日志：`logs/h593-pack-5158723.out`、`.err`
- authority：`registry/rc_h593_packed_gpu_authority_v2_20260923.json`
- 方案：`plan/RC_H593_PACKED_GPU_EXECUTION_V2_20260923.md`

V1 只完成准备，未提交、未暂停任何原作业；因旧调度器跨登录节点，V2 改用完成文件衔接，保留 V1 为工程历史。
