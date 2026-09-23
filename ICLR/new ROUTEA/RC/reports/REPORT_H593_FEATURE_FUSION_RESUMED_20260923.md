# H593 粗细特征融合分支恢复续提

用户明确要求“续提啊”，恢复当前融合分支。这一指令覆盖 M-priority 计划中对 fusion 的暂缓安排；坐标精度分支仍按原安排暂缓。M 内部通路和视觉干预继续执行。

## 实际操作与验证

- 已解除 `5158421_[48–92]` 共 45 个任务的 `JobHeldUser`，沿用原作业 ID，没有重复提交。
- 保留 `accelerated`、每片 1 GPU / 8 CPU / 64 GB / 13 分钟、原数组最大并发 46。该并发值属于融合数组；M 分支原有调度独立继续，并非两分支合计 46。
- 全部 45 个任务在释放后已退出 held 状态。释放即时仍为排队，不代表已经运行或实验完成。
- 已核对训练源码、训练 authority、原续提程序的 SHA，以及实际提交 spool 与冻结 launcher 逐字节一致。
- 当前四卡过渡批次的融合配置是 94–101，与恢复数组 48–92 无交集，没有同配置并行写入。
- 原 `rc_h593_feature_fusion_dispatch_v3` 监督器心跳正常，将沿用原 200 配置计划和检查点继续分批提交；原 48 小时监督期限和每配置最多 128 个 chunk 上限不变。

## 已有进度与数据

恢复时 10/200 个配置已有 `FUSION_FOLD_NUMPY_READOUT_PASS`，payload SHA 和训练 authority 已核验。其索引为 8、9、18、19、28、29、38、39、98、99。这是完成配置数，不是新的检索准确率或科学结论。

复用现有完整特征缓存以及 `results/rc_h593_feature_fusion_train_v2/fit*/` 的 checkpoint、正常退出 chunk 和中间输出；不重做特征提取，不改变训练顺序、损失、精度或候选集。全部 200 配置齐全后仍需原 join 和结果验证。

机器核查记录：`results/rc_h593_feature_fusion_resume_20260923_v1/intent.json`、`release_verified.json`、`5158421_spool.sh`。
