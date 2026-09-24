# 内部M：V3失败定位与V4单因素修复

2026-09-24。用户目标为“内部能有效使用M”，不要求超过外部头；末端继续不直接读取M。本记录基于TRAIN16已封存缓存，没有读取probe。

## 先归档

论文收口稿与V3最终结果已推送 `yinaiden6-dev/new_HYP` 的main，提交 `0f1eead88613d5d875f81bd45ef0dea0f173ed3b`。109个变更文件约1.7MB；重新读取远端指针与本地一致。包含完整文本终点、逐候选分数和验收，不包含权重、token缓存、原图或err/out。入口：[GitHub归档](https://github.com/yinaiden6-dev/new_HYP/tree/0f1eead88613d5d875f81bd45ef0dea0f173ed3b/backup/internal_m_v3_final_closure_20260924)。

## 已经定位的两层问题

第一，原联合训练头没有充分利用现有终点内容。保持L固定，使用相同warm头、2000次全TRAIN16均值COST1更新，真实M终点从8/16变成9/16，恒定M训练臂从8/16变成10/16。真实M模型的常量/错绑干预也变成9/16。因此读出优化确有影响，但这些事后诊断没有建立真实M的作用。

| 固定终点内容 | V3原头 | 统一CPU重拟合头 | 重拟合平均COST1 |
|---|---:|---:|---:|
| PRE_REAL128，真实M | 8/16 | 9/16 | 0.831176 |
| PRE_REAL128，M换常量 | 8/16 | 9/16 | 0.828283 |
| PRE_REAL128，M错绑 | 8/16 | 9/16 | 0.827166 |
| PRE_CONSTANT128 | 8/16 | 10/16 | 0.804650 |

这些不是新留出成绩。CPU拟合已接近相同固定特征的数值凸最优损失，说明继续只调这三个参数未必解决问题。线性容量与事后偏置见证另有更高的TRAIN正确数，但相应COST1可能更差，且常量/错绑也存在相同现象；这些数学见证不作为模型性能。

第二，当前M输入支路可能太弱。在V3已保存的同一720-patch TRAIN图上，标准化M那一列对down层的RMS相对内容支路为1.414%→0.387%→0.260%（16/64/128步）。终点该trace候选的真实M-vs常量残差变化约为普通残差RMS的0.074%。全TRAIN16中M干预确实改变数值，却未带来纠错；不能说M完全没进模型，或被归一化完全抹掉。

支路幅度是单图证据，不是全体的统计因果结论。原TRAIN顺序为8张RAW正确后8张错误，头参数在逐query更新时也有摆动；本轮不同时修改顺序、损失和输入尺度，以免无法辨别改动作用。

## V4采取的最小动作

只将标准化M输入乘`sqrt(3584)`；参数量、网络结构、LLM前位置、COST1、初始化、原顺序和128步预算不变。新增正确绑定M、固定错绑M两条训练；原常量输入为0，增益乘0仍为0，因此复用V3常量臂，原gain1真实M也直接作为参照。

原共同头、reference tokens、RoMa M、merger输入全部复用。真实M的效果必须在正确绑定、恒定、错绑之间区分；不能将新增训练共有的收益计作M成功。固定终点统一增加CPU头重拟合诊断，旧新全部同规约，以免末端欠优化掩盖结果。

这是对“单维M条件尺度弱”假说的检验，尚未证实修复有效，也不保证扩充预算就成功。完整规约见[V4执行协议](../plan/RC_COLNOMIC_INTERNAL_M_CONDITION_SCALE_V4_EXECUTION_20260924.md)。

## 可复核来源

- [支路尺度审计脚本](../programs/audit_rc_internal_m_v3_condition_scale.py)、[330项来源绑定与量化](../results/rc_internal_m_v3_fit_diagnostics/condition_scale_audit.json)。
- [读出容量/优化审计脚本](../programs/audit_rc_internal_m_v3_readout_capacity.py)、[102项来源绑定及完整诊断](../results/rc_internal_m_v3_fit_diagnostics/readout_capacity_audit.json)。
- [V3最终报告](REPORT_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_FINAL_20260924.md)。

V3源码、authority和结果没有被原地修改；新实验在独立目录执行。后续作业与验收信息在V4结果目录中记录。

## 本轮提交与验证

- 2026-09-24 09:27 UTC 已提交 `5161817`（CPU复用验证）及依赖它的 `5161818`（真实GPU首步验证）。首次核查CPU已在`dev_cpuonly`启动，GPU等待`afterok:5161817`。
- 两个作业均为10分钟，可保存断点续跑；GPU申请单卡、8CPU/64GB，同时排`dev_accelerated,accelerated`。首步工程验收通过后，程序自动提交正确绑定、固定错绑两臂训练及依赖汇总。
- V3原有16项、V4新增21项测试全部通过；48个完整C128导数核对中，适配器梯度最大误差`2.22e-16`，头梯度最大误差0，内部头对直接M输入不敏感。
- 调度器保存的`5161818`脚本与冻结脚本逐字节相同。V4 authority SHA256为`6edb7e2e89524112e83148886f799b89106428ecc6dd05c1ede2c945d7d896f4`。
- 提交收据：[submission.json](../results/rc_internal_m_condition_scale_v4/submission.json)；测试收据：[prefreeze_tests.json](../results/rc_internal_m_condition_scale_v4/prefreeze_tests.json)；预检：[preflight.json](../results/rc_internal_m_condition_scale_v4/preflight.json)。
- 提交后复核：`5161817`已完成（37秒，退出码`0:0`）；三个复用头的参数、全部预测与原V3完全一致。`5161818`已分配`dev_accelerated`的`hkn0403`，调度状态RUNNING；此时尚无首步通过收据。详见[状态验证](../results/rc_internal_m_condition_scale_v4/submission_status_verified.json)。

这些是工程验证和提交状态，尚非V4科学结果。Git提交`0f1eead`覆盖本轮启动前的收口稿与V3最终结果；V4新程序、来源诊断与本报告暂在workspace，不能误称已随该提交上传。
