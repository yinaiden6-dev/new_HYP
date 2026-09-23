# M 为什么有用：恢复 accelerated 短分片并行

用户要求优先回答 M 为什么有用，并恢复原多 GPU 短任务并行。此次为调度切换，原实验定义和结果目录保留。

## 已执行

- 新调度器：`programs/run_rc_h593_m_priority_dispatch_v1.py`；宿主 `hkn1993.localdomain`，PID `2451554`，在独立工具调用中确认持续运行。
- 原四卡调度器 PID `2386443`：核对宿主、UID、完整命令后正常终止。原冻结脚本未修改；新的监督进程持有原 watch lock，防止重复接管。
- 当前运行的 `5158740` 四卡批次正常收尾，其全部清单内的 M 分片暂时从新队列排除。
- `5158760_[42-69]`：28 个 inside 分片；`5158761_[56-69]`：14 个 visual 分片。
- 两组均为 accelerated，每片 1 GPU、8 CPU、64GB、12 分钟。真实 scontrol 字段已核验；提交 spool 与原 shared-pooling launcher 字节一致。
- 全局上限为 46 张 GPU，过渡期 42 个短分片加当前 4 张 GPU。空位自动补齐，inside:visual 约 2:1，一类完成后另一类使用全部余量。不依赖整批结束后的 CPU 回调排队。
- 提交后首次检查两组均为 PENDING；不能把并发上限报告为已经获得 46 张 GPU。

2026-09-23 04:26 Europe/Berlin 快照：inside 已验证 37/593，visual 50/593；当前四卡批次仍在推进它已接手的工作。

## 科学范围和后续

优先完成 M 内部组成和图像输入干预。inside593 全部通过原验证后，自动释放现有 `5158704`，由原链路做固定头/重训头的五折检索读出，避免只报告 M 的直接排名。两类原 join 同时保留。

融合训练和坐标精度扩展暂停续提；现有四卡批次已经接手的分片正常完成。旧融合数组 `5158421_[48-92]` 保持 held，避免远端旧监督器重复提交。所有旧 checkpoint、中间特征、token 权重和统计文件保留，未删除实验臂或缩短训练步数。

原 593 图、自然 C128、FP64、分组划分、干预定义和完整验证门保持不变。正常分片若未完成 C128，按原 part 文件续跑；失败时停止续提，不将异常片认作通过。最多每 query 32 次尝试，监督期限 14 天（含排队），仅做有界执行维护。

## 时间解释

此前 90–120 小时预算针对四张卡完成四类全部扩展。现在优先两类 M 图像计算，其剩余工作按试运行折算约 120 GPU 小时。若能持续获得 46 张 GPU，有效工作量约 2.6 小时，考虑启动、收尾和 CPU 读出，可按约 3–4 小时纯运行时间规划。此为条件估算，非排队或完成时间保证；其余扩展仍有工作量，没有凭空消失。

## 可核查记录

- `registry/rc_h593_m_priority_dispatch_authority_v1_20260923.json`
- `results/rc_h593_m_priority_dispatch_v1/preflight.json`
- `results/rc_h593_m_priority_dispatch_v1/takeover.json`
- `results/rc_h593_m_priority_dispatch_v1/waves/0000.json`、`0001.json`
- `results/rc_h593_m_priority_dispatch_v1/spools/5158760.sh`、`5158761.sh`
- `results/rc_h593_m_priority_dispatch_v1/status.json`
- `results/rc_h593_m_priority_dispatch_v1/supervisor.out`、`supervisor.err`

准备检查实际覆盖：语法、46 总配额扣除旧四卡、完整任务排除、尝试上限、原冻结启动参数、spool 字节一致和登记后才 release。新调度器不读取身份标签来挑选分片。
