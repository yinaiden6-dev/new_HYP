# 恢复暂停的统一采集

2026-09-27，用户要求“71张缓存？现在先启动原来暂停的统一采样”。以最后封存的128张恢复方案为本轮范围，不扩大到593，也不恢复旧200配置融合训练。

## 实时核查

- 最后恢复方案：`results/rc_h128_unified_resume_v1/authority.json`，自然执行索引0–127。
- 旧GPU 5160470、CPU 5160477、汇总5160478均已取消，当前队列没有该采集任务。
- 原三个分支通过验收的数量分别为inside42、visual56、coordinate71；三者共同完成42张：0–40与70。
- 补缺索引41–69、71–127，共86张；57张新增匹配坐标，使后续QR/QRR完整输入可由71扩到128。已有41–69坐标和部分其他分支由原采集器复用。
- 原代码、模型profile、worker轴和128恢复authority源SHA全部核对一致；旧统一controller及finish70 controller的GPFS锁当前空闲。

## 执行修复

不改封存科学代码或旧文件。原128 wrapper只修了同Job续跑回执重名，没有带上已在query70验证的重复reference内部hook修复。新v2 GPU wrapper同时保留两项修复：

1. 同一个query/reference图片对的matcher缓存命中时，如果本次还需要采集内部head通路，则复用单图描述子、重放该matcher，保证内部hook执行；核对前后matcher输入及输出一致。已有最终张量不能自行重建未保存的内部hook输入。
2. 每次续跑回执包含restart序号，避免同Job ID覆盖不可变记录。

原pair capsule、分支part和完成标记继续复用。每query原GPFS锁保证只有一个写者；已经完成的分支不重算。原CPU export及三分支验证器继续使用。

## 调度

GPU仍为原配置：1GPU、8CPU、64GB、15分钟切片、同Job最多31次requeue，普通accelerated最多46并行，dev_accelerated承接其中一个待运行元素。CPU导出14个pack，每个8CPU/16GB/10分钟，依赖采集完成；最终128三分支验证依赖所有CPU导出。

任务先hold提交、逐项检查资源与提交spool再release。具体新Job ID写入本轮提交记录。科学完成必须以128张三个原分支全部验证PASS为准，不能以Slurm正常退出替代。

## QR/QRR修复分支

冻结B_CAL训练起点的v2代码与15项旧基线精确回放已准备。按用户最新优先级，本轮不提交新关系模型训练。现有F71协议保持独立；采集完成后使用F128需要另行冻结面板和训练协议，不能把新增样本直接混入已经定义的F71结果。
