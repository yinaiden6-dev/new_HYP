# ColPali 原生 C128 评分改用 CPU

用户询问将5160876剩余评分迁到cpuonly。评分复用已保存的1030×128 ColPali tokens与RoMa整体M，不运行编码器或RoMa。

原评分程序强制CUDA且旧任务请求1GPU，不能只更改分区。保留原程序、authority和已完成结果；新执行包装只将评分函数的CUDA可用性断言改为CPU断言，并将两个tensor.to设备字面量从cuda改为cpu。AST验证必须恰好3处变化，其余评分、FP64、候选轴、TopK、统计量、trace和写入流程均不变。

CPU资格验证先在隔离目录回放已完成GPU首片的query ordinal0、50，每图完整128候选。对全部分数、5项统计和trace浮点值核对，最大误差容差2e-9，RAW winner/候选轴必须一致。记录argmax索引差异；不能宣称CPU/GPU逐bit一致。只有资格通过才运行剩余分片。原首片结果不覆盖。

剩余CPU任务沿用原分片1–49、每片11–12query，上限50并行，原验证文件齐全的query直接复用；逐query原子写入，约180秒预算/260秒进程超时/5分钟Slurm时限，至多16次同Job requeue。每次执行保存额外CPU执行authority和query验证绑定。

资格通过后才替换仍PENDING的旧GPU片：提交不请求GPU的CPU数组并先hold，核对脚本和资源，修改五折拟合5160877的afterok为新数组；完成核验后取消对应旧GPU待运行副本并释放新数组。若旧GPU已开始运行，先重新核查范围，不能取消正在运行的科学计算。

CPU/GPU设备变化不改变候选来源（ColPali自身C128），不改模型/监督/下游损失，不插入正确答案，不读取评分阶段的身份标签。CPU资格耗时用于估计分片，不套用GPU2分钟实测。
