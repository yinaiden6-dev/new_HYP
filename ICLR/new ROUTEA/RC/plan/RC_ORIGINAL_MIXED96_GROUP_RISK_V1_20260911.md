# 原训练协议上的单因素组均衡

用户明确要求继续推进且不再重训原ec7基线。原头和旧32/128全部160条预测直接取已完成并独立验证的fixed269封存；不调用旧train_head训练基线，也不重算其预测作为研究前置步骤。

唯一新头GROUP_MIXED96：保持原PAIR64、原FULL TRAIN32、PAIR原64顺序、FULL原execution顺序、4128x6原FP64特征（哈希16739684a5902783b0793750998ed86d7e5cc89464fd9d424de5d288521ab6ee）、原正确检索标签、6特征加bias、零初始化seed17、AdamW .03/.001及2000步。训练函数从原源码精确抽取，仅替换两个mean。

PAIR保留原HOLD错误成本4、SWITCH成本1的BCE；FULL保留原SIGN损失及错误成本4。PAIR20组内每行权重1/(20*n_g)，FULL12组内每query权重1/(12*n_g)；两个池各自总质量1，总损失仍PAIR+FULL（1:1）。不把96行合并成一个组均值，不删除PAIR，不改变训练图、不加特征、无超参搜索。

现有2x2实验已检验PAIR有/无与SIGN/RANK，并完整复现过ec7。它没有检验本次组均衡；组jackknife删组也不是此条件。直接复用历史结果，避免重复基线训练。

仅训练新头并新进程重放新头；160图预测完全封存后才开启评价标签。两个旧固定面板均报告，RAW C128与127-challenger HOLD/SWITCH不变。以GROUP_MIXED96相对冻结ec7的救回减损失为观察净增标准，允许损失；没有自动替换模型。CBIND为附加诊断。仅有已查看开发数据，不是独立外部确认，也不宣称统一理论成功。

Slurm accelerated/1GPU队列分配，实际CPU8，16G，10分钟，UTC2026-09-11 16:00截止。单任务自动fit/replay、join、independent review及报告。唯一入口slurm/rc_original_mixed96_group_risk_v1.sbatch。
