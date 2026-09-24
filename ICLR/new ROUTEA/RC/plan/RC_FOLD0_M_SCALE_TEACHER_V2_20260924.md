# 修复 M 拟合后进入原第一折跨组复查

启动前置条件：`rc_colnomic_m_scale_fit_v2` 两臂固定2000步完成，再按明确记录的V3共享标度校准，至少一臂同时通过原四项TRAIN拟合门。按预先顺序FULL优先、否则BALANCED，禁止看held表现选配方。V2未校准两臂未通过的结果保留；V3结果单列。未通过就不创建执行authority、不提交训练。

复用原fold0划分：457条自然C128包含目标的TRAIN用于M教师拟合；held119保留6条目标不在C128的query。该held曾被历史实验使用，是既有开发折复查，不是新的独立确认。

ColNomic tokens、完整候选轴、原PAIR relation、8353参数网络均固定。重新seed17初始化，不复用四图已训练权重；输出偏置设为457条TRAIN教师log(M+1e-8)均值对应的logit。AdamW lr=.001、wd=.001、同原2000次query更新及顺序。FULL全程原log MSE；BALANCED前256步原log MSE、之后4倍中心化项加1倍平均偏差平方，与小面板配方一致。

学生训练只读取TRAIN M。held无M输入、无身份输入，全部M预测和u/v追踪封存后才读held教师/身份进行评估。报告TRAIN/held质量误差、排序、候选M离散度、原头救回/误伤及候选绑定打乱。

网络冻结后，先预测457条TRAIN的M，再拟合一个共享标量b，使TRAIN平均log误差为0，M'=sigmoid(logit(M)+b)。增加1个校准参数；不按query分别校准。held仅应用这个冻结b。保存校准前后M及TRAIN标量的来源；u/v追踪对应校准前M。推理不需要教师数据。

两条冻结头回放必须分开解释：

- ORIGINAL7：只替换M，仍保留RoMa原局部权重，用于定位整体质量的作用，不能称为完全去掉RoMa。
- FREE5：自由ColNomic内容匹配＋学生M，学生推理无需RoMa；冻结五参数头不重训，先检验原质量证据能否被替代。

CPU打包full-token relation仅改变批量执行方式，首轮已经核对前向/梯度，新增权重4损失通过梯度检验。复用已有tokens和M，不启动GPU、RoMa、编码器作业。不落盘整个457条的巨大派生relation库；输入tokens、确定性relation代码、检查点、逐候选M及held u/v追踪均保存。

dev_cpuonly，8CPU/32GB、每段10分钟，450秒主动断点，最多48段。训练/封存完成后自动汇总；意外失败停止。新结果不覆盖原教师失败结果或COST1主结果。训练预算内未出现可信跨组效果也完整报告，不据此证明信息论上的缺失。
