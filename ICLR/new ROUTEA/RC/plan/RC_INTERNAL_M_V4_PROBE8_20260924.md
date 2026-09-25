# V4冻结终点：跨组PROBE8探索检验

用户在V4完成后要求继续跨组验证。本版本只追加推理；V4源码、128步checkpoint、TRAIN16结果与原authority保持封存。

## 固定方案

- 复用原fold0的8张probe；与V4 TRAIN及完整fold0 TRAIN的component、identity隔离，存在group时同时验证group隔离。
- 这些probe已在早期内部M实验中打开；本轮不是独立未触碰确认，不能以8图宣称可靠总体泛化。
- 同一ColNomic自然C128、同一候选物理顺序、RoMa原M。无目标插入，无候选筛选。
- 主结果固定为V4 PRE_REAL第128步adapter和联合训练三参数头。推理换常量或错绑M时共用同一头。
- 训练错绑臂用同样第128步终点，训练常量臂复用数学等同的V3 PRE_CONSTANT。M输入增益、标准化、残差尺度全部固定。
- 同时报告RAW、未适配的INTERNAL3及在TRAIN16训练的外部ADDITIVE4/PRODUCT5。内部末端不含M，外部头含M。
- 次结果只使用已在TRAIN16完成的五个冻结refit头；REAL的常量/错绑干预还报告共用REAL-refit头的结果。无probe重训、不按probe择优。

## 执行与证据

两个GPU任务：REAL负责真实/常量/错绑三种推理；CONTROLS负责训练错绑与训练常量。每项单GPU、8CPU/64GB、10分钟，普通与dev GPU共同排队，按候选保存断点，最多8次同ID续跑。

CPU汇总依赖两任务成功，8CPU/32GB、10分钟。逐图128个L、127个logit、HOLD=0、候选轴、snapshot SHA保留。NumPy独立重算所有动作后封存，再读取curator核对身份与分组，报告原正确误伤、救回、目标缺席。GPU禁止读取curator、旧probe结果及保护数据。

现有8张merger/输入缓存已经齐全；无需重算RoMa或视觉编码器，只执行冻结LLM条件化推理。结束后不自动追加训练、超参搜索或593扩展。

主问题是内部真实M能否在未参与其训练的组上产生纠错，不要求超过外部12/16，也不把训练集成绩当probe基线。
