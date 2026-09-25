# POST内部M：出现1次依赖真实配对质量的probe纠错

2026-09-24。5162570按用户要求由cpuonly移到dev_cpuonly，保留8CPU、16GB、10分钟；最终COMPLETED/0，用时1分18秒。三条GPU训练臂和CPU refit均已正常完成。初始pilot的调度失败来自后续CPU任务提交数量上限，工程验证本身通过，依赖链已恢复。

固定ColNomic自然C128、TRAIN16、128次完整候选损失更新；比较LLM前与LLM后/检索投影前的位置。RoMa保留，末端三参数头不直接读M。probe8与训练组分离，但此前已经反复用于开发分析；本轮不是新的独立确认。

| 路径 | TRAIN16 | probe8 | probe对RAW救回/误伤 |
|---|---:|---:|---:|
| RAW | 8/16 | 4/8 | 0/0 |
| PRE真实M原定联合终点 | 10/16 | 4/8 | 0/0 |
| POST真实M原定联合终点 | 10/16 | 5/8 | 1/0 |
| POST真实M，固定适配器后TRAIN-only重训头 | 12/16 | 5/8 | 1/0 |
| 外部ADDITIVE4 | 12/16 | 6/8 | 2/0 |

POST的恒定M训练、错绑M训练，以及同一真实模型/同一头的恒定或错绑推理，probe均4/8。各自或共用真实头的预定refit结果也均为4/8。因此这次新增纠错有内部真实M绑定的对照支持，不是只增加适配器参数就共有的结果。

## 新增纠错具体发生在哪里

新增目标为`cefdinir-fig2`（H593-90acde9b0567a472f4232120），原RAW答案`gabapentin-fig11`。

| 内部路径 | 正确目标动作分数 | 决策 |
|---|---:|---|
| PRE真实M | -0.385122 | HOLD错误原答案 |
| POST真实M | +0.360524 | SWITCH到正确目标 |
| 同一POST模型/头，内部M改常量 | -0.611695 | HOLD |
| 同一POST模型/头，内部M错绑 | -1.156864 | HOLD |

PRE和POST真实M的正确目标都排在内容第一；关键区别是相对原winner的内容分差从0.005578扩大到0.032254。联合头的内容项贡献从+0.123476变为+0.840273，目标最终超过固定HOLD=0。没有根据probe手动更改阈值。

这支持一个有限但直接的结论：**在本实验配置和这张跨组开发probe上，M经LLM后的表示调制路径，已能转化成实际身份纠错。** 用户要求的“内部有效即可”出现正向证据，无需先超过外部6/8。

这不说明所有错误已解决。Infants仍选择错误ChildsIbuprofen：POST正确目标动作分数+1.573466，错误候选+1.808085；该问题仍属于强错候选竞争。另有1张目标不在C128，仍计入8张分母。

## 验收与边界

- 全部24图零适配器GPU回放token误差0；未重跑RoMa、视觉编码器或LLM。
- 独立重建24图、12条评分路径、36,576个挑战者logits，最大误差2.665e-15；HOLD/SWITCH和身份计数与正式join一致。
- 原定联合终点与预定TRAIN-only refit均救回同一张，未根据probe挑选checkpoint或阈值。
- 只有8张、6个component且已打开的开发面板，不能据此宣布POST普遍优于PRE、稳定广泛泛化，或将新数字并入H593主成绩。
- POST改变的是LLM输出表示进入检索投影时的处理；没有改变LLM内部注意力，也未建立空间ownership。

完整各臂表：[正式结果](REPORT_POSTLLM_M_V1_20260924.md)。
逐图分数分解：`results/rc_postllm_m_v1/probe_mechanism_decomposition.json`。
独立复核：`results/rc_postllm_m_v1/independent_audit/validation.json`。
