# new HYP：固定支持漏读与错误候选正证据的TRAIN结果

5139121已完成，COMPLETED 0:0，2分18秒。固定前4条TRAIN、每条C128，共512项、1,024次LP全部完成；独立精确端点证书512/512合格。没有EVAL读取、模型训练或新准确率。

| TRAIN query | 原支持margin | 优化可达margin | 错误候选正margin数/127 | 最优值明确高于目标的错误数 | 目标支持token数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DIFFICULT-0101 | -0.00125697 | 0.17824540 | 121 | 6 | 5 |
| DIFFICULT-0128 | 0.03378575 | 0.35235348 | 127 | 0 | 6 |
| NDV2-007-P01 | 0.10730407 | 0.33190517 | 127 | 1 | 4 |
| NDV2-007-P04 | 0.10517488 | 0.34828395 | 127 | 0 | 5 |

DIFFICULT-0101在固定支持下为负，允许重新分配非负支持后为正，直接证实原支持漏读可区分证据。其余三个目标原来已经为正，提升margin不能称新增识别成功。

全部512项中506为正，其中正确目标4/4、错误候选502/508。各候选选择自己的支持，因此几乎每个错误候选也能找到局部有利证据；不能用存在正支持作为身份判定。上表比较的只是候选各自最优支持值，并非共享action的预测或4条TRAIN的新准确率。

这把两个问题分开：原支持确有损失；恢复可分性仍不足以确认身份。后续唯一比较是将优化值作为第七列接入旧六特征与原共享训练，与同容量固定支持列对照。原28/32模型保持不变，是否保留28并新增必须由完整已打开EVAL32的封存预测回答。

候选支持集中在4–6个query tokens是数值求解所得的可行见证，不证明唯一解必然稀疏、不证明物理对象区域，更不证明这种集中导致错误。

证据：
- results/rc_reference_support_maxmin_train_pilot_v1/result.json：d2f059554bbe62e74dd112b7802449063dae4aa74204a0906c86e82917c941b0
- 同目录validation.json：75bf7b3c73bafa4e5279912e40ae2be9e94db79ade0cf7b69e862870d13e9426
- results/rc_reference_support_maxmin_train_pilot_description_v1/description.json：bbb8e4a15af7a6b57a764ebe86e80081de9f33b62fe5abb823b7a9f67ac4fdc8
- registry/rc_reference_support_maxmin_train_pilot_completion_receipt_v1_20260910.json
- 后续固定计划：plan/RC_REFERENCE_SUPPORT_MAXMIN_READOUT_V1_20260910.md
