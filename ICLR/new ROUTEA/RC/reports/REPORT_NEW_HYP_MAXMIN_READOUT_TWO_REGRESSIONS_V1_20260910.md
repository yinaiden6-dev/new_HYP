# new HYP：0220/0618 的固定参数计算分解

仅分析已验证5139151中的两条指定样本，原 RAW C128、REAL scorer；没有重训、调参、选模型或改变 DIR9。
六组完整127 logits 均与原封存值逐bit一致。反事实保留各新头的原六列系数及 bias，仅把第七输入置零；它不是新的可采用成绩。

| Query | 头 | 实际动作 / physical row | 实际最大 logit | 第七列置零动作 / physical row | 置零最大 logit |
| --- | --- | --- | ---: | --- | ---: |
| OUTCOME-0220 | ORIGINAL7 | SWITCH / 1024 | 0.012491441 | SWITCH / 1024 | 0.012491441 |
| OUTCOME-0220 | MAXMIN8 | SWITCH / 1024 | 0.068484026 | SWITCH / 1024 | 0.200173643 |
| OUTCOME-0220 | PROJECTION8 | HOLD / 1311 | -0.455664682 | SWITCH / 1024 | 0.140877233 |
| OUTCOME-0618 | ORIGINAL7 | HOLD / 3234 | -0.010972946 | HOLD / 3234 | -0.010972946 |
| OUTCOME-0618 | MAXMIN8 | SWITCH / 3430 | 0.647513709 | SWITCH / 1325 | 0.535654627 |
| OUTCOME-0618 | PROJECTION8 | SWITCH / 3430 | 0.234169369 | SWITCH / 1325 | 1.137327003 |

## OUTCOME-0220

RAW winner physical row=1311；target physical row=1024。

MAXMIN8 在实际最强 challenger physical row=1024 上：
原头 logit 0.012491441；重训后的旧六列+bias 净变化 +0.187682202；新增项 β×feature -0.131689617；完整 logit 0.068484026。
置零第七列后 final_correct=True；完整头 final_correct=True。

PROJECTION8 在实际最强 challenger physical row=1024 上：
原头 logit 0.012491441；重训后的旧六列+bias 净变化 +0.128385792；新增项 β×feature -0.596541915；完整 logit -0.455664682。
置零第七列后 final_correct=True；完整头 final_correct=False。
在已重训系数固定的这一次计算中，加入第七项改变了正确决策；不能据此断言只重训旧六列一定保持正确。

## OUTCOME-0618

RAW winner physical row=3234；target physical row=3234。
target 就是 RAW winner，不在127个 challenger 中；其 HOLD 动作分数按原规则为0，不能伪造 target 对自身的 logit。

MAXMIN8 在实际最强 challenger physical row=3430 上：
原头 logit -0.309953413；重训后的旧六列+bias 净变化 +0.310589388；新增项 β×feature +0.646877735；完整 logit 0.647513709。
置零第七列后 final_correct=False；完整头 final_correct=False。
第七项置零后仍错误，因此这次退化不能全部归因于新增项的直接贡献；旧六列及 bias 的共同重训已改变决策。

PROJECTION8 在实际最强 challenger physical row=3430 上：
原头 logit -0.309953413；重训后的旧六列+bias 净变化 +0.513147222；新增项 β×feature +0.030975561；完整 logit 0.234169369。
置零第七列后 final_correct=False；完整头 final_correct=False。
第七项置零后仍错误，因此这次退化不能全部归因于新增项的直接贡献；旧六列及 bias 的共同重训已改变决策。

精确有理数分解使用所有输入和参数的原binary64端点。记录了矩阵乘加舍入误差及其差，
不把分项求和强行当成原FP64执行顺序。每个候选的完整分解及三组127 logits 保存在 result.json。
产物SHA：`0368a7fd221be6fbe1d83b9c6fc29b3b274c293b2ef134d38bed22b5a67466f3`。
