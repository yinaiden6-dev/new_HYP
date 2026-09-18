# 593图正向增量：偏置与增量绑定归因结果

593张、68身份、64组；RAW自然C128，完整127 challenger HOLD/SWITCH。所有基头和补偿均折内训练。这是开发复用交叉验证，不与旧99/128直接比较。

| 模型 | 正确/593 | MRR |
| --- | ---: | ---: |
| RAW | 426 | 0.790204 |
| BASE7 | 440 | 0.805633 |
| BIAS1 | 440 | 0.805633 |
| CONSTANT1 | 444 | 0.808316 |
| CONDITIONAL4 | 445 | 0.809849 |

自然C128召回：570/593。

所有预测经过新进程重训重放；最终动作、完整gallery排名、总正确数及分组净增再次独立复算。组bootstrap区间沿用producer，验证器未声称独立重算该区间。

- BASE7__to__BIAS1: 0救/0损；等权组差0.000000；组bootstrap95% [0.0, 0.0]。
- BASE7__to__CONDITIONAL4: 5救/0损；等权组差0.012488；组bootstrap95% [0.0026041666666666665, 0.025627367424242424]。
- BIAS1__to__CONDITIONAL4: 5救/0损；等权组差0.012488；组bootstrap95% [0.0026041666666666665, 0.025627367424242424]。
- BIAS1__to__CONSTANT1: 5救/1损；等权组差0.011620；组bootstrap95% [0.0013020833333333333, 0.024857954545454544]。

[机器结果](../results/rc_h593_increment_attribution_v1/result.json)，[独立复核](../results/rc_h593_increment_attribution_v1/result_validation.json)。
未自动替换旧部署，未将正净增自动宣布为普遍new HYP或外部确认。
