# 统一假说检验：组风险与强基线补偿

593张、68身份、64组；RAW自然C128，完整127 challenger HOLD/SWITCH。所有基头和补偿均折内训练。这是开发复用交叉验证，不与旧99/128直接比较。

| 模型 | 正确/593 | MRR |
| --- | ---: | ---: |
| RAW | 426 | 0.790204 |
| ALL_BASE | 440 | 0.805633 |
| ALL_CONST | 444 | 0.808316 |
| ALL_COND | 445 | 0.809849 |
| SMALL_BASE | 444 | 0.810060 |
| SMALL_CONST | 446 | 0.811325 |
| SMALL_COND | 445 | 0.809849 |
| GROUP_BASE | 447 | 0.813432 |
| GROUP_CONST | 447 | 0.812519 |
| GROUP_COND | 446 | 0.811957 |

自然C128召回：570/593。

所有预测经过新进程重训重放；最终动作、完整gallery排名、总正确数及分组净增再次独立复算。组bootstrap区间沿用producer，验证器未声称独立重算该区间。

- ALL_BASE__to__GROUP_BASE: 8救/1损；等权组差0.012179；组bootstrap95% [0.003596735168426345, 0.022253787878787876]。
- ALL_COND__to__GROUP_COND: 4救/3损；等权组差-0.001511；组bootstrap95% [-0.015625, 0.011075367647058822]。
- ALL_CONST__to__GROUP_CONST: 3救/0损；等权组差0.004391；组bootstrap95% [0.0, 0.010569852941176471]。
- GROUP_CONST__to__GROUP_COND: 2救/3损；等权组差-0.005035；组bootstrap95% [-0.018489583333333334, 0.0062499999999999995]。
- SMALL_BASE__to__SMALL_COND: 3救/2损；等权组差0.006611；组bootstrap95% [-0.003605769230769231, 0.01953125]。
- SMALL_BASE__to__SMALL_CONST: 3救/1损；等权组差0.007812；组bootstrap95% [-0.0014204545454545455, 0.020951704545454544]。
- SMALL_COND__to__GROUP_COND: 4救/3损；等权组差-0.001511；组bootstrap95% [-0.015625, 0.011075367647058822]。
- SMALL_CONST__to__SMALL_COND: 0救/1损；等权组差-0.001202；组bootstrap95% [-0.003605769230769231, 0.0]。

[机器结果](../results/rc_h593_group_risk_strong_base_v1/result.json)，[独立复核](../results/rc_h593_group_risk_strong_base_v1/result_validation.json)。
未自动替换旧部署，未将正净增自动宣布为普遍new HYP或外部确认。
