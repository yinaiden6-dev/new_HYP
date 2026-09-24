# ColNomic internal-M pilot v2 — 2026-09-24

固定模型：ColNomic 7B（原检索 LoRA 保留，主干冻结）。数据：已打开的 H593 开发集，原始 fold0；16 张 TRAIN（8 张 RAW 正确、8 张 RAW 错误）及按预冻结 SHA 顺序选出的 8 张 outer-held 探针。

候选：原始自然 RAW C128。评分：冻结 COST1_REFIT_M1Q0R0，所有实验臂的输出评分都继续使用真实外部 M。PRE 在 LLM 前调制视觉 token，POST 在 LLM 后、检索投影前调制；REAL 与 CONSTANT 使用相同参数量和训练顺序。

四臂固定训练到第 16 次更新；所有预测和 127 个 challenger 分数先封存并经 NumPy 独立重算，再读取 held 标签。没有按探针结果选 checkpoint。

独立重算最大误差：1.07e-14（阈值 1e-10）。正确性按 gallery identity 判断。

实际组数：TRAIN 16 个 component；PROBE 6 个 component / 6 个 group。探针与完整 fold0 TRAIN 的 component、identity 分离，存在 group 元数据时同时核验 group 分离。

## TRAIN16

| 路径 | 正确/总数 | 对 RAW 救回/破坏 | 对旧外部 M 救回/破坏 | 对旧外部 M 改动决策 |
|---|---:|---:|---:|---:|
| RAW | 8/16 | 0/0 | 0/1 | 2 |
| OLD_EXTERNAL_M | 9/16 | 1/0 | 0/0 | 0 |
| FRESH_ZERO | 9/16 | 1/0 | 0/0 | 0 |
| PRE_REAL | 9/16 | 1/0 | 0/0 | 0 |
| PRE_CONSTANT | 9/16 | 1/0 | 0/0 | 0 |
| POST_REAL | 9/16 | 1/0 | 0/0 | 0 |
| POST_CONSTANT | 9/16 | 1/0 | 0/0 | 0 |

C128 含目标：16/16。每条救回/破坏的 query ID、原正确样本损失率保存在 result.json。

## PROBE8

| 路径 | 正确/总数 | 对 RAW 救回/破坏 | 对旧外部 M 救回/破坏 | 对旧外部 M 改动决策 |
|---|---:|---:|---:|---:|
| RAW | 4/8 | 0/0 | 0/2 | 3 |
| OLD_EXTERNAL_M | 6/8 | 2/0 | 0/0 | 0 |
| FRESH_ZERO | 6/8 | 2/0 | 0/0 | 0 |
| PRE_REAL | 6/8 | 2/0 | 0/0 | 0 |
| PRE_CONSTANT | 6/8 | 2/0 | 0/0 | 0 |
| POST_REAL | 6/8 | 2/0 | 0/0 | 0 |
| POST_CONSTANT | 6/8 | 2/0 | 0/0 | 0 |

C128 含目标：7/8。每条救回/破坏的 query ID、原正确样本损失率保存在 result.json。

## 内部 M 错绑诊断

只在推理时把内部 M 按候选轴循环错移一位；外部评分的 M 保持真实绑定。没有训练错绑模型，因此只解释推理敏感性。

| 子集 | 诊断 | 正确/总数 | 相对对应 native 救回/破坏 | 决策改动 |
|---|---|---:|---:|---:|
| TRAIN16 | PRE_REAL_SHUFFLED | 9/16 | 0/0 | 0 |
| TRAIN16 | POST_REAL_SHUFFLED | 9/16 | 0/0 | 0 |
| PROBE8 | PRE_REAL_SHUFFLED | 6/8 | 0/0 | 0 |
| PROBE8 | POST_REAL_SHUFFLED | 6/8 | 0/0 | 0 |

## 证据边界

这是拟合可行性与已打开开发集的小样本探针。8 张图片及其实际组数不能支持泛化确认，也不据此自动扩大实验、选择超参数或宣称优于完整五折 COST1。

结果：[result.json](../results/rc_prellm_m_adapter_v2/result.json)；验证：[validation.json](../results/rc_prellm_m_adapter_v2/validation.json)；每张图完整 C128 L、127×4 特征和 127 logits：[independent_numpy](../results/rc_prellm_m_adapter_v2/independent_numpy)。

补充 CPU 对照已在 held 标签读取前封存：[对照报告](../results/rc_prellm_m_adapter_v2/cpu_controls/report.md)。
