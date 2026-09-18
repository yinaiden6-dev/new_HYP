# new HYP方向历史核查更正

2026-09-11。用户质疑“原来没做过”后，扩展到RC之前的RouteA以及9月8日V7/V8。此前只检查近期J/端点/标量头，范围不足。撤回将两项描述为此前未探索的新科学方向的判断。本次未提交新实验。

## 参考端重复计票：设计与训练均有前例

- 9月8日 `plan/RC_HYP_IDENTITY_INFORMATION_RESEARCH_DECISION_V1_20260908.md:60` 已写相同摘要不可恢复丢失信息；72–86行明确同一reference cell重复指派不增加质量、128D向量信息、SuperGlue容量与DeepEMD预算。当时完整预算方案仍未定义shared readout，不代表完整实现过。
- V7/V8落实了reference分组去重。`src/rc_aslo_xf/reference_conditioned_coherent_residual_p_only_v8.py:56` 使用unique reference indices、组内amax真实witness、组间mean及全局max。V8自然数据POSTHOC TRAIN32训练512updates，识别9/32、NO-GO；`results/rc_coherent_residual_development_v8_validation/result.json`为独立验证PASS。它是旧P-only开发，不是原RAW action的28→9。
- `reports/REPORT_RC_V8_FULL_FAMILY_CLOSURE_AND_WRONG_IDENTITY_CERTIFICATES_20260909.md:74` 已诊断candidate各自的reference分组数和分母会改变query观测权重，甚至反转逐点支配；共同query测度后仍有wrong证书，不能把去重当未经尝试的充分解法。
- 更早RCDE的 `src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py:1299` 已有column_load与inverse_crowding，是明确针对reference列使用拥挤度的实现，非仅术语相似。

## 逐token有符号证据＋可靠性：已有实现与训练

- 8月2日父目录 `reports/REPORT_ROUTEA_C6DF1C_F1D_MECHANISM_DIRECTION_DECISION_20260802.md:38` 明确同一winner/challenger的逐patch signed contrasts、shared token encoder、reliability gate、signed contribution及全patch读出。
- 父目录 `scripts/c6df1d_selector_core.py:111` 已实现sigmoid可靠性、tanh有符号贡献、全patch聚合、跨view均值/方差和基线差额。
- job5041811实际完成资格训练检验，`../results/route_a_c6df1c_target_free_crossfit_selector_v1/formal_job5041811/result.json`为C6DF1C_SELECTOR_MECHANISM_NO_GO，validation为C6DF1C_QUALIFICATION_VALID；0救0损。672为4×168资格条目，来自224图，非672张独立测试图；selector在其训练examples上评价，未进入F1D外层正式评价。
- 更早C6d已有signed local目标训练，C6d-E也实际比较过signed与negative reject。因此“保留负号/学可靠性/局部候选竞争”都不是新的科学原理。

## 仍须保留的精确区别

V7/V8以RoMa既定对应下的残差分组为主；C6d-F1C读取D1/A10/A20局部状态。此前提议的“当前冻结RAW+RoMa/ColNomic、完整reference联合软容量、保留原action”的完全相同组合，尚未找到完整实测。

这属于旧机制换输入/接口的受控复验候选，不足以宣称新理论；旧条件的失败也不能直接证明新组合必败。提出任何后续之前，应明确与已做算子、输入、损失、对照和失败原因的差异。不能仅凭新文件名或局部实现不同宣称是新方向。
