# V4 冻结适配器的 CPU 机制分解

本次仅读取 PRE_REAL 第128步冻结权重、TRAIN16/PROBE8 已有 merger tokens 与各自自然C128的 M，未读取身份标签、未训练、未做 LLM/RoMa 推理、未提交作业。CPU限制为2线程。

## 核心发现

V4 确实在 LLM 前改变了每个 query patch 的特征；但当前学到的 **M 特异变化主要是所有 patch 共享的特征方向偏移**，不是明显的逐区域差异化调制。

| 冻结 PRE_REAL 第128步 | TRAIN16 × C128 | PROBE8 × C128 |
|---|---:|---:|
| 比较的 candidate pairs | 2048 | 1024 |
| M 特异残差的共同向量能量占比，中位数 | 99.4293% | 98.6903% |
| 共同向量占比最小值 | 92.9327% | 93.8035% |
| 所有 pair 合并后，空间变化部分能量占比 | 1.4598% | 2.1477% |
| M 特异残差 / 原视觉特征 RMS，中位数 | 1.6373% | 1.4368% |
| 总 adapter 残差 / 原视觉特征 RMS，中位数 | 7.8368% | 7.1840% |

3072对中，每一对的 M 特异残差都至少90%属于 patch 共同向量。这里的共同向量可以随 query 内容与 M 改变；并非所有图片共享同一个固定偏置。

这支持更窄的解释：当前模块把配对质量编码成了广泛作用于 query 表示的条件信号。它可能由此实现候选质量校准；仅凭内部路径的纠错或内容排序变化，还不能声称学会定位身份文字、重新选择关键区域，或获得新的语义辨识能力。

## 实际结构

对同一 candidate 的全部 query patches，输入的是同一个标量 M，不是 RoMa 的空间可见性图，也不是 reference token。reference 的最终检索 tokens 保持冻结。

令视觉 merger token 为 x_p，训练集标准化的 log M 为 z，当前增益 g = sqrt(3584) ≈ 59.8665：

```text
r_p(M) = 0.1 × [W_up GELU(W_content LayerNorm(x_p) + w_M × g × z + b_down) + b_up]
x'_p = x_p + BF16(r_p(M))
x'_p → 原冻结 LLM → 原检索投影 → 单位范数检索 tokens → 自由 reference MaxSim
```

因此，理论结构确实允许通过 GELU 使不同 patch 对同一个 M 作不同响应。它没有显式修改 attention logits，也没有输入哪个 patch 更可信的空间位置提示。本次分解是在测量已学到的实际响应，而不是仅凭结构推断有效性。

## 分解定义与校验

对每一 query–candidate 对，在同一个适配器权重下定义：

```text
delta_p(M) = r_p(M) - r_p(z=0)
common(M)  = mean_p delta_p(M)
vary_p(M)  = delta_p(M) - common(M)
E_total    = mean_p ||delta_p||²
E_common   = ||common||²
E_spatial  = mean_p ||vary_p||²
E_total    = E_common + E_spatial
```

z=0 是现有 constant control 的准确函数定义，不是原始 M=0。差分去除了与 M 无关的 adapter 调整及 up bias。表中共同份额来自每对的 E_common/E_total；合并空间份额是所有 pair 的 E_spatial 总和 / E_total 总和，不是份额简单平均。

全量结果通过16维 bottleneck 的 up-projection Gram 矩阵计算，避免重复大矩阵展开。对24张图各自 candidate0直接展开3584维残差核对通过。另用原始适配器类，在第1张 PROBE 的 M 最小/中位/最大3个候选上调用真正 forward，并用 up 层 hook 捕获 BF16 cast 前的输出独立核对：

- 能量最大相对误差 1.123×10⁻⁷。
- 共同份额最大绝对误差 4.524×10⁻¹⁰。
- 正交能量等式闭合误差约 10⁻¹⁷。
- 这3个候选的 BF16 实际输入差分也保持高度共同化；数值见 `direct_validation.json`。

## L2 归一化意味着什么

原检索投影对每个最终 token 做 L2 归一化。若只在最终投影输出上乘一个正标量，归一化会消掉该尺度，不能直接实现 M × L。

当前 adapter 加的是3584维方向残差，并且位于非线性的冻结 LLM 之前，所以不会被这一尺度不变性简单消除；它能够改变最终检索向量方向与相似度。因此“内部 M 只是把最终 token 乘大”不符合实现。

但是，最终 token 都有单位范数，并不排除以向量方向编码 candidate 质量。一个几乎共同的输入方向偏移，经 LLM 和投影后，也可以使许多匹配相似度随 M 整体变化。需要区分“真实经过内部表示路径”和“已经学会更好的局部内容辨识”，这不是同一个结论。

## 尚不能从本次分解推出

- 表中98.69%不是 attention 权重占比，也不是最终 token 变化占比；它是 **LLM 前、BF16 cast 前** 的 M 特异残差能量分解。
- 冻结 LLM 的非线性与跨 patch 交互可能将共同输入偏移转为不同 patch 的输出变化；本次没有对 attention 或所有最终 token 做干预分解。
- 空间变化能量小不等于其作用小；少量方向变化可能改变排名。
- 高共同份额不证明当前内部读出与某个外部标量校准函数严格等价；它只为这一简单解释提供明确机制依据。
- 现有 TRAIN16 与已打开 PROBE8 不能证明普遍泛化或 H593 上的内部改造收益。

若之后继续验证这一机制，最直接的对照是冻结当前模型，在相同 C128 上分别保留共同残差、空间变化残差，再比较最终内容与动作。该对照需要重跑受影响的 LLM 前向；本次未执行。

## 文件

- `result.json`：24张图、3072个候选的完整分解，源文件SHA及汇总。
- `audit_adapter.py`：2线程、无标签、无编码器推理的复现程序。
- `direct_validation.json` / `verify_direct_projection.py`：原适配器 forward hook 与直接投影核对。
- 实现来源：`programs/rc_prellm_m_adapter_v1.py`、`programs/rc_prellm_m_scaled_adapter_v4.py`；冻结权重 `results/rc_internal_m_condition_scale_v4/PRE_REAL/snapshots/0128.pt`。
