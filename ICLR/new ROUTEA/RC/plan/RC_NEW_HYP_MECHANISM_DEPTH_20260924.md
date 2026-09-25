# new HYP：从有效性推进到机制解释

日期：2026-09-24。用户要求继续解释原因，指出仅整理增益、绑定和迁移仍不够深入。本轮不重新采集图像、RoMa 或 LLM 特征；固定已有候选、折划分和封存模型，使用 CPU 读取缓存。

## 已有解释及真正缺口

已有机制链不应丢失：RoMa 的粗阶段两图交互入口干预，在固定 41 张面板上经 M_ONLY 路径使原 3 次纠错全部消失；只删除后续置信度细化则仍保留 3 次。55 张输入干预中，reference 低通只改变 M 已使原 5 次纠错消失，ColNomic 内容保持不变。它们回答实际模型的计算依赖，但不把离分布置零变成架构普遍必要性证明。

另一个被忽略的历史事实是：R5-V6 全图库、8 张无标签 query 的 RoMa 分数存在明显 reference hubs。4 个身份进入所有 8 张的 RoMa C128，跨 query 分数 Pearson 约 0.401。该实验是上游候选源 STOP，不允许重开其保护标签；这里只读其既有报告。它提示 M 可能混有 reference 普遍可匹配因素。

因此，当前不能从“错绑 M 掉分”直接跳到“收益完全来自真实两图交互”；错绑也会破坏 reference 主效应。内部 POST 的 5/8 同样不能直接解释为更好地定位了 patch：一个共同的质量方向，经归一化也可能改变所有 patch 的相似度。

## A. 分离真实 M 的 reference 主效应与配对剩余量

数据复用：`results/rc_h593_simple_explanations_v1/cache.json`，H593 原五折、自然 ColNomic C128，已有 COST1 PRODUCT5 头。没有新增视觉数据。

每折只在原 TRAIN 候选边上、不读身份标签，按固定数值规则拟合：

    log M(q,g) = mu + a_query(q) + b_reference(g) + residual(q,g).

使用真正的 physical reference key，不混淆身份去重。保持原 RAW、自由内容 L、冻结头和 HOLD=0，比较：原 M；只有 reference 主效应；去掉 reference 主效应后的 M。保存同一候选轴、全部 L/M/127 个 logit、逐图动作与计数，以及源文件 SHA。数值超范围或未见 reference 不可悄悄裁剪/丢弃，必须明确记录固定处理、覆盖率及分层结果。

诊断的含义：如果仅 reference 主效应复现大部分收益，“两图交互”的表述应收缩；如果去主效应仍保留收益而主效应不足，则排除了这个具体的 reference 先验解释。冻结头下降可能来自尺度/分布变化，不据此断言任何分量不可替代；如需重训对照，按同一原配方另行封存，不按 held 结果挑参。

注意：这是自然 C128 条件下的加性 log 主效应模型，不能排除全部非线性单图因素；未观测 reference 的回退必须单列。已有六臂单图网络不是这个分解的替代。H593 是已打开的开发数据，原五折仍隔离拟合与留出，但不改称独立确认。

## B. 分离 POST 内部的共同质量方向与 patch 特定响应

固定 `rc_postllm_m_v1/POST_REAL` 的 128 步端点、原 joint head、TRAIN16/PROBE8、原自然 C128。不重新训练、不调阈值、不按 probe 选 checkpoint。

令 H_real 和 H_const 是实际 BF16 加残差后的 hidden states。先转 FP32 取差：

    D_i = H_real_i - H_const_i = mean_patch(D) + (D_i - mean_patch(D)).

以 H_const 为共同底座，回放 CONSTANT、REAL、COMMON_ONLY、PATCH_RESIDUAL_ONLY、RECOMPOSED。都经过同一冻结检索投影（含原 LoRA）、归一化及完整 reference MaxSim。检查 CPU 原生预测与封存 GPU 端点的数值/决策差，检查重组是否恢复 REAL；不能默认 BF16 与设备之间逐位相同。

再固定 CONSTANT 内容匹配的 reference 命中位置 j0，计算 REAL_FIXED_ARGMAX，并沿用同一头回放。由此区分更新内容相似度与改选命中位置的贡献。对每个候选有精确的评分分解：

    L_real - L_const
      = mean_i[(z_real_i-z_const_i) dot r_j0(i)]
        + mean_i[max_j(z_real_i dot r_j) - z_real_i dot r_j0(i)].

第二项非负，但它不是身份收益保证；错误候选也可以因重新取最大值获益。保留逐 patch 相似度/命中、各候选 L、全部 logit 和身份结果，按 TRAIN/PROBE 分开报告。

如果 COMMON_ONLY 和固定旧命中就保留纠错，内部路径可以通过共享质量方向和归一化后的度量变化起作用，不需要将成功解释为 patch 定位。若 patch 特定部分确有额外作用，再说明具体路径和边界。能量占比不等于决策贡献，二者都要报告。

## 数学解释不冒充新理论

正数且 epsilon=0 时，sym(M_g,M_w)=tanh((log M_g-log M_w)/2)，query 共同尺度会抵消。此处揭示主效应分解为何能直接连接实际决策；实际 epsilon/floor 仍按原代码执行并数值核对。乘积对比恒等式以及接口不可恢复性早在 9 月 9–10 日的理论文件中已有，不重新包装成这次的发现。

共同向量也能产生不同 patch 响应。对 z=p/||p||，小扰动 d 的一阶变化是 (I-zz^T)d/||p||。因此输出 patch 变化不自动证明学会了空间关注。POST 不修改 LLM 的 attention；它发生在 LLM 之后、检索投影之前。

最终应回答：整体支持中哪些成分在帮助候选判定，以及内部路径通过什么计算把该成分变成内容差距。实验阴性同样会限定解释，不能预设必须支持更复杂结构。

来源：`REPORT_H593_SUBSET41_FROZEN_PATH_INTERPRETATION_20260923.md`、`REPORT_H593_VISUAL55_FINAL_INTERPRETATION_20260923.md`、`REPORT_R5V6_ROMAV2_FULL_GALLERY_SOURCE_E0A_STOP_20260905.md`、`NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md`、`NEW_HYP_THEORY_DEFINITION_AND_PROPOSITIONS_V2_20260910.md`、`REPORT_POSTLLM_M_V1_20260924.md`。
