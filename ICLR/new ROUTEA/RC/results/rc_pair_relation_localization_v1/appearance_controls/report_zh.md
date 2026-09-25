# 既有配对读出与 H71 融合：归因辅助核算

本记录只核算已经完成、已经打开标签的结果。不训练、不运行 GPU、不修改全量 POST 实验。依据最新取舍，ColNomic 固定关系读出只保留为辅助诊断，不扩展为新的研究分支。

## 1. 原折的质量与内容确实具有不同的错误分布

范围：H593 原 fold0 held119、原 ColNomic 自然 C128；113 张 target 在池内，6 张候选召回失败保留在总分母。RAW 为 98/119。表中每一项直接对全部128个候选取最大值，**没有训练头、没有 HOLD/SWITCH、没有选阈值**，不能作为 COST1 结果。

|直接排序依据|正确/119|相对 RAW 救回/损失|15 张池内 RAW 错误中 target > RAW winner|原 COST1 的9张救回中 target > RAW winner|原9张中 target 全128第一|
|---|---:|---:|---:|---:|---:|
|原 RoMa M|78|10/30|14/15|9/9|6/9|
|ColNomic 自由内容 L0|99|1/0|2/15|1/9|0/9|
|ColNomic 固定 soft-return|88|4/14|6/15|5/9|3/9|
|ColNomic 固定集中度（1−entropy）|86|2/14|2/15|0/9|0/9|
|COL_PAIR 已训练质量 M，单独取最大|8|1/91|10/15|6/9|0/9|

这张表最可靠的意义是**错误条件下的互补性**：原 M 单独识别率并不优于 L0，却能在 L0/RAW 困难的候选竞争中提供另外一种相对排序。M 直接替代检索同样会大量误改，所以不能把“M 更大”或“M 单独判身份”作为原 COST1 的解释。

固定 soft-return 有少量局部正信号，但整体低于 RAW，且会损失14张原正确；没有稳定泛化或可替代 RoMa 的证据。这里只保留现有核算，不据此新增模型或训练。

## 2. 为什么 learned COL_PAIR 的 M 排名只有8，而固定 return 是88

两者读取的底层关系来自同一组冻结的 ColNomic 最终 tokens，**不是 LLM 前视觉 tokens**。`rc_pair_quality_core_v1.py::relation` 先计算余弦矩阵 S，再以固定温度0.1求行、列 softmax A、B；return 为两个方向的返回概率乘积求和，entropy 也在学习模块之前计算。

本轮从封存 prediction 的 `query_relation_summary`、`reference_relation_summary` 读取这些统计，逐文件校验 SHA、128 候选顺序、对应 M 与已封存分析一致。已静态核对该统计的计算位置；**未新增图像前向，也未在本轮重新计算全部关系矩阵**。

质量 MLP 对520维 token 关系向量做 520→16→1→sigmoid，再以两侧均值的几何平均形成 M。训练目标是 MLP 与五参数决策头共同满足 COST1，既没有直接监督 M 的身份排名，也没有要求 M 复制 RoMa。

因此，8与88说明当前学习映射没有保留固定 return 的独立排序行为；它不证明实现错误，也不证明 ColNomic 缺信息。尤其不能把 M 单独排名8误写成六臂最终模型8：COL_PAIR 加完整决策头的 held 结果是96/119，TRAIN 是435/457。训练拟合、目标选择、读出容量与跨组泛化仍没有被这张直接排序表单独分开。

L0 与历史 RAW 也应区分：L0 在原 ColNomic token bank 上进行 FP64 L2 normalization 和 mean-MaxSim；RAW 来自冻结的全图库候选供应结果。两者实际排序不完全相同（99与98），本表并未将它们混用。

## 3. H71 融合负结果并非“适配器根本没动”

范围：预先固定 execution ordinal 0–70 的71张、37组、原五折；seed17、每折240步、自然C128、原七参数 COST1 联合训练。50个配置均已完成。FREE 五种来源均51/71；ROMA_WEIGHTED 五种来源均57/71。对应 RAW 为51，原 COST1 为57。

本轮重读所有50个最终模型和 optimizer checkpoint、710份逐查询预测，并用 FP64 重算127个 challenger logits。全部与封存预测一致。40个可学习适配器的输出层范数均非零，最小为1.227；optimizer 二阶矩非零，保存的 query gate 接近设计上限0.1。相对 NO_ADAPTER：

- FREE 分支内容 S 的最大绝对变化约0.00653–0.00838；最大 logit 变化约0.269–0.335。
- ROMA_WEIGHTED 分支 S 最大变化约0.000346–0.000557；最大 logit 变化约0.453–0.487。
- 所有正确性向量与对应 NO_ADAPTER 一致。仅 COARSE_FINE/ROMA_WEIGHTED 出现一次“错 reference 换成另一个错 reference”；不能说所有最终选择完全一致。
- 冻结原头、只替换融合特征的代数回放仍只有这一次错误候选变化；只换训练后头没有改变最终选择。

这排除了“零梯度/没有应用适配器/只保存了未变化结果”这一类解释。它没有排除更有表达能力的融合。

当前融合使用单图 coarse/fine 特征在 ColNomic patch 网格池化，固定随机压缩到128维，经过 128→16→128 线性低秩模块，以 `1+0.1*tanh(...)` 对最终 ColNomic tokens 做逐通道乘法。它既受±10%幅度限制，也不能任意写入新的描述子方向；适配器本身没有 query–reference 交互。短240步、小面板、输入压缩和受限融合仍然是明确的解释边界。

## 4. 剩余边界

现有六臂与 H71 不能单独确认“原 RoMa 的收益只能来自匹配过程、不能来自额外表示”。COARSE 臂用的是池化后且随机压缩的特征，不等于保留全部原生描述子的单图外观对照。更不能由学习失败推出输入表示缺失身份信息。

本轮归因主线仍应是原 COST1 的 RoMa 配对证据及其受控干预；本记录提供已有对照究竟排除了什么、尚未排除什么。它不提出新的有效模型，也不声称新的跨组规律。

## 可复核文件

- `pair_trace_recount.json`：原119张、每候选的10种统计及 source SHA。
- `candidate_rank_diagnostics.json`：上述直接排序的逐查询选择、target/RAW/最大值及救回/损失清单。
- `fusion_training_effect_recount.json`：50配置的权重变化、optimizer证据、gate、内容/logit变化及 head-swap。
- `programs/analyze_rc_pair_relation_appearance_controls_v1.py`：上述trace与融合核算。
- `programs/summarize_rc_pair_relation_fixed_rankers_v1.py`：固定排序核算。
