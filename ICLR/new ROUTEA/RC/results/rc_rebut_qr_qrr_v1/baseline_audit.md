# 当前真实计算图与基线

2026-09-27。本轮首先保留全593原始481/492对照，新增QR/QRR使用71个完整对应缓存；没有替换原系统。

## 数值定义

原RoMaV2 match输出双向overlap_AB/overlap_BA。原cache将其按ColNomic image-token几何框取cell mean，得到u(query tokens)与v(reference tokens)。它们是已投影的支持响应，不是新训练出来的identity概率。

M0=sqrt(mean(u)*mean(v))，双侧各自token均值，分母分别为各自token数；原score路径没有新增mask裁切或高置信token选择。本轮严格复用缓存，不引入新的区域/分母定义。新增F层mask与原M0独立，不能用新mask重新计算/替换历史M0。

S0=M0 × sum_i u_i max_j[v_j cosine(q_i,r_j)] / max(sum_i u_i,1e-12)。自由内容L0=mean_i max_j cosine(q_i,r_j)，原头Lcur=S0/max(M0,1e-12)，二者不同。b为冻结检索产生的自然RAW候选分数；原头使用相对RAW winner的差，按该C128的RAW标准差缩放。

原六输入按候选与固定RAW winner计算：RAW gap、对称S差、对称M差、对称Lcur差、query-roll控制差、reference-roll控制差；外加bias。原control与operator有自己的具体roll实现，本轮直接复用已逐位回放的native_X，不自行换roll。原reference MaxSim仍自由搜索全reference，warp不硬绑定ColNomic的内容匹配点。

代码：`programs/run_romav2_colnomic_visibility_xf_six_case_v1.py`、`programs/rc_quality_content_operator_v1.py`、`src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py`；具体五折参数来自封存loss_binding/fold*/payload.json，而不是文件里的单套示例部署常量。

## 实际训练与部署路径

|路径|训练参数|冻结参数|目标/推理|
|---|---|---|---|
|原481|6权重+bias|ColNomic、RoMa、原分数与候选|COST1；最高challenger>0时SWITCH，否则HOLD|
|492 GAP_BIAS2|原头固定，拟合gap系数和接受偏置|原头候选排序及两backbone|仅重新放行原HOLD的最高challenger，旧SWITCH不变|
|496 CONTENT18|18参数含额外内容统计及原特征平方项|backbone、固定特征/C128|历史FULL128 CE，[-64,64]有界FP64优化；held不用于参数拟合|
|原internal POST_REAL|后LLM适配器+INTERNAL3小头|缓存LLM hidden、检索projector、RoMa|M经过表示支路；末端不直接读M；不同于只训练外部头|
|本轮四新臂|共同18参数头；QR/QRvec/QRR另训练局部读出|backbone/原M0/原内容/candidate axis|统一新hit/miss loss，内层选择epoch与tau，全部候选竞争|

原COST1在RAW正确时压低最危险错误logit；RAW错且target在候选时提高target并压低最危险错误logit（softplus形式）。原源码训练2000步AdamW lr=.03、weight_decay=.001。新计划的identity-set CE+缺失候选softplus是新协议，不冒充原COST1。历史CE的候选缺失处理与新协议也须分开，不仅数字对齐就称单因素。

## 已验证结果与输出

全593：RAW426；原481；GAP492；历史CONTENT18为496。原自然候选召回570/593。全部结果都是已多轮开发的grouped OOF，不是新未触碰测试集。

`../rc_h593_m_481_492_comparison_20260927_v1/validation.json`独立核对两头593×128候选分数、CSV、网页及分类计数。原481重算最大差1.78e-15；492全部127非HOLD logits逐位回放一致。

`historical496_predictions.json`保留496全部593的完整128分数；`common_features.pt`为只读18维旧基底、无target标签。prepare重算原496最大差1.07e-14。训练进程不加载这份预测做初始化；它只用于后续封存评测比较。

当前协议是固定图库、query身份/组件/图像分组。`fold_exposure_audit.csv`显示所有held身份reference都曾在TRAIN候选负例中出现，不能声明完全未见reference身份泛化。

人工“可见信息是否足够”没有完成盲审；数值failure类型不能替代照片盲审。当前步骤1已完成计算层归类，人工视觉原因仍标未决，不伪造人工一致性。
