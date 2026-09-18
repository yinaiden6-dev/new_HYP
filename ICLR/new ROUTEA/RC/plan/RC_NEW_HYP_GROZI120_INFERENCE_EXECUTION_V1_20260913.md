# GroZi480冻结推理执行V1

承接已获用户明确授权的GroZi只读外部确认，数据计划`RC_NEW_HYP_GROZI120_FROZEN_DATA_V1_20260913.md`不变。120 reference、480 query已按预定清单提取；600图内部字节与RGB像素均无精确重复，与既有987-query公开ledger及5413 gallery行无字节重合。这里不声称已排除所有语义SKU重合或预训练暴露。

## 已完成的工程入口

Job5143664于2026-09-13完成（5分41秒）：4张旧TRAIN图的image/template token、所有5413 RAW分数及自然C128均逐bit一致；120张新reference已编码并通过独立CPU哈希、图格与物理行验证。结果`encoded_references/{legacy_bridge.json,payload.pt,receipt.json,validation.json}`。没有GroZi query forward或head评分。

五个全H593固定头已提前验证并封存，所有GroZi推理零训练、零校准。此次只运行已声明模型，不搜索新头、损失、阈值、特征或支持区域。

## 固定流水线

共60片，每片8条匿名query，严格480顺序。先运行片0的RAW（10分钟）→RoMa+固定head（15分钟），只检查工程完整性与数值；不会join其target或查看准确率。片0完成后才释放其余59片RAW，最多并行46，每片10分钟；全RAW完成后释放其余59片RoMa+固定head，最多并行46，每片15分钟。两大阶段不叠加成92并行。任何失败阻止依赖推进。最后只在全部60片预测封存并验证后一次性join480标签。

RAW使用原ColNomic模型/processor、原FP32全gallery MaxSim求和再转FP64；原5413 gallery物理行及714/715等价身份不变，再追加120新reference，名义5533物理行/5532身份。新reference保存原完整token序列（含template）供RAW使用，image token单独供C4使用。每个query自然C128后按物理行保存，完整127 challenger，不插target或改候选数量。

RoMa直接编译复用原已验证数值循环，source profile、模型权重、precise模式、seed17、TORCH_HOME、canonical geometry和C4均固定。CPU独立重算每片4096个C4标量并要求逐bit一致。head复用FC.candidate_feature构造原六统计；CBIND为完整证据在128物理位置轴循环位移64，RAW特征保持。每个固定head、REAL与CBIND都输出127 logits和唯一HOLD/SWITCH选择，NumPy独立核算误差<2e-10且动作全同。主/次模型的科学判据不因附带报告更多对照而改变。

## 输入验证和结果屏障

每个GPU生产阶段验证实际用到的模型/数据/源码绑定。RAW阶段不加载RoMa权重，RoMa阶段不加载ColNomic大权重或原RAW gallery；CPU重放验证其消费的已封存输入及小型代码/身份文件，不重复哈希未使用的大模型。所有相关SHA仍写在authority中，生产者源资格保持可追踪。

推理worker拒读curator_roles、D1-MI、formal target_join和报告。输出先逐片独立验证、再全量封存；最后join程序才读取唯一GroZi curator文件。全部自然召回缺失保留在480分母中。所有已注册臂完整报告，包括失败结果。

最终主检验及视频组bootstrap严格沿数据计划。只有完成全部预测及独立核算才能作科学结论；工程PASS、reference注册或提交任务不能宣称外部new HYP GO。旧28/32、99/128和H593 OOF数字不变。
