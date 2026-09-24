# 内部真实 M v3：固定多遍拟合实验执行协议

2026-09-24。用户授权顺序为“先将上轮实验推至 GitHub，然后开始下轮实验”。本轮按[已确认设计](RC_COLNOMIC_INTERNAL_M_LEARNED_USE_V3_DESIGN_20260924.md)实施第一阶段：保留推理时 RoMa，检验 ColNomic 内部利用真实 M 的训练可行性。不是蒸馏，不改变既有模型或封存 v2。

## 数据、信息与共同起点

- 沿用 v2 的 fold0 TRAIN16（原 RAW 8正确/8错误）、同一16个component、原自然 ColNomic C128和候选物理顺序、真实 M、reference tokens、已验收 merger 输入。所有16个target均自然在C128中，不插入target。
- Phase A 不读取8张已打开probe的标签，不做外部或五折确认；原完整 TRAIN 拟合头仅作为历史锚点，不把它的标签预算称为16张。
- CPU只在 TRAIN16 的原内容 L0 上拟合共同的三参数读出 RAW差、sym(L0)、bias，2000次完整TRAIN16均值 COST1更新、AdamW lr0.03、weight_decay0.001、FP64。随后冻结共同起点。内部适配器则按原manifest行次序逐query联合更新；两者的更新单位与实际计算量分别报告。
- EXTERNAL_ADDITIVE4和EXTERNAL_PRODUCT5由这个共同起点增加零初始化M/ML项，初始决策必须一致；各自再训练2000次。这是充分优化的低成本末端基线，不声称和GPU适配器计算量相同。
- PRE_CONSTANT和PRE_REAL的118304参数残差沿用原rank16、scale0.1、末层零初始化。共同三参数读出继续训练；内部末端**只能读RAW、sym(Lθ)、bias，不能直接读M或ML**。真实M只进入query内部适配器，reference自由MaxSim不改。

## 冻结训练设置

- 位置：merger后、LLM前，仅query分支；原ColNomic、原检索LoRA、投影层、RoMa及reference tokens全部冻结。
- 两内部臂固定128更新，即同一TRAIN16访问8遍。seed17；适配器FP32、lr0.0003；读出FP64、lr0.03；AdamW weight_decay0.001；gradient clip1。M的log标准化沿用原TRAIN16统计；CONSTANT使用标准化条件0。
- 保存step16/64/128参数。固定在16及128终点重新评估整个TRAIN16，初始终点由缓存完成。不得将不同query的在线loss比较当作全TRAIN学习曲线；step64只保存checkpoint。
- step128追加PRE_REAL模型内部M的候选循环错绑和常量干预，均评估原16图。没有额外训练错绑模型；只解释推理敏感性。
- 保留全部C128。先完整计算候选内容分数，联合求出读出梯度和对128个L的导数，再对非零内容梯度分支精确反传。必须与直接autograd核对两部分梯度。
- 恒定M臂的相同query前向可以复用一次；对全部reference仍计算独立MaxSim并累加精确梯度。不得把真实M臂也错误合并为单前向。

## 执行、检查点与停止

1. GitHub归档远端确认后，CPU共同初始化、外部基线及数学预检。
2. accelerated 单GPU pilot先检查原缓存来源、零残差一致性、实际LLM梯度以及一次完整更新。只有工程检查通过才自动提交两条内部训练路径；REAL复用其首步checkpoint，不重复算作新初始化。
3. 每次8CPU/64GB、1GPU、10分钟，checkpoint含适配器、读出、优化器、完整或部分128分数、步数、authority绑定。必须按原步原候选恢复；超时最多48次同Job requeue，不无限续提。
4. CPU准备与最终汇总不申请GPU。每次模型前向、完整更新、缓存读取、评估时间分别记录；不将排队时间当训练耗时。
5. 两臂完成后自动产出TRAIN阶段汇总；不根据TRAIN提升宣称泛化，不自动读取probe标签或扩到593。若数值不有限、来源漂移、梯度/前向不一致，保存失败信息并停止。

## 主要比较与结果口径

- PRE_REAL对PRE_CONSTANT：相同参数、起点、标签和更新，内部质量条件的增量。
- PRE_REAL对EXTERNAL：内部是否承接或超过同一16图上末端质量利用；需同时列读出/模块训练成本。
- 每臂列固定TRAIN的平均COST1、target名次及对最强错误margin、相对RAW救回/误伤、原正确损失率、改动决策数。正确性按gallery identity，HOLD保持原RAW，阈值固定0。
- 非零梯度、token变化或工程PASS不等于科学成功；拟合改善也不等于可推广。若内部路径尚未拟合，先定位数值/优化/表达限制；不能断言ColNomic缺信息、RoMa不可替代或LLM前注入无效。
- M可能通过内容分数编码整体质量；即使有收益，也不自动称为区域注意力、语义理解或ownership改善。

执行来源及checkpoint策略由新authority固定；代码、命令和实际Slurm spool核对后才发布提交记录。原v2及其Git归档不被覆盖。
