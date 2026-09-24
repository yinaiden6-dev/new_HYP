# ColNomic 内部 M：v2 预处理修复与补充对照

继承 v1 的四臂、16 TRAIN / 8 probe 清单、原自然 C128、真实 RoMa M、冻结原检索 LoRA、原 fold0 COST1_REFIT_M1Q0R0 头、16 次更新及所有门槛。v1 程序、authority、失败产物均保留。

5161411 在训练前触发 SOURCE_TOKEN_PARITY，四臂均未训练。独立诊断 5161422 固定同一 TRAIN 图和同一模型，仅恢复历史默认 torchvision 图像预处理，就使原 token 平均余弦从 0.947473824 恢复至 1.0；FP16 缓存 tokens 完全一致，L 最大差约 1.13e-11，原决策一致。v2 因此只删除强制 PIL 的 use_fast=False，不更换模型、LoRA dtype、attention 实现或候选，不放宽校验门槛。每张新缓存仍须逐项验收；首图通过不代表全部工程验收通过。

新目录：results/rc_prellm_m_adapter_v2。修复诊断、原 authority 与全部复用代码都写入新 authority 的来源清单。四臂通过完整缓存和真实梯度验收后才自动提交训练，固定终点后封存预测，再在 CPU 上完成下述对照及统一标签核算。

来源验收复用已完成的全文件 SHA，跨节点仍逐项检查 inode、size、mtime_ns。登录节点和 GPU 节点的同一 GPFS 挂载 st_dev 实测分别为51与50，其余三项完全相同；设备号属于节点本地挂载信息，只记录而不作为跨节点文件改变的判据。科学输出一致性门槛全部不变。

补充对照见 RC_COLNOMIC_INTERNAL_M_ADDITIONAL_CONTROLS_20260924.md：

- 已有完整 fold0 TRAIN 的 ADDITIVE4 固定头重算：零模块、四臂及错绑推理，共同检查读出依赖；这是辅助诊断，没有重训内部模块。
- 纯 L 排名：不显式使用末端 M，保留全部候选分数；最终统一使用目标标签核算。它仍可能包含内部 M 编码的质量，不能自动解读为语义改善或区域定位。
- SCORE_UPDATE5：从同一个完整 TRAIN 原头出发，只训练零初始化五参数残差，使用完全相同的 TRAIN16、COST1、顺序、16 次更新、AdamW lr3e-4 / wd1e-3 / clip1。与内部模型共享既有监督起点；同更新预算不保证各参数化都充分优化。

上述新增对照只使用现有分数和将生成的缓存，不增加 GPU 训练臂，不增加 RoMa 前向。所有补充预测也须在读取 probe 标签前封存。核心位置比较是 PRE_REAL−PRE_CONSTANT 与 POST_REAL−POST_CONSTANT 的差异，同时报告对应基线的救回与误伤。8 图探针只支持开发方向检查，不据此选择参数或宣称跨组普遍优越。
