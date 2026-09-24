# H593：Qwen 强重排与整体质量的联合校准

本方案服务于三项收口任务的第3项。先完成第2项简化头固定外部迁移，再提交本方案；当前准备不代表实验已运行。所有模型、训练预算和主比较在本轮新预测之前固定。已有H593结果曾被查看，本轮是分组留出的开发性机制检验，不是全新独立确认。

## 问题与最小对照

原Qwen3-VL-Reranker-2B直接logit排序为522/593，三参数CE校准为523/593。原COST1与Qwen存在错误互补，但按真值选择的并集不能部署。本轮只回答：用TRAIN身份监督学习一个联合校准头，M能否在强Qwen证据和自由内容之外提供实际纠错增量？不以七参数头的CPU耗时代替RoMa的端到端成本。

固定四臂，两种原损失、原五折，共40个拟合：

| 臂 | 非偏置特征 | 参数数 | 职责 |
|---|---|---:|---|
| QWEN3 | RAW差、Qwen logit差 | 3 | 逐折复现原强重排校准基线 |
| QWEN_L4 | RAW差、Qwen差、自由L差 | 4 | 内容增量及同参数量对照 |
| QWEN_M4 | RAW差、Qwen差、整体M差 | 4 | 配对质量增量 |
| QWEN_ML5 | RAW差、Qwen差、整体M差、自由L差 | 5 | **预声明主臂（CE）** |

主比较是CE_QWEN_ML5相对CE_QWEN3；CE_QWEN_ML5相对CE_QWEN_L4用于检查已有自由内容之后的质量增量。CE_QWEN_M4相对CE_QWEN_L4为同参数量对照。COST1同四臂为预声明的次比较。所有臂全部报告，不用held结果挑选新的主臂、阈值、损失或训练轮数。没有新增乘积或高阶交互项搜索。

## 数据、公式与训练

- 沿用H593自然ColNomic C128，593张均评估；目标缺席23张仍计入分母；每折训练仅使用原TRAIN中目标在C128内的query。
- 冻结Qwen的75904个原logit和ColNomic/RoMa已有缓存。原图、C128轴、RAW第一名、物理行号、五折、身份和来源component全部逐项对齐。只做CPU计算，无新增编码器或RoMa前向。
- 对RAW和Qwen，各自使用 `(score(challenger)-score(RAW winner))/max(std(score(C128)),1e-12)`，精确沿用原Qwen实现。
- M为既有两侧可见性均值的几何均值；L为未加局部权重的完整reference自由MaxSim。M/L差均为 `(a-b)/(|a|+|b|+1e-12)`，沿用简化头已验收缓存。
- 每头零初始化、FP64、AdamW、lr0.03、weight_decay0.001、2000步、8 CPU线程；采用原COST1或CE，无held阈值选择，固定HOLD=0。原Qwen本身和ColNomic/RoMa全部冻结。
- 仅读取当前折TRAIN身份标签，禁止读取held身份/正确性；不把其他折训练所得的OOF头分数作为训练特征，避免间接held标签泄漏。
- 每折先重训QWEN3两种损失，验证原参数/全部held候选logits数值一致及选择完全一致；通过后才拟合新增臂。原基线复现失败则停止该折，不以新基线悄悄替换523/477。

## 封存、统计与边界

源程序、协议、输入hash和五折清单先写入新authority。每个头保留参数、训练拟合统计、训练步数和断点；每折封存全部127个挑战者logit及HOLD=0、最终物理行号。五折全部完成并先写预测封存清单后，才统一打开curator核算。

报告每折和总体正确数、对RAW及Qwen的rescue/break/changed、MRR@C128、目标召回、身份/来源组数；给出component均衡净增及按component重采样的95%区间，同时给出query加权的cluster-bootstrap区间。主结果在原64个component隔离下产生，区间仅作为开发性不确定性描述；不把扫描多个次臂得到的最好数字当作独立显著性。

若主臂未稳定超过Qwen，结果就是该预声明联合校准方案未建立强重排之外的增益；不继续在held上修改阈值。不声称低成本、普遍优于强重排、RoMa必不可少或已完成外部HYP确认。

程序：`programs/run_rc_h593_qwen_quality_joint_v1.py`。输出：`results/rc_h593_qwen_quality_joint_v1/`。阶段为prepare、fit（逐折，可断点）、join；不包含GPU提交或自动扩展实验。
