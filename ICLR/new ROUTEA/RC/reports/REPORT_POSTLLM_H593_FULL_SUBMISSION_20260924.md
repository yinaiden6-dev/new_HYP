# 后 LLM 内部 M 改造：H593 全量扩展已提交

2026-09-24，按用户「先把内部改造，从8扩展到全量」授权执行。

## 范围与待回答的问题

从原 TRAIN16 / probe8 的后 LLM 试点扩展到原 H593 分组五折。每折独立训练适配器和 INTERNAL3 头，最终593张只计各自留出折一次，保留23张目标未进入自然C128的图片。原RAW为426/593，候选召回570/593。此面板已用于开发，属于全量分组OOF，不是新外部确认。

冻结 ColNomic 骨干与原检索投影，在LLM输出与检索投影之间加入质量条件适配器。真实RoMa M保留；最终头只读取RAW差、调整后内容差、偏置，不能直接读取M。目标是验证内部路径是否产生有效纠错，不要求超过外部M头。

三个训练臂为 POST_REAL、POST_CONSTANT、POST_SHUFFLED，均按原五折独立训练。同一个POST_REAL最终模型还做恒定M与错绑M的推理干预。本轮TRAIN新拟合的外部INTERNAL3、ADDITIVE4、PRODUCT5用于同源对照；历史NATIVE7的481仅列作历史参考。

## 缓存与训练量

- 593张的RoMa M、RAW、原C128及reference tokens齐全，不重跑RoMa，不重新编码reference。
- 已验证旧24张LLM hidden直接复用，仅补569张query hidden；每张新query只做一次原始编码前向，五折与三臂共享。
- 50个GPU编码分片，最大并行50。三臂×五折共15个训练作业，最大并行15。每次请求10分钟，预算到达后保存当前候选、参数、优化器与RNG，并在同Job ID续跑。
- 每折候选内TRAIN数量457、454、451、461、457；固定每图8遍，各臂总18,240更新，三臂合计54,720更新。每次完整评分C128，无候选抽样。
- 纯训练时间按试点计时推算约3.05 GPU小时合计；不含新编码、缓存读取、端点评估和排队，不是实际完工时间承诺。

## 任务链

首图验证 **5162615** 已提交；首次回读为PENDING/Priority，分区dev_accelerated,accelerated，1GPU、8CPU、64GB、10分钟。

后续实测：5162615在dev_accelerated完成，退出0，总用时2分58秒。旧缓存GPU投影误差0；首张新query与历史FP16 tokens逐值一致，原内容重放最大误差2.22e-16；新旧FP64内容差1.02e-11。首张新query编码及完整C128核查耗时2.56秒。

已自动展开 **5162617_[0-49%50]** 编码数组和 **5162618** CPU缓存验收，afterok依赖已回读正确。编码第0片允许dev_accelerated,accelerated共同排队，其余49片仍accelerated；全部保留1GPU、8CPU、64GB、10分钟与原数组并行上限50。当前已验证hidden为旧24张加新1张；全量训练仍待缓存验收后自动提交。

```text
5162615：旧缓存投影一致性 + 首张新query编码
  → 50片补齐query hidden
  → CPU独立验收593张缓存
  → CPU五折TRAIN-only头初始化
  → GPU首个完整C128更新与断点验证
  → 15个GPU训练作业及全部端点回放
  → CPU独立NumPy分数核算、标签汇总与报告
```

后继仅在工程验收文件PASS且父任务成功时自动提交；具体新Job ID写入`results/rc_postllm_h593_v1/dispatch/`。若提交后继暂时失败，父任务保存已有子任务receipt并同ID续跑，避免重复提交或破坏afterok依赖。达到重试上限仍会明确失败，不静默跳过。

## 已完成与尚未完成

数据来源、候选轴和分组核对已PASS。训练四项工程测试PASS，包括完整C128损失/梯度、tie梯度、RNG恢复、三臂零输出初始化。独立汇总五项测试PASS，包括原Torch与NumPy三种头、strict零阈值HOLD、并列选择、缺折封存时提前失败与篡改拒绝。Slurm包装脚本正常/预算退出/超时/硬错误/重试上限及半提交续跑测试PASS。

首图资格现已PASS；真实训练首步资格仍待缓存与五折warm头完成，**尚无全量内部模型准确率结论**。不将提交、单元测试或旧5/8结果当成H593成功。

## 固定入口

- [实验计划](../plan/RC_POSTLLM_H593_FULL_EXPANSION_V1_20260924.md)
- [封存authority](../registry/rc_postllm_h593_authority_v1_20260924.json)，SHA256：`4e244190e58a73b27239a400394ba362b9221d2d8cfd342f841f85ff59248bad`
- [输入验收](../results/rc_postllm_h593_v1/input_manifest_validation.json)
- [任务与续跑程序](../programs/dispatch_rc_postllm_h593_v1.py)
- [训练程序](../programs/run_rc_postllm_h593_v1.py)
- [独立核算程序](../programs/join_rc_postllm_h593_v1.py)

最终结果将写入`results/rc_postllm_h593_v1/result.json`、`validation.json`与`reports/REPORT_POSTLLM_H593_V1_20260924.md`；每图全部128个L/M/RAW、127个logit、参数快照与训练轨迹均保留。
