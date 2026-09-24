# ColNomic 内部 M 改造：启动记录（2026-09-24）

四臂：PRE_REAL、PRE_CONSTANT、POST_REAL、POST_CONSTANT。前者在 merger 后/LLM 前，后者在 LLM 末层后/128维检索投影前；结构均为3584→16→3584的118304参数残差模块。常量条件是TRAIN logM均值，仍训练内容适配器。

主干与原检索LoRA冻结；所有臂沿用真实RoMa M和同一个fold0 COST1_REFIT_M1Q0R0冻结头。与旧完整七参数头区分：这里的输出头是既有整体M简化版本的有效5参数，不重新训练七参数头。

最初4张盲选工程图都RAW正确，原清单和脚本已保留。16张TRAIN按原TRAIN身份监督分层为8 RAW正确和8错误、16个component；8个outer-held探针按预冻结SHA选出，不用目标标签挑选。只训练新增模块16次更新；原头使用过完整fold0 TRAIN，不能声称整套系统只用16图训练。

已完成：

- 真实安装版本的tiny ColQwen2_5及PEFT：11项接口/零初始化/缓存/梯度/严格权重加载测试通过。
- 24 query×127 logits与既有原头及独立NumPy复算：最大差2.13e-14。
- 32项scalar-VJP vs普通autograd检查：最大差3.55e-15；并列最大值的全候选梯度保留。
- 1950个唯一token文件与原来源校验，manifest独立重建通过。
- sbatch成功、失败、超时续跑、达到重启上限的4个分支已模拟执行；提交spool与源脚本逐字节一致。

已提交5161411：dev_accelerated，单GPU、8CPU、64GB请求、10分钟，支持最多16次原Job requeue。首次scheduler核对为PENDING(None)，不是已运行通过。通过真实7B新鲜编码与旧缓存一致性、零适配器等价和实际梯度/显存验收后，自动提交四臂训练及预测至accelerated；全部预测封存后CPU独立核算。

RoMa和reference tokens直接复用；每张query补一次merger/冻结LLM末层/原检索tokens及输入信息。POST臂复用末层状态，PRE臂只复用merger。两者优化更新数相同，计算成本不同，保存每任务运行时间。首/末训练步保存patch调制前后特征和token变化；每query保留完整128候选L/M与127挑战者分数。

目前没有新准确率或HYP GO结论。8图探针是开发方向检查，不足以确认可推广性；真实7B验收仍由首GPU任务执行。

文件：
- plan/RC_COLNOMIC_INTERNAL_M_PILOT_V1_20260924.md
- registry/rc_prellm_m_adapter_authority_v1_20260924.json
- results/rc_prellm_m_adapter_v1/input_manifest.json
- results/rc_prellm_m_adapter_v1/submission.json
- programs/rc_prellm_m_adapter_v1.py
- programs/run_rc_prellm_m_pilot_v1.py
- programs/rc_prellm_m_join_v1.py

启动后复核：5161411已在hkn0401进入RUNNING（采样运行43秒），模型来源文件哈希验收MODEL_SOURCE_HASHES_PASS；尚无真实7B查询回放通过记录，不把RUNNING当作工程验收完成。

后续状态：5161411 最终 FAILED（2分50秒，ExitCode 1:0），在训练前因 SOURCE_TOKEN_PARITY 停止，四臂尚未提交。原失败记录完整保留。

独立诊断 5161422 已 COMPLETED（1分23秒，ExitCode 0:0；程序数值诊断约47秒）。仅恢复历史默认 torchvision processor，首图 FP16 tokens 即逐元素一致、平均余弦1.0、L最大差1.13e-11；无须改权重或放宽门槛。见 [根因报告](REPORT_COLNOMIC_INTERNAL_M_LOADING_DIAGNOSTIC_20260924.md)。修复版写入独立 v2 程序/目录，仍逐图验收后才能推进。

新增的 CPU 对照方案已写入 [补充方案](../plan/RC_COLNOMIC_INTERNAL_M_ADDITIONAL_CONTROLS_20260924.md)：ADDITIVE4 重算、纯 L 排名、同 TRAIN16 的 SCORE_UPDATE5。它们复用分数，不增加 GPU 训练臂；当前仍没有内部注入的科学效果结果。

修复版已提交 **5161427**：dev_accelerated，1 GPU、8 CPU、64GB请求、每次10分钟，可断点续跑。初次核查 PENDING(None)。v2 authority、原始16/8清单字节一致、全部代码来源SHA以及实际提交spool一致性核对通过；FULL_C128_VJP、launcher五分支、CPU补充对照三项验证通过。通过24图原缓存一致性与实际梯度检查后，自动推进accelerated四臂训练/预测，最后CPU先生成并封存所有补充预测，再统一读标签。

新执行入口：[v2方案](../plan/RC_COLNOMIC_INTERNAL_M_PILOT_V2_20260924.md)；[v2提交记录](../results/rc_prellm_m_adapter_v2/submission.json)。当前无新科学结果。
