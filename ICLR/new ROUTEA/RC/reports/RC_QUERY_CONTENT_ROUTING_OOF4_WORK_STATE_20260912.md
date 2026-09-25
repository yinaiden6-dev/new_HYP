# 2026-09-12：照片内容条件化TRAIN分组检验已提交

完成补记：5142573、5142574全部四折及5142577现均COMPLETED 0:0。81,280 logits与动作/排名/计数独立验证通过。TRAIN128四折RAW86、BASE108、GLOBAL114、STATS107、QUERY108、QUERY_PERM107。QUERY主臂8救8损无净增；GLOBAL副对照7救1损、净+6/6正组1负组，精确组p=.0625。前三query主成分保留均值描述方差28.4%–31.4%，不代表身份信息保留率。当前无待监控任务、无新EVAL或部署变更。报告REPORT_QUERY_CONTENT_ROUTING_OOF4_V1_20260912.md，机器结果与result_validation.json均已生成。以下为执行历史。

最新补记：四折中的0、1已完成0:0，分别69秒与71秒；新参数fresh重放和独立NumPy logits/动作核对通过（22225与17145 logits，最大误差约3.2e-14/5.7e-14）。2、3已开始运行。名额释放后，5142577已通过scontrol实际迁移到dev_accelerated,accelerated并申请1GPU、实际CPU计算，仍为同ID/10分钟/完整afterok依赖。追加记录results/rc_query_content_routing_oof4_v1/join_partition_amendment.json。下文cpuonly是最初提交状态。

用户在照片类型/细粒度证据讨论后明确继续。本次执行一项固定TRAIN128四折检验，没有启动旧EVAL32/128复测或其他593训练。

缓存5142573在dev_accelerated完成0:0，用时52秒。原runtime全图image tokens与grid数量、图片SHA、tokenSHA、RAW C128轴均核对；128维FP64均值在新进程以NumPy重新计算通过。无需重跑图像编码器或RoMa。

四折5142574_[0-3%4]已提交dev_accelerated,accelerated，8CPU/16G/1GPU分配，实际CPU计算，每折10分钟。提交核对时0、1已RUNNING，2、3尚待排程；开发分区运行名额限制可能使其串行，但普通accelerated也可调度。不承诺四折同时运行。

汇总5142577已提交cpuonly，无GPU，10分钟；调度器实际依赖为afterok:5142574_*，要求完整数组成功。dev队列最多4个提交名额，四折已占满，故汇总先放普通CPU队列。后续如开发名额释放可仅迁移该作业分区，不更换模型或任务ID。三个提交脚本均与本地launcher逐字节相同。

四臂为GLOBAL7、STATS28、QUERY28、QUERY_PERM28；原四折BASE7直接复用，原头训练更新0。所有新臂同FULL-C128交叉熵/FP64/2000步/零残差初始化；QUERY为预定主臂。query描述只在本折TRAIN做PCA3和规范化；打乱对照只在TRAIN/heldout各自内部置换query描述。没有使用人工照片类型、数字转录、EVAL救损标签、SAM或OCR训练。

各折自动训练→fresh-process参数/预测重放→独立NumPy logits/动作核算。汇总在四折预测及独立验证全部封存后才读heldout身份标签。报告救回、损失、净增、MRR及来源组统计，允许损失，不恢复零损失门。

当前证据仅为缓存与合成工程资格、训练提交/启动；尚无新识别结果。128是TRAIN分组留出，不能与原EVAL128的99/128或LISTWISE的101/128混为同一数据。此内容摘要检验即使失败，也不排除全部照片条件化；它尚未实现候选细粒度内容差异读取器。

入口：

- 计划：plan/RC_QUERY_CONTENT_ROUTING_OOF4_V1_20260912.md
- 权威：registry/rc_query_content_routing_oof4_authority_v1_20260912.json（SHA 3b2dbbadffae73dca8f6213f9d3015f26ccfaf007ce61a691a58754ddc423249）
- 程序：programs/run_rc_query_content_routing_oof4_v1.py
- 独立核算：programs/validate_rc_query_content_routing_oof4_v1.py
- 输出：results/rc_query_content_routing_oof4_v1/
- 提交记录：submission.json；缓存：cache_receipt.json、cache_validation.json
- 各折：fold00..fold03/{fit_predictions.json,validation.json,independent_validation.json}
- 预期汇总：result.json、result_validation.json、report.md（未生成前不视为已完成）

既有固定面板保持：原ec7为28/32、99/128；LISTWISE_UNIT1为26/32、101/128；本轮不修改这些数字或自动替换部署。
