# H593 简单解释对照：已提交

本轮回答：COST1的收益能否由更简单的质量/内容融合或置信度门控复现。使用已有593张缓存、原五折、原ColNomic自然C128；不重新运行编码器或RoMa，不改现有主模型。

- 缓存整理：5161365，cpuonly/dev_cpuonly，已核对启动于dev_cpuonly。
- 五折训练：5161366_[0-4%5]，cpuonly，依赖缓存完成。
- 自动验收与报告：5161367，cpuonly，依赖五折全部成功。
- 每次8 CPU、16GB、10分钟；每完成一个模型保存，超时可在同Job ID续跑，最多8次。
- dev限制最多提交4个任务，最初五折数组被QOS拒绝；训练/汇总改为cpuonly后已成功提交并逐项确认依赖。未取消其他实验。

9种学习头×2种损失×5折；每臂有训练内部拟合和全TRAIN重训，另加固定规则形式的置信度门控。保留固定0阈值和TRAIN内选择阈值两套结果；内部验证误伤预算不代表held误伤保证。

模型含单通道、RAW加单通道、加性融合、乘积交互、同参数量平方项加性对照。对照不是ELViS或To Match or Not to Match的官方复现。

代码语法、缺失target、HOLD/并列选择、COST1标量损失、阈值预算、乘积对比定义均通过预检；真实Bash参数传递与调度器spool逐字节一致。训练及科学结果尚未完成。

计划：plan/RC_H593_SIMPLE_EXPLANATIONS_V1_20260924.md
产物：results/rc_h593_simple_explanations_v1
最终报告：reports/REPORT_H593_SIMPLE_EXPLANATIONS_V1_20260924.md（汇总通过后自动生成）

启动复核：5161365正式COMPLETED/0:0，46秒；593图缓存PASS，最大公式差4.440892098500626e-16。5161366的0–4五折均已在cpuonly运行；5161367依赖正常。
