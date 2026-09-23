# H593 打包依赖修复

5153195在导入openpyxl时失败，5153215因此显示DependencyNeverSatisfied。原因是此前仅在允许用户site-packages的登录环境检查依赖，批处理却设置PYTHONNOUSERSITE=1；openpyxl只存在于用户目录。

修复通过新增隔离依赖目录与包装启动器完成，不修改封存训练、预测、两份publisher源码或原authority。openpyxl 3.1.5与et_xmlfile 2.0.0使用固定文件清单，pandoc使用固定共享路径。env -i加PYTHONNOUSERSITE=1下的Excel/HTML/DOCX预检已通过。

六项消融汇总包已实际生成，ZIP/解压文件全部哈希一致，封存result.json与源结果完全一致。RAW/COST1_FULL/CE_FULL仍为426/481/486。

原失效等待任务5153215已取消；5153235在开发分区重新验证实际计算节点打包环境和已生成汇总包；5153236依赖afterok:5153208:5153235，使用新启动器生成联合汇总包。提交时尚未宣称这两个新Slurm任务完成。

[消融汇总包](new_hyp_complete_results_20260920_ablation_v1/new_HYP_complete_results_20260920_ablation.zip) · [消融报告](new_hyp_complete_results_20260920_ablation_v1/H593_ablation_zh.md)
