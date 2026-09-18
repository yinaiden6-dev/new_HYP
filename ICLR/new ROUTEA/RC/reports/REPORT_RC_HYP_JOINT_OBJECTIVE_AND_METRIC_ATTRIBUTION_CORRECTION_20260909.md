# 联合目标与27/32、69/90的归因校正

2026-09-09。用户明确：识别更准确与HYP有效，两项都要。此前将“相似物体
识别更准确”单独作为最终目标、将P开发26/32描述成整体突破条件，表述不完整，
在此纠正。本报告不改变任何冻结门、执行authority、数据访问边界或运行任务。

## 最终目标不拆开

最终需要同一个冻结系统同时实现可靠身份识别，以及可检验的reference-
conditioned spatial hypothesis作用，并在独立评估中联合成立。P-only只是
当前研究阶段；它的独立排序能力不是整个系统的全部目标。不会用准确率
替代HYP证明，也不会用区域证书或工程PASS替代准确率提升。

## 哪些结果已经成立

旧冻结RAW+RoMa/ColNomic-C系统保留RAW base score/winner，并用七参数head
处理完整127challengers，执行至多一次HOLD/SWITCH。current-runtime内部
EVAL32为25→27，2rescue/0break；原历史运行24→27，3rescue/0break。
已打开difficult90为61→69，8rescue/0break。这些是真实系统识别收益。

新NATIVE7/C_PAIRED另有25→28的内部结果，但并未获准替换旧默认head：
live结果明示deployment_default=FROZEN_C、new_head_replacement_authorized=false。
不能把28/32与旧head的69/90合并描述成同一已采用版本。

## 这些收益为什么还不能直接归因于HYP

本轮重读原difficult90结果：

| 路径 | 正确数 | 原REAL的8个rescue保留数 |
|---|---:|---:|
| RAW | 61/90 | — |
| REAL纠错系统 | 69/90 | 8 |
| C_BIND错误reference控制 | 50/90 | 0 |
| Q空间控制 | 70/90 | 8 |
| R空间控制 | 69/90 | 8 |

这支持specific-reference内容对收益有作用；现有空间控制没有消除收益，
因此尚不能确认这些纠错依赖所主张的空间HYP机制。该结果不证明所有空间
信息无用，也不是“RoMa无效”。原结果明确strict_spatial_causal_claim_authorized=false。
“来自采用RoMa的系统”与“收益由已验证的空间HYP带来”必须分开。

## 为什么不能画27→23→9的系统曲线

P-only屏蔽base score/rank/winner/gap，不调用旧七参数action，是新增的proposal
资格任务。其TRAIN32和上述EVAL32实际按query_id比较交集为0，属于不同面板。
原P TRAIN32绑定J1 ledger SHA为
e574e01febf5f6be35a876cb7e22d2cec819d5cdad6007d4612d1eea2e9e5e87。

P分支V6的23/32、V7的7/32、V8的9/32可以比较为该分支的真实退步；它们
不能直接改写完整旧系统27/32或69/90。当前路线比“对27/32系统补一次
空间消融”增加了建模要求：新proposal必须自己承担区域生成及独立区分。
这解释了为什么不能指望自动继承既有系统的准确率。

26/32来自本轮P开发需比冻结query-only22/32净增至少4，是一项必要条件；
它不是超过旧系统27/32的判据，也不是完整HYP的充分条件。最终仍需要
完整系统的同口径性能和HYP机制证据联合通过，不能拼接不同面板的正结果。

## 本轮来源

- `results/romav2_colnomic_difficult90_frozen_regression_v1/result.json`，SHA
  bdb962e4b07e6b27a86a35723465141c1f34f4b6dbda8f41810adb5199b79f31。
- `results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json`，SHA
  591b9787403417efa774e7bdb264291f7e8e4a827bfd4f479256d4e7a8e134c3。
- `reports/REPORT_ROUTEA_OMNIMEMORY_SYSTEM_RECONCILIATION_AUDIT_V1_20260908.md`，
  已有系统目标、P-only屏蔽边界与各模型版本关系。其P分支状态是历史快照，
  不能用于覆盖本对话已完成的V6—V8及后续诊断结果。
- `registry/rc_coherent_residual_development_authority_v8_20260908.json`，当前P门。

本轮没有提交/修改/监控任务，没有读取正式392的P/RoMa结果或操作V/action，
没有访问受保护D1-MI产物。这里只读已有结果并校正目标和归因。
