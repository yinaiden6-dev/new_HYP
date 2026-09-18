# new HYP 四头读出：独立动作复核

独立重训验证已完成，本复核重新读取封存 logits，独立执行全部 C128 动作及评价算术。
范围：原 RAW C128、已打开 matched EVAL32（11组）；原 RAW 25/32、ORIGINAL7 28/32。

| 模型 | TRAIN /32 | EVAL /32 | 对 RAW 救/损 | 对 ORIGINAL7 救/损 | 保留原28并新增 |
| --- | ---: | ---: | --- | --- | --- |
| ORIGINAL7 | 28 | 28 | 3/0 | 0/0 | 对照 |
| FIXED8 | 28 | 27 | 2/0 | 0/1 | 对照 |
| MAXMIN8 | 29 | 27 | 3/1 | 0/1 | False |
| PROJECTION8 | 28 | 26 | 2/1 | 0/2 | False |

主比较 MAXMIN8 达成用户目标：False；用户提出的次比较 PROJECTION8：False。
两者分别报告，不用次比较替换主比较。

复核覆盖97,536个 challenger logits、768个动作；按 physical row 确定平局，logit>0才 SWITCH。
ORIGINAL7/FIXED8 的参数、全部预测和动作与前序完全一致；RAW 原正确损失名单、救回保留和 group/MRR 均已复算。

- MAXMIN8：新增 []；丢失 ['OUTCOME-0618']；group 平均准确率差 -0.04545455。
  CBIND 下新增救回保留 []；丢失 []。
  EXTRA_BIND 下新增救回保留 []；丢失 []。
- PROJECTION8：新增 []；丢失 ['OUTCOME-0220', 'OUTCOME-0618']；group 平均准确率差 -0.06818182。
  CBIND 下新增救回保留 []；丢失 []。
  EXTRA_BIND 下新增救回保留 []；丢失 []。

这是内部开发结果，不是未触碰外部确认、严格空间 ownership 或自动部署。
本复核没有重训、重新求解 LP 或读取容量 oracle；参数/特征的独立重训由前置验证承担。
复核产物 SHA：`d4c05cf450ebc83f9eaa54a4195b61e578023eb45cf5cd6707f05e83d9624eef`。
