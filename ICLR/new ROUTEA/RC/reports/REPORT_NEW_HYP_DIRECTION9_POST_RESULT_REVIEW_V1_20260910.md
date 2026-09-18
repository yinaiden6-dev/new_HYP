# new HYP：DIR9 独立结果复核

原 RAW C128、已打开 matched EVAL32（11组）；RAW 25/32，前序 ORIGINAL7 28/32。

| 模型 | 参数数 | TRAIN /32 | EVAL /32 | 对 RAW 救/损 | 对 ORIGINAL7 救/损 |
| --- | ---: | ---: | ---: | --- | --- |
| FIXED_PROJECTION8 | 8 | 28 | 26 | 2/1 | 0/2 |
| DIRECTION9 | 9 | 28 | 27 | 2/0 | 0/1 |

DIRECTION9 保留原28并新增至少1个：False。
共享 theta=-0.318628454819 rad；对应轴角度（仅展示，模180°）=26.743934306°。
实际系数轴 (J,T)=(0.893026588677, 0.450003902111)；推理仍保留原 sum/difference 运算顺序。
theta 梯度调用 2000 次；首步 0x0.0p+0；非零 1999 次。
相对 ORIGINAL7 新增 []；丢失 ['OUTCOME-0220']；group 平均差 -0.022727273。

- REAL：FIXED_PROJECTION8 26/32；DIRECTION9 27/32。
- CBIND：FIXED_PROJECTION8 2/32；DIRECTION9 5/32。
- EXTRA_BIND：FIXED_PROJECTION8 21/32；DIRECTION9 25/32。
- CBIND 对新增救回的保留：[]；丢失：[]。
- EXTRA_BIND 对新增救回的保留：[]；丢失：[]。

独立复算384份冻结特征矩阵、48,768个FP64 logits及384个完整动作；固定头参数、损失、预测和动作精确回放前序PROJECTION8。
本复核没有重训、LP求解、方向搜索或encoder执行。首步梯度为0符合全零投影系数初始化；不能把这一点误报为梯度断开。
结果属于内部开发；共享方向增加一个参数，不自动证明新信息、外部泛化、ownership或部署资格。
复核SHA：`bdcaf9bd916e035abd41a05ce3dd9f686617c893eacfd3b20a05906302eada80`。
