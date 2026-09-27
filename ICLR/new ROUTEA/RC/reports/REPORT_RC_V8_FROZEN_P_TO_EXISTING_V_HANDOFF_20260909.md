# 从已有RoMa收益回到P/V职责：可交付的冻结P几何

2026-09-09。原系统的识别收益是研究基础，目标是在同一完整系统中验证HYP
并改善识别；P独立分类只是此前选定的一条资格路线，不能代替整个系统目标。
用户现要求校正职责。本轮未启动V自然评分，没有重训或修改旧系统。

必须区别同名C：旧retrieval C_PAIRED仍为w_q/w_r加权full-reference MaxSim；
旧B_QUERY为w_q和reference权重1的MaxSim。当前V的FROZEN_PAIRED_REGION则
确实只读P配对reference atoms。后来的硬配对P又是第三个对象。
因此不能用D1 B/C平局推出“旧C是硬绑定，必须换B”；保留w_r的路径仍接近
旧C定义。现有RAW V没有w_r输入，增加该权重需单独明确协议。

## 现有V已经具备所需内容匹配分支

`src/rc_aslo_xf/frozen_p_raw_colnomic_three_arm_v1.py`已包含
`FROZEN_QUERY_REGION_FULL_REFERENCE_RAW_COLNOMIC`：在冻结candidate query支持
上对完整reference做cosine MaxSim。与paired arm共享pair query union分母，
不读取P score或head状态，不选择或改变P区域。

这允许直接检验“固定P支持时，full-reference身份读取是否比硬配对更有效”。
不能拿一个更换V8残差head的代理实验冒充这个RAW V接口；此前新写的
`run_rc_v8_assignment_intervention_v1.py`保持未冻结/未提交草稿。V9继续暂停。

## 已交付产物

`results/rc_v8_frozen_p_views_v_bridge_v1/manifest.json`

SHA `1704bd34f5fb21879ed46f8a03e5fa3f8d4539f0c9d46c7a1ccf0bb720e8a1e8`

`results/rc_v8_frozen_p_views_v_bridge_v1/validation.json`

SHA `2a08cd3c86785c2b5cd8bf34e7a4e38cd5ab5c7bc2f1d2ba0be67f23c6e150d7`

全部32×C128×3控制，12288个FrozenPView。REAL及C_BIND各1672H1、2424H0；
P_COORD全4096个H0。全部selected geometry精确匹配原V8 MAP component，
经过既有FrozenPView封存/验证器与JSON-int64重载。没有读取raw tokens或
target roles；没有导出P数值分数、pooling witness或head状态。

接收函数：`programs/export_rc_v8_frozen_p_views_v_bridge_v1.py:load_frozen_view`。
来源authority仍为原V8 authority，不伪造新的P科学资格。

## 使用边界与合同衔接

这些产物只证明ENGINEERING_STRUCTURAL_P_VIEW_COMPATIBILITY。原V8
source_V8_P_GO=false、old_26_P_gate_passed=false保留。该批数据为已打开
TRAIN32，不是旧27/32的EVAL32，也不是untouched测试。

现有V合同只有E0，自然运行须绑定准确P schema、population与hash的新
authority。职责改变应以新的前向开发合同明确接受这批几何用于P/V桥接，
而不是等待P独立分类通过，或把原NO-GO改成GO。桥接结果仍不等于最终系统
与HYP联合通过；完整action、对照及独立评估后续须统一绑定。

可审阅草案：
`plan/RC_P_V_ROLE_RECONCILIATION_AND_DEVELOPMENT_HANDOFF_DRAFT_20260909.md`。
它没有自动开放自然V、正式392、action、ownership或受保护D1-MI。
