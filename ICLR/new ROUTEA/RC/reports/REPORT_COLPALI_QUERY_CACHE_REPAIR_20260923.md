# ColPali query缓存加载兼容修复

此前5160112的22个已运行分片均在旧reference锚点对比失败，未发布任何query tokens。原数据与日志保留。

根因证据：旧checkpoint的`vlm.language_model.model.*`在Transformers 5.6.2中成为UNEXPECTED，而当前`vlm.language_model.*`成为MISSING并被重新初始化。原程序只在输出锚点比较处阻止错误传播，没有在加载阶段提前拒绝。

修复程序：`programs/cache_colpali_h593_queries_v2_compat.py`。使用当前可用环境和明确的checkpoint命名映射，将上述前缀转换到当前模型名称，不修改学习到的权重张量；检查missing/unexpected/mismatched/error列表，并核对checkpoint到模型参数的一对一完整覆盖。不完整加载立即失败并记录`model_loading.json`。旧三reference锚点阈值保持不变，只有PASS后才能编码query。

旧Conda环境的导入探针未在45秒内完成，故没有将它作为已验证环境切换。修复采用显式键映射，不是声称已经验证回退环境。

新输出隔离为`results/rc_colpali_h593_query_tokens_v2_compat/`，输入manifest与v1逐字节相同；查询集合、模型snapshot、BF16计算/FP16存储、处理与数值验收标准不变。v1失败产物没有覆盖。

已提交替代链：

- 5160327：dev_accelerated首片0，15分钟，12张query。
- 5160328：accelerated其余1–49片，%50，每片15分钟，afterok首片。
- 5160329：全部成功后的CPU汇总。

先通过首片权重加载、三reference锚点和query保存验收，再由依赖自动释放其余分片；保留超时续提机制。已取消旧失败/暂停数组5160112及旧汇总5160114。最后核对时首片因Priority排队，后续两项为Dependency。

代码修复与提交已完成，真实GPU数值验收尚未通过，不能提前宣布兼容性恢复。实际运行证据以后续`model_loading.json`、`compatibility.json`和分片`validation.json`为准。

## GitHub增量归档时的状态更新（2026-09-23 19:08 UTC）

5160327_0已FAILED/1:0，运行2分06秒；5160328和5160329因DependencyNeverSatisfied停止推进，query输出仍为0/593。

此次与v1不同：Transformers加载报告的missing、unexpected、mismatched、error四项均为空，语言模型键名修复已经生效。失败发生在新增的参数集合核对：程序只在手工期望集合中处理语言模型改名，没有同步处理库自动执行的`vlm.vision_tower.vision_model.*`到`vlm.vision_tower.*`改名，因此437个视觉参数被误判为未映射/未使用。模型参数总数与checkpoint均为605。

这确认了当前检查器本身的遗漏，但尚未执行reference锚点前向，不能据此宣布旧gallery兼容性通过。该失败JSON本次一并归档；本次上传没有通过删除检查或放宽阈值绕过失败，也未宣称恢复成功。
