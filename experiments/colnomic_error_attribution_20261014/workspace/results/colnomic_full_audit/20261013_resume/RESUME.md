# ColNomic 全量分析恢复记录

本目录从 2026-10-13 断流任务恢复。历史结果、dataset 和 cache 只读；新输出存独立目录。原始任务规格保存在 `checkpoint_2108/TASK_SPEC.md`。

## 已核查状态

- 当前没有 ColNomic 布局编码进程。服务器另有 content-box 四 GPU 作业，不终止、不重复启动编码。
- `20261013_v2/stage4_extended_photos_v3` 的编码已完成 228/228，38 个来源身份，train/val/test = 20/8/10。
- 新增独立恢复校验逐份检查图像 SHA、Token 哈希、manifest 字段、形状、有限值、processor、原图缓存行。228 份通过，38 张 base 均与原缓存逐位一致。
- 完成标记不是唯一依据。详细证据在 `checkpoint_2108/checkpoint_validation.json` 与 `encoding_records.csv`。
- 从最后有效检查点接续了 `analyze_layout_photos.py`，没有重启编码。新增分析输出为 v2 的此前不存在的 `stage4_extended_photos_v3/analysis`；运行日志在本目录。
- 历史完整图库标量评分有 1117×5405，B03/F03 的逐 Token 队列分别为 247/145 错误查询。视觉审核不能据此认定全量完成。

## 最新用户修正

1. 阈值分类准确率不足以证明不可分，必须报告 AUC、同 Query 配对排序、差额、包装宏平均与验证阈值。
2. 池化矩不能代替完整 Token 集合信息检验，需独立输入消融及集合关系头。
3. OCR 标签是纵向排序位置，不是真实阅读行号；保留 y-only 基线。
4. 划分通过必须来源于实际交集与断言，区分 feature-only 与坐标/ROI 对照头。
5. 重点图仅用于查验；正式样本从 dataset 原图和 cache 原 embedding 定位，不能用 wrong 浏览图编码或筛选 Probe。
6. 区分冻结分数重放与重新编码缓存复现。近期照片的 PIL 后端漂移已定位，历史图库链路仍待核实；不得未经证据归因 GPU 或 CPU 卸载。

## 接续中的独立任务

- 逐样本和头权重独立重放 Probe 指标；扩大布局解释与编辑痕迹控制。
- 定位历史图库缓存生成链路；不能用新旧混用排名声称完整新运行时性能。
- 补齐 1117 查询逐 Token、完整候选档案、区域守恒与可浏览匹配证据。
- 拆分 gamma/lambda 与空间破坏对照、Query 权重、融合和完整图库损益。
- 持续补齐全量视觉审核、成功对照、统计、报告、复现及归档验收。未审核或未执行内容保留为未完成。

## 恢复原则

先读最新 state、checkpoint validation 和进程；已有完成标记还须核验文件。输出若已存在，先校验 hash 与参数后复用；不通过时新增隔离目录，不覆盖历史结果。长日志只看必要末尾或错误片段，完整输出写日志文件。

## 2026-10-14 最终接续与验收

原恢复段落作为历史保留。当前分析已通过 delivery/acceptance.json：1117主档案、1117实际overview、三个独立场景维度、全部Token分析、35配置39095重复评测、完整原图库排名以及拆分机制对照均已验。语义ROI仅4张，细节验证限于明确列出的13范围，未知与反证保留；不表示每张原因已确定。

新增28参考+12base编码全部完成，12base逐位等缓存；60兼容编辑臂、新旧290矩阵、特征变化和按身份统计均独立验收。源码worker变化和ordinal/physical区别保留。编码未重复启动，原标注8765程序已关闭。

一键入口 REPRODUCE.sh 使用最终dispatcher，旧10模块全量回执保持原样；新入口core8×5405快检及完整新增分支真实通过。完整参数与证据映射见FINAL_PROTOCOL.json；缺失/未知见MISSING_AND_LIMITATIONS.json。此时仅归档、真实解包和外部最终验收待做；封存后不改payload，最终完整状态以delivery_support外部receipt为准。
