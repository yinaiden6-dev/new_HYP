# REBUT QR / QR-vec / QRR：共享证据缓存审计

日期：2026-09-27。范围：用户新版计划的步骤2，核对现有缓存是否支持同输入信息预算的三臂比较。本轮不运行编码器/RoMa、不修改已有任务、不按正确与否选择query。

## 结论

**全H593已有完整终端u/v及单图RoMa池化描述子；完整候选的可复核warp目前只覆盖固定执行索引0–70的71张。** 因而无需新增GPU即可构造一个71张、每张完整自然C128、三臂共用的池化F层。不能把全593终端权重齐备写成全593匹配坐标齐备，也不能把历史“恢复128张”的提交记录写成128张实际采集完成。

这里的F是单图RoMa描述子在ColNomic token对应图像单元上的池化值，加上实际保存的RoMa坐标；不是未压缩原生高分辨率激活。三臂都使用同一种池化表示，能研究该表示上的早/晚比较；负结果不能排除更高分辨率F的信息。

## 当前实际覆盖

| 字段/来源 | 磁盘覆盖 | 结论 |
|---|---:|---|
| `results/rc_h593_quality_operator_v1/queryNNN/intermediates.pt`及validation | 593/593 | 完整C128的终端u/v、原生评分、自由/加权MaxSim命中及贡献可直接复用 |
| `results/rc_h593_feature_fusion_cache_v1/ready.json` | 593 query，5324唯一图片/几何条目，21片，复用129 pilot条目 | 双侧单图粗/细池化描述子齐备；ready状态为`FUSION_ALL593_FEATURE_CACHE_PASS` |
| `results/rc_h593_roma_coordinate_precision_v2/query000`–`query070` | 71/71有`ROMA_COORDINATE_QUERY_PASS` | 各128候选，NATIVE+3种坐标精度对照；本实验只读取NATIVE |
| `results/rc_h593_m_inside_v1` | 42张：0–40、70 | 七阶段及内部干预；不是全593的完整激活库 |
| `results/rc_h593_m_visual_origin_v1` | 56张：0–54、70 | 七阶段/视觉干预；不能由16×16摘要恢复完整warp |
| `cache/rc_h593_unified_acquisition_v1` | 仅query070具有`export_validation.json` | 统一采集不额外扩展当前71张warp交集 |
| `cache/rc_h593_shared_native_v1/coarse` | 129个`.pt` | 首自然query及其完整C128的原生50×50×1024粗特征；不是5324图完整原生粗特征 |

以上覆盖来自当前目录、metadata与首个tensor文件检查；没有重新完整读取/哈希所有大型tensor来重复旧验证。新证据构建仍应逐个读取时核对已有binding及query/candidate轴。

补充现场完整性检查：5324/5324个`ready.features`绑定的文件均存在，593/593个query目录payload存在；71份坐标validation/payload均为正确状态、各8个part及128物理候选，568/568个part文件均存在，未发现缺文件或元数据轴长度异常。

71张为原先按执行顺序固定的0–70，历史面板共37个component、原五折held数量21/15/15/5/15，见`plan/RC_H71_FEATURE_FUSION_PILOT_V1_20260923.md`。新任务必须重新按原fold清单连接，不因哪折有收益而选择面板。只用71张训练/评估的新结果不能与593总数比较；593训练所得同折旧头可作为对应71张的封存基线，但要注明训练量不同。

## 真实schema与shape

### 单图描述子和几何

查找入口：

```text
results/rc_h593_feature_fusion_cache_v1/catalog.json
  queries[*]: query_id, execution_ordinal, query_image_key, payload
  images[key]: item, input, pilot_validation
results/rc_h593_feature_fusion_cache_v1/ready.json
  features[key]: {path, sha256}
queries/queryNNN.pt
  query_image_key, candidate_physical_rows, winner, challenger_positions
  pairs[*]: position, physical_row, image_key,
            query_visibility, reference_visibility, native_c4
```

feature payload在全量`images/<key>/payload.pt`或旧pilot对应目录。不能假定全部在全量images目录，须经过`ready.features`解析。

```text
item.grid / geometry.grid_shape
geometry.processor_input_frame / raw_size_hw / oriented_size_hw / exif_orientation
cell_boxes_xyxy: [T,4], oriented normalized [0,1]
valid_patch_mask: [T], bool
components.coarse_11: [T,1024], float32
components.coarse_17: [T,1024], float32
components.fine_lr_1 / fine_hr_1: [T,64], float32
components.fine_lr_2 / fine_hr_2: [T,128], float32
components.fine_lr_4 / fine_hr_4: [T,256], float32
projected_inputs: old 128-D fusion input; not raw descriptor
full_resolution_activations_saved: false
```

首query实际T=720，grid=36×20；首reference T=742，grid=53×14。两侧不能按同数组下标相减。粗层来源分别为RoMa `model.f(lr)`的两个输出；描述子经过FP64面积池化，随后存float32。首版建议固定`coarse_17`，全部三臂同字段，不按表现选层。

源码：`programs/run_rc_h593_feature_fusion_cache_v1.py:160`，`programs/rc_feature_fusion_core_v1.py:15`。

### 双向坐标

```text
results/rc_h593_roma_coordinate_precision_v2/queryNNN/part00.pt ... part07.pt
  pairs[*]: candidate_position, physical_row, reference_tokens_sha256
    coarse_matcher:
      warp_AB / warp_BA: [1,200,200,2], float32
      confidence_AB / confidence_BA: [1,200,200,1], float32 logits
    arms.NATIVE:
      query_visibility: [Tq], float64
      reference_visibility: [Tr], float64
      intermediate.query_coordinates / reference_coordinates:
        grid_hw: [1280,1280]
        cell_boxes_xyxy: [T,4], float64
        valid_patch_mask: [T], bool
        center_xy: [T,2], float32
        cell_mean_xy: [T,2], float64
        cell_cov_xx_yy_xy: [T,3], float64
        cell_out_of_frame_fraction: [T], float64
        visualization_sample_64x64: [64,64,2], float32
```

`center_xy`是真实终端1280warp在各token cell中心的采样，目标坐标系为**另一张图EXIF-oriented RoMa输入，归一化[-1,1]**。`cell_mean_xy`是坐标均值，不能无损代替中心采样或密集场。64×64仅为可视化子采样。完整终端1280×1280warp未保存；完整200×200粗matcher输出保存。

粗confidence必须按该源代码的第一通道sigmoid解释，不能当作已归一化u/v或直接与终端M混用。其采样口径见`programs/rc_roma_visual_origin_v1.py:80`；坐标记录见`programs/rc_roma_coordinate_intermediates_v2.py:7`。

## 有效性mask究竟是什么

现有`valid_patch_mask`只表示token图像几何/非padding有效性；`cell_out_of_frame_fraction`记录warp落到目标图像范围外的比例。二者**不是物体可见性真值、遮挡真值或正确匹配标签**。

可在CPU根据保存的center warp构造映射有效性：源cell有效、坐标有限且在[-1,1]内、坐标所在reference cell有效。低certainty的位置仍保留；不得通过u/v阈值将其丢弃。几何无效位置将对侧采样值置0，同时保留明确mask，不能把零值当作“可见但不支持”。

采样必须使用已存的oriented cell boxes查找位置，或经验证的等价网格/EXIF变换。直接reshape成`grid_shape`后不处理EXIF可能错位。reference侧按BA warp执行完全相同规则。

## 最小真实可执行方案

1. 冻结71张现有warp完整交集，保留每张全部128物理候选，不按label、M名次、历史错误选择。
2. 读取双侧`coarse_17`，以同一个预先固定、无标签的32维投影用于q/r及所有实验臂；保留原始描述子binding，明确新增投影只是控制CPU规模。
3. **先warp采样，再做8×8几何池化。** 在每个原query token中心读取真实AB坐标，根据reference cell boxes采样其描述子和v；随后按query cell中心在共同8×8图像网格的归属均值池化。反方向同理。先把warp平均再采样会改变对应关系，不能偷换为等价过程。
4. 每个query存两份`[128,64,72]`局部证据：自身F32、对侧warp采样F32、本侧支持及对侧支持2维、本侧坐标2维、有效warp坐标均值2维、源几何有效比例1维、映射有效比例1维；另存显式`source_mask`和`mapping_mask`。对侧缺失不删除本侧内容或本侧支持。
5. QR独立池化每个候选；QR-vec独立产生向量；QRR仅在共同query 8×8轴上联合比较anchor与challenger，reference侧只分别汇总。三个臂读取完全相同E；原M0及冻结原决策证据并行保留。
6. 保存原物理轴、queryID、winner/challenger索引、原始终端M0（FP64）、全部源binding、投影seed和坐标/池化规则。训练接口不可读取ID字符串语义；这些字段仅作连接与追踪。

上述是**池化粗描述子＋终端坐标/支持**的明确定义接口；“同层级”指q/r的描述子均来自同一`coarse_17`层，不声称支持和坐标也来自该粗层。若决定全部改用coarse matcher support/warp，必须对三臂共同改为另一套预先冻结证据接口，原终端M0仍单独保留，不能静默混用。

此方案不需要新RoMa/ColNomic前向，也不需要reference–reference匹配。它能检验当前池化F上的标量压缩、向量晚比较、局部早比较，不足以直接得出全H593新成绩。

## 不可由现有字段推导的部分

- 全593缺失的522张RoMa warp不能从u/v、M或MaxSim索引反演。
- 原生高分辨率RoMa描述子不能从token池化值或128-D随机投影无损还原。
- 物体/遮挡真值mask不能由certainty或几何有效mask等同得到。
- 新reference–reference RoMa匹配不在该证据包中；QRR-v1无需引入它。
- 若用ColNomic MaxSim索引替代RoMa warp，可另做新协议，但不再是上传计划要求的相同冻结RoMa对应证据。

因此先做F71是可执行方案；若要主比较覆盖593，需有明确授权和预算补522张匹配坐标，或者将研究问题明确改为支持S层。不能用一个缺关键字段的“F593”替代公平输入控制。

## 已实施的CPU共享证据及工程检查

用户确认后，新增构建器`programs/build_rebut_qr_qrr_evidence_v1.py`，固定源码SHA256为`44624f73c214a74a53fc0ad70bcffbdd6b2a078e67d47fd50a34f4ee6960b9f7`。新产物独立存放在`results/rc_rebut_qr_qrr_v1/`；原缓存及原预测不变。具体字段文档为`results/rc_rebut_qr_qrr_v1/evidence_schema.md`。

首query的完整128候选构建用时13.48秒、单payload约4.91MB；只复用CPU缓存。其余71张构建由父任务提交CPU分区，480秒后在query边界以exit75退出，可跳过已封存且SHA核验通过的query续跑。本地短程已停止，避免与batch作业并发写同query。

独立验证器`programs/verify_rebut_qr_qrr_evidence_v1.py`的SHA256为`8b85ebe1a26438b3444155af6cb8e3f46f095afc6f63d2f038e9086a7bc1cf72`。当前29-query快照、3712候选的输出SHA、候选轴、shape、原M0和native_X逐位核对通过；首query固定候选0/1/127的双向27648个值以brute-force cell定位和逐bin显式均值独立重算，最大误差0。

合成工程检查覆盖：零/近零支持仍保留；reference数组重排由显式几何正确对应；越界/NaN/padding失去mapping但不删除自身内容；不同位置先采样再pool；共同query网格一致。部分工程回执为`evidence_engineering_checks.json`，绑定固定`evidence_engineering_manifest_snapshot_029.json`，不绑定可能继续变化的live manifest。

这只验证共享输入与对齐实现，不是新训练结果。全部71张完成后，必须运行不含`--allow-partial`的独立验证，得到`evidence_validation.json`的`POOLED_F_EVIDENCE_INDEPENDENT_PASS`，再进入训练。
