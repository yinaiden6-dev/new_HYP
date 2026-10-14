# 跨原图同内容布局匹配 pilot

全130身份按预注册质量规则扫描，4对进入实际视觉核验；3对保留，1对因OCR框包含两个词而排除。保留3身份、6张不同原照片、36个编码材料（2 train、1 val、0 test）。
材料沿用固定OCR、均匀纸面、±0.35词宽和像素守恒条件。两原图的物理方向由共同GT的已接受单应矩阵映射；package_030 的B相机方向反转，左右标签据此交换。跨原图物理位移大小差异如实记录，未按模型响应选择来源或放宽规则。

12个预定义跨原图比较中，正布局差额为5/12。平均 edit-minus-sham 布局差额 -8.4987742e-05，平均 layout-minus-editor 差额 -0.00010255924。差额正负混合；这批材料不能支持稳定的条件布局匹配结论。
这些数是3身份内的重复描述性比较，不能当作12个独立样本、自然身份检索精度或test性能；未训练Probe。

|身份|比较数|正差额数|平均布局差额|平均edit−sham|平均layout−editor|
|---|---|---|---|---|---|
|package_010_crossview|4|2|-0.00013100668|-0.00013100668|-0.0001762566|
|package_030_crossview|4|2|0.0001753616|0.0001753616|0.00035548949|
|package_106_crossview|4|1|-0.00029931815|-0.00029931815|-0.00048691062|

完整36×36 forward mean MaxSim矩阵保留所有原始、±、sham和editor-only跨来源组合。条件比较只使用预注册的不同原图A/B对应方向与相反方向。self/同原图矩阵格仅供诊断，不进入上述差额。

edit−sham 分解为固定sham赢家的feature变化和argmax切换，二者逐token守恒。词框/编辑带汇总仅使用token中心位置代理，不能解读成精确感受野或语义定位。
最大布局差额分解误差 4.89e-10，最大逐token误差 5.96e-08。独立重放48个像素材料（包含被排除身份），逐像素、字形RGB、编辑带外和±/sham色直方图断言均通过。编码complete清单SHA及每个tensor/image SHA已核对。

[完整矩阵](complete_similarity_matrix.csv) · [12个预定义关系](predefined_layout_contrasts.csv) · [feature/argmax分解](feature_argmax_decomposition.csv) · [身份汇总](identity_pilot_summary.csv) · [验收](validation.json) · [像素重放](pixel_replay_validation.json)

[预注册方案](../protocol.json) · [全130身份盘点](../identity_inventory.json) · [实际视觉核验](../manual_reviews.json) · [编码清单](../encoding_manifest.json)
