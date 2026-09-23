# 特征融合试运行中间数据

该目录是 engineering pilot，无自然数据训练、无身份正确率。样本选择见manifest及authority；不是全593图的训练结果。

- `manifest.json`：原始raw/roma封存来源，固定ordinal0 query与完整128候选，129图像/几何条目；图像SHA、原token SHA、grid、显式frame。
- `images/<key>/payload.pt`：原ColNomic tokens、几何和valid mask；八块对齐粗细特征；四来源的固定128维投影；独立CPU直接区域均值误差。池化为FP64计算、FP32保存，投影为FP64。未保存未池化的全分辨率隐藏激活。
- `images/<key>/validation.json`：来源与payload SHA、维度、池化核验。每图提交完成后可恢复。
- `hook_qualification.json`：首个自然候选中query/reference所有八块池化值独立编码与完整RoMa前向逐位一致性；原生overlap逐位一致性。只证明池化接口一致，不声称全稠密激活均已落盘。
- `payload.pt`：原候选轴/RAW分数/winner；每候选两侧u/v；FREE及ROMA_WEIGHTED完整四统计、自由/加权命中索引和逐token贡献；四来源×两评分的全部127×6决策输入；零初始化参数；八臂无标签梯度检查。
- 试运行中门控严格为0，融合后未归一化tokens与原tokens数值完全相同，八臂之间的重复trace通过`trace_reference`显式复用。此时没有训练后的头或决策logits，不能把127×6输入当作127个模型预测。正式训练另存投影/门控、融合tokens、参数、完整127logits与HOLD=0。
- `runtime.json`：GPU型号、峰值显存、实际耗时、单图数量。`validation.json`：新进程回放所有图片投影、全部候选四统计/匹配/贡献和旧六特征公式。

已存在的图像payload未封存则停止检查，不能当作完成。最终payload已原子写入时可从独立核验恢复；不可变文件以SHA绑定。
