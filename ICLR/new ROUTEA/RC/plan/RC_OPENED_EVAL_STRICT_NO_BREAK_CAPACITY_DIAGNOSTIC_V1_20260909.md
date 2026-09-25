# 已打开EVAL的严格无损增一容量诊断（非模型结果）

## 唯一问题与标签边界

原NATIVE7在已打开matched EVAL32为28正确，剩余四错图为
DIFFICULT-0050、OUTCOME-0213、OUTCOME-0373、OUTCOME-0676。
近期已测读出没有新增修正这些原错误。本诊断问：固定原六维特征，
是否存在一个实数线性读出，严格保留原28正确并再修正指定的一条？

这会显式读取已打开EVAL标签进行事后可行性分析。求得的系数仅为
NON_DEPLOYABLE_LABEL_AWARE_EVAL_DIAGNOSTIC数学证书，不是训练好的
新模型、可部署权重、新准确率或GO证据。不得据它调整当前/后续模型
参数、阈值或挑选训练子集。运行中的5138859及其冻结协议完全不依赖
本诊断，不能将证书输入该训练过程。

## 四个固定系统

按上述四个错误的原execution顺序，各建一个系统，包含原28个正确
query加当前这条，共29个query。其他三条原错误完全不约束，不删除
或重定义原EVAL。每条仍使用原自然RAW C128与全部127 challengers。

设原native6特征加bias为phi，theta为七维自由实数系数：
- target等于RAW winner：每个wrong满足−phi_wrong·theta≥1。
- target为challenger：phi_target·theta≥1；另外126个challenger满足
  (phi_target−phi_wrong)·theta≥1。

每个系统29×127=3683个不等式。固定有限样本上的严格正margin可以
共同缩放为单位margin；精确平局、零边界及依赖physical-row tie rule
的解不在这个严格诊断结论内。不称完整浮点部署函数类的无条件上界。
原28本身应由冻结原theta的精确正margin核验，防止构造错误。

## 精确证书与独立核验

只使用原冻结FP64六特征；通过其原标量独立重建并逐bit比较。以这些
原浮点端点的精确有理数构造差分，不对已舍入差分作证书。可复用
TRAIN约束诊断的求解/证书方法；浮点solver只用于搜索，结论须由
精确有理数可行witness或Farkas证书支持。

四系统全部报告，不选一个有利系统当结果。证书由新进程独立重建
源特征/约束并验证。若精确证书无法闭合，报告未决，不用solver状态
代替证明。不额外扫描容量、加新特征或做其它EVAL标签拟合。

若可行，说明该严格条件在原特征实数线性类内数学上可实现，不能
据此说原训练应当泛化成功。若不可行，只排除这个保留原28+指定一错
的严格正margin系统；不排除改变正确集合、非线性模型、其它特征或
精确平局解。所有结论属于已打开数据的容量诊断。

无新encoder/RoMa forward，无Slurm作业监控，无D1-MI、GroZi、正式392、
SAM、superregion或ownership；原生产模型和结果保持不变。
