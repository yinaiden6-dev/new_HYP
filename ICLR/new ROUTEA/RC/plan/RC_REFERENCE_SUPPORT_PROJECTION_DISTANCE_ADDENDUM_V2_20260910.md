# 执行前更正：投影点到原点的欧氏距离

用户再次明确：“是点在y=x的投影点距离原点的距离”。本文件更正此前
RC_REFERENCE_SUPPORT_PROJECTION_SUM8_ADDENDUM_V1_20260910.md中由assistant
擅自改为有符号右上坐标的解释。V1保留为未执行的历史记录，本V2取代
其第四arm定义；不增加第五arm。全量缓存、新头训练和EVAL输出均尚未
执行，原三个arm及MAXMIN8主比较不变。

## 数学定义

对图中点(x,y)，到y=x的正交投影为P=((x+y)/2,(x+y)/2)，因此投影点
到原点的欧氏距离为d=abs(x+y)/sqrt(2)。负方向的投影也有正距离，这是
用户提出的定义，不能用有符号和替换。

仅对已封存TRAIN pilot的512个点核对：目标按精确abs(x+y)排名为
6/1/1/1，与本批数据的有符号和排名相同。仍为相同3/4目标第一，不
优于单独横轴的3/4；这不否定机制假设，但尚无新增识别结果。四query
来自3个TRAIN身份/组。没有以该图拟合投影角度、尺度或阈值。

## 第四arm：PROJECTION8

模型使用主计划已固定的FP64 J和T=float(exact L)。唯一运算次序为：

    sum_g = binary64_add(J_g, T_g)
    distance_g = binary64_div(abs(sum_g), math.sqrt(2.0))
    PROJECTION8第七列 = symmetric(distance_c, distance_w)。

保存sum_g和distance_g的hex及附加列hash。这里J是原已冻结的FP64
clamped池化值，T是精确L的就近舍入表示；不能声称与绘图的精确横轴
无舍入差。按实际模型数值另核对TRAIN示例。sqrt(2.0)为CPython math
返回的binary64值，保留除法步骤；原symmetric含epsilon，不能擅自把
共同除数约掉。无角度、权重、温度、阈值或额外归一化搜索。

PROJECTION8与FIXED8、MAXMIN8均为8参数；原六特征和原检索监督/优化
顺序保持。先对原candidate计算distance，再按旧donor定义构造C_BIND；
EXTRA_BIND只转移第七列。所有127challenger、原HOLD/SWITCH与tie-break
不变，不训练控制专属head。原signed J/T继续完整保存。

## 比较、冻结与口径

原主比较MAXMIN8 vs ORIGINAL7保留。PROJECTION8 vs ORIGINAL7/FIXED8/
MAXMIN8为用户提出的TRAIN启发次比较。分别报告MAXMIN8及PROJECTION8
是否保留原28并新增至少1个，不用EVAL选择一个arm冒充唯一预设方法。

四头共同在原TRAIN32+PAIR64拟合，全部FULL64预测封存后才读取EVAL
角色及旧结果；新进程独立重训四头。ORIGINAL7/FIXED8参数、预测和
动作精确复现门不变。任务只有原检索身份/配对监督，无空间标注、
identity专属分类参数、SAM、D1-MI、GroZi、旧V9、正式392或ownership。

沿用主计划RC_REFERENCE_SUPPORT_MAXMIN_READOUT_V1_20260910.md。
数量更新为四头，读出仍8CPU/4G/15分钟。缓存不变，仍8,320项；本
更正不新增LP、内容编码或RoMa调用。读出authority绑定主计划及本V2。
V1及其旧源码预检保持append-only，不授权按旧SUM8自然执行。
