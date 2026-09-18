# new HYP：TRAIN0101对任意原点线性投影的精确限制

独立复核root给定的两个wrong点，结论成立。仅针对已打开TRAIN
DIFFICULT-0101的这组二维图点，不读取EVAL，不生成模型参数。

图中x为原精确归一化支持margin，y为精确可达下界L。直接从原有理数
端点解2×2系统，得到严格等式

    target = −a·wrong120 + b·wrong87
    a≈0.1543901427986508，b≈0.6291994148855622
    a>0，b>0，a+b≈0.783589557684213<1。

两个wrong向量线性无关。对任何非零线性变换A和任意范数，三角不等式
及齐次性给出

    ||A target|| ≤ a||A wrong120||+b||A wrong87||
                 ≤ (a+b) max(||A wrong120||,||A wrong87||)
                 < max(||A wrong120||,||A wrong87||)。

因此，单靠“原点线性变换后的距离最大”无法让该目标严格排第一；任意
原点投影方向、45度旋转、线性轴缩放/剪切后取范数均包含在此范围内。
零变换只能产生平局。该结论来自精确证书，不是离散角度搜索没有找到。

另用实际生产FP64 J和T=float(L)作为精确端点重新求解，仍有正系数且
系数和≈0.7835895576842138<1；因此不是绘图横轴与生产J的微小舍入差
造成的反例。此处证明的是这些端点上的数学几何关系，不声明任意
浮点程序的病态溢出行为。

限制：不排除平移、非线性变换、其它query或保留原六特征的完整共享
校准head。尤其不能用这一个TRAIN例子否决群体增益目标；共享可学习
方向仍可能改善其它样本或与原特征联合发挥作用，但不能承诺单独
投影距离会解决这个例子。

本项为NON_DEPLOYABLE_LABEL_AWARE_TRAIN_DIAGNOSTIC。a/b不能用于当前
或后续模型的初始化、选角、训练、阈值或评分。未调用新LP或训练。

证书：results/rc_train_origin_linear_projection_capacity_review_v1/certificate.json。
SHA：f3ab3c12c12bba687dcaf47f9576c256db66bbfd5c1b6710976bf843be57a6af。
原pilot result/prejoin/validation SHA均在证书中绑定。
