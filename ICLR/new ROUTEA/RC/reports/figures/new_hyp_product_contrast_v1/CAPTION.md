# Figure: product contrast as a nonlinear joint calibration term

**Suggested caption.** For positive quality mass M and normalized content L, let
x=(M_c−M_w)/(M_c+M_w), y=(L_c−L_w)/(L_c+L_w), and S=M L. The symmetric product
contrast is (S_c−S_w)/(S_c+S_w)=(x+y)/(1+xy). Panels show (a) the additive contrast,
(b) the product contrast, and (c) their difference over x,y∈[−0.98,0.98]. Panels
(a) and (b) share the color range [−2,2]; panel (c) uses [−1,1]. Dashed curves mark
zero. Although the calibration head is linear in its supplied features, including
the product contrast supplies a nonlinear function of the two underlying
contrasts. These panels illustrate an algebraic identity under epsilon=0 and no
denominator floor; they contain no experimental data and do not demonstrate
accuracy gains or causal necessity. Actual model features retain their original
FP64 values, epsilon and division floor.

**中文解释。** a 为两个相对证据的简单相加，b 为原乘积分数对应的非线性
对比，c 为二者之差。图说明删去乘积分数对比可能改变线性头可表达的函数；
不说明哪种表示准确率更高。实际实验继续使用原数据列。

**Artifacts.** SVG and PDF are the publication exports; PNG is a 300 dpi preview.
The figure is 7.5×3.05 inches, uses a 401×401 analytic grid, and embeds its raster
color field with vector text/axes. `manifest.json` records the script, software
versions, numerical ranges and artifact hashes. No gallery, query, model, label,
prediction or scheduler input is read.

**Reproduce from workspace root:**

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 .venv-colpali/bin/python \
  'ICLR/new ROUTEA/RC/programs/plot_new_hyp_product_contrast_v1.py'
```

The derivation with nonzero epsilon, applicable training-data conditions and
experiment interpretation is in
`reports/NEW_HYP_PRODUCT_CONTRAST_MATHEMATICAL_REVIEW_V1_20260909.md`.
