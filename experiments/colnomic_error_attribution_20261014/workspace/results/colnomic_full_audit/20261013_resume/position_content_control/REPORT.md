# 文字内容与OCR纵向序号的关联控制

头只输入训练集词表的二值词存在向量，不输入坐标/索引/ID/OCR行数或Token向量。OCR多边形和主体框只用来关联同一已核验标签样本，属于额外监督。参数仅按验证身份MAE选取，全部主头及三个打乱对照保存并独立闭式重拟合。

[
  {
    "control": "OCR_content_only",
    "split": "train",
    "samples": 32597,
    "packages": 91,
    "mae": 0.24334743205319867,
    "package_macro_mae": 0.2513738249342042,
    "r2": 0.169975393759893,
    "dim0_four_bin_accuracy": 0.2590729208209344,
    "dim0_package_macro_four_bin_accuracy": 0.24408283097650146
  },
  {
    "control": "OCR_content_only",
    "split": "val",
    "samples": 7715,
    "packages": 21,
    "mae": 0.26712311482752715,
    "package_macro_mae": 0.2745484033644441,
    "r2": 0.03416560489295051,
    "dim0_four_bin_accuracy": 0.23123784834737524,
    "dim0_package_macro_four_bin_accuracy": 0.21822214816860974
  },
  {
    "control": "OCR_content_only",
    "split": "test",
    "samples": 6100,
    "packages": 18,
    "mae": 0.27744568006597436,
    "package_macro_mae": 0.2877630357669277,
    "r2": 0.034065128103134334,
    "dim0_four_bin_accuracy": 0.2124590163934426,
    "dim0_package_macro_four_bin_accuracy": 0.20110427325091657
  },
  {
    "control": "label_shuffle",
    "seed": 17,
    "split": "test",
    "samples": 6100,
    "packages": 18,
    "mae": 0.28391233611756933,
    "package_macro_mae": 0.29683898343996606,
    "r2": 0.002568339179243928,
    "dim0_four_bin_accuracy": 0.18918032786885247,
    "dim0_package_macro_four_bin_accuracy": 0.17366055260331822
  },
  {
    "control": "label_shuffle",
    "seed": 31,
    "split": "test",
    "samples": 6100,
    "packages": 18,
    "mae": 0.2847489630321157,
    "package_macro_mae": 0.29664356066515596,
    "r2": -0.002780895044127707,
    "dim0_four_bin_accuracy": 0.20442622950819672,
    "dim0_package_macro_four_bin_accuracy": 0.1878915595645715
  },
  {
    "control": "label_shuffle",
    "seed": 53,
    "split": "test",
    "samples": 6100,
    "packages": 18,
    "mae": 0.2848433868865662,
    "package_macro_mae": 0.2961679146637014,
    "r2": -0.0016579100570643934,
    "dim0_four_bin_accuracy": 0.1977049180327869,
    "dim0_package_macro_four_bin_accuracy": 0.18351921846096073
  }
]

这是相关性检查，不能单凭它断言feature-only头使用了文字捷径，也不证明自然同文字包装布局可分。
