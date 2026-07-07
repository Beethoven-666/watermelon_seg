# 融合西瓜 YOLO 分割数据集

该目录记录根目录主数据集的融合过程输出，不重复保存图片和标签。

- 主数据集图片：`images/train`、`images/val`、`images/test`
- 主数据集标签：`labels/train`、`labels/val`、`labels/test`
- 类别：`0: watermelon`
- 随机种子：`20260707`
- 拆分比例：train 80% / val 10% / test 10%

## 来源

- Roboflow `Watermelon Detection` v5 `raw_annotations`：521 张图片 / 1148 个实例。
- Roboflow `Watermelon Detector` v1：489 张图片 / 590 个有效实例。
- 本地 `raw/my_labelme`：47 张图片 / 511 个实例。

## 融合结果

- train：845 张图片 / 1826 个实例
- val：105 张图片 / 226 个实例
- test：107 张图片 / 196 个实例
- 总计：1057 张图片 / 2248 个实例

## 文件

- `summary.json`：融合统计摘要。
- `split_manifest.tsv`：每张图片的来源、目标拆分、标签文件和实例数。
