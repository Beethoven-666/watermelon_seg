# 外部西瓜数据来源

整理日期：2026-07-07。

当前根目录主数据集由两个 Roboflow 实例分割数据集与本地 `raw/my_labelme` 标注融合生成。旧的 FruitSeg30 和 Lightly 外部数据已从 `external` 中移除，旧根目录 `images`、`labels` 也已替换为新的融合数据集。

## 1. Roboflow: Watermelon Detection

- 来源：https://universe.roboflow.com/team-128-sd-2026/watermelon-detection-1gb9h
- 使用版本：https://universe.roboflow.com/team-128-sd-2026/watermelon-detection-1gb9h/dataset/5
- Roboflow 项目名：`Watermelon Detection`
- 版本：v5 `raw_annotations`
- 任务类型：Instance Segmentation
- 类别：`watermelon`
- 许可证：CC BY 4.0
- 本地原始导出目录：`external/roboflow_team128_watermelon_detection_v5_coco_segmentation`
- 原始导出格式：COCO Segmentation
- 原始导出拆分：train 365 / valid 104 / test 52，共 521 张图片。
- 标注统计：1148 个 polygon 实例，均已映射为 `0: watermelon`。

## 2. Roboflow: Watermelon Detector

- 来源：https://universe.roboflow.com/drago-x1kks/watermelon-detector-te4ra
- 使用版本：https://universe.roboflow.com/drago-x1kks/watermelon-detector-te4ra/dataset/1
- Roboflow 项目名：`Watermelon Detector`
- 版本：v1 `2024-01-04 10:51pm`
- 任务类型：Instance Segmentation
- 类别：`Watermelon`
- 许可证：CC BY 4.0
- 本地原始导出目录：`external/roboflow_drago_watermelon_detector_v1_coco_segmentation`
- 原始导出格式：COCO Segmentation
- 原始导出拆分：train 421 / valid 45 / test 23，共 489 张图片。
- 原始预处理：Auto-Orient，Stretch to 640x640。
- 原始增强：每个训练样本 3 个输出，含 90 度顺/逆时针旋转。
- 标注统计：591 个 polygon 实例，均已映射为 `0: watermelon`。

## 3. 本地 LabelMe 标注

- 来源目录：`raw/my_labelme`
- 对应图片目录：`raw`
- 原始格式：LabelMe polygon JSON
- 类别：`watermelon`
- 标注统计：47 张图片 / 511 个 polygon 实例。
- 单独转换结果：`exports/my_labelme_yolo_seg`

## 4. 本地 test_images LabelMe 标注

- 来源目录：`test_images/label`
- 对应图片目录：`test_images`
- 原始格式：LabelMe polygon JSON
- 类别：`watermelon`
- 标注统计：10 张图片 / 116 个 polygon 实例。
- 单独转换结果：`exports/test_images_yolo_seg`
- 并入主数据集日期：2026-07-07
- 并入命名：`local_test_images_20260707_0001.jpg` 到 `local_test_images_20260707_0010.jpg`
- 并入拆分：train 8 张 / val 1 张 / test 1 张，种子 `20260707`

## 融合结果

- 输出位置：根目录 `images/train|val|test` 和 `labels/train|val|test`
- 标签格式：YOLO 实例分割多边形
- 类别：`0: watermelon`
- 随机种子：`20260707`
- 拆分比例：train 80% / val 10% / test 10%
- 人工可视化检查后已从根目录主数据集中移除 64 张问题图片及对应标签。
- 训练集：802 张图片 / 1798 个实例
- 验证集：100 张图片 / 257 个实例
- 测试集：101 张图片 / 254 个实例
- 总计：1003 张图片 / 2309 个实例
- 2026-07-07 标签复查：删除 2 个极小异常多边形和 5 个重复实例。
- 2026-07-07 新增本地 `test_images` LabelMe 标注：10 张图片 / 116 个实例，按 8/1/1 并入 train/val/test。
- 拆分清单：`exports/fused_watermelon_yolo_seg/split_manifest.tsv`
- 统计摘要：`exports/fused_watermelon_yolo_seg/summary.json`

## 已移除的旧外部数据

- `external/FruitSeg30_Mendeley_v3.zip`
- `external/FruitSeg30_Mendeley_v3`
- `external/FruitSeg30_watermelon`
- `external/FruitSeg30_watermelon_yolo_seg`
- `external/lightly_dataset_fruits_detection`
- `external/lightly_watermelon_detection_yolo`
