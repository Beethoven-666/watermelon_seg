# 西瓜 YOLO 实例分割数据集

本目录是 Ultralytics YOLO 实例分割数据集，用于训练识别西瓜实例轮廓的分割模型。当前主数据集由两个 Roboflow COCO segmentation 导出数据集和本地 `raw/my_labelme` 标注融合生成。

## 目录结构

- `images/train`、`images/val`、`images/test`：融合后的训练、验证、测试图片。
- `labels/train`、`labels/val`、`labels/test`：与图片一一对应的 YOLO 分割标签。
- `raw`：原始采集数据和本地 LabelMe 标注来源，不应批量删除。
- `exports/fused_watermelon_yolo_seg`：融合拆分清单和统计摘要。
- `exports/my_labelme_yolo_seg`：本地 LabelMe 标注单独转换结果。
- `external`：当前采用的 Roboflow COCO segmentation 原始导出和来源记录。

## 当前主数据集

类别只有一个：

```text
0: watermelon
```

融合来源：

- Roboflow `Watermelon Detection` v5 `raw_annotations`，Instance Segmentation，CC BY 4.0。
- Roboflow `Watermelon Detector` v1，Instance Segmentation，CC BY 4.0。
- 本地 `raw/my_labelme` LabelMe 多边形标注。

当前按随机种子 `20260707` 统一重拆分为：

- 训练集：845 张图片 / 845 个标签 / 1826 个实例。
- 验证集：105 张图片 / 105 个标签 / 226 个实例。
- 测试集：107 张图片 / 107 个标签 / 196 个实例。

总计：1057 张图片 / 1057 个标签 / 2248 个实例。

详细拆分来源见 `exports/fused_watermelon_yolo_seg/split_manifest.tsv`，统计摘要见 `exports/fused_watermelon_yolo_seg/summary.json`。

## 标签格式

每张图片都应在对应的 `labels` 目录中有同名 `.txt` 标签文件。YOLO 分割标签每一行格式为：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

所有坐标均归一化到 `0.0-1.0`。本项目中 `class_id` 固定为 `0`，表示 `watermelon`。

## 训练命令

```powershell
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=8 device=0
```

如果只能使用 CPU：

```powershell
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=4 device=cpu workers=0
```

## 快速验证

```powershell
yolo segment val model=path/to/best.pt data=D:/MelonDataset/watermelon_seg/data.yaml
```

## 最新训练结果

融合数据集的 GPU 微调结果、参数和评估指标见：

```text
runs/segment/fused_finetune_640_gpu_from_mylabelme/TRAINING_REPORT.md
```

## 注意事项

- 当前根目录 `images` 和 `labels` 已替换为融合后的实例分割数据集。
- 不要把检测框标签直接混入本数据集；本项目主数据集只接受实例分割多边形标签。
- 修改数据来源、拆分、类别编号或标签格式时，请同步更新 `README.md`、`data.yaml`、`classes.txt`、`AGENTS.md` 和 `external/SOURCES.md`。
