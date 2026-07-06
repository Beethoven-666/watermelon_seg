# 西瓜 YOLO 实例分割数据集

本目录整理为 Ultralytics YOLO 实例分割数据集，用于训练识别西瓜实例轮廓的分割模型。

## 目录结构

- `images/train`、`images/val`、`images/test`：训练、验证、测试图片。
- `labels/train`、`labels/val`、`labels/test`：与图片一一对应的 YOLO 分割标签。
- `raw`：原始采集数据和未筛选素材。
- `exports`：标注工具导出的标签或转换后的中间数据。
- `external`：外部下载的数据源、原始压缩包和转换结果。

## 当前主数据集

根目录下的主数据集来自 FruitSeg30 的西瓜分割子集，类别只有一个：

```text
0: watermelon
```

当前拆分如下：

- 训练集：34 张图片 / 34 个标签。
- 验证集：4 张图片 / 4 个标签。
- 测试集：4 张图片 / 4 个标签。

## 标签格式

每张图片都应在对应的 `labels` 目录中有同名 `.txt` 标签文件。YOLO 分割标签每一行的格式为：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

其中坐标已经归一化到 `0.0-1.0`。本项目中 `class_id` 固定为 `0`，表示 `watermelon`。

## 训练命令

```powershell
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=8 device=0
```

## 快速验证

```powershell
yolo segment val model=path/to/best.pt data=D:/MelonDataset/watermelon_seg/data.yaml
```

## 注意事项

`external/lightly_watermelon_detection_yolo` 是目标检测数据集，标签是检测框，不是实例分割多边形。除非明确训练检测模型，否则不要把它直接混入根目录的分割数据集。
