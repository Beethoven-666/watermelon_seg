# AGENTS.md

## 项目概览

这是 `D:/MelonDataset/watermelon_seg` 下的西瓜数据集工作区。根目录主数据集用于 Ultralytics YOLO 实例分割，目标类别只有 `0: watermelon`。

当前主数据集来自 FruitSeg30 的西瓜分割子集，已经转换为 YOLO 分割多边形格式，并拆分为：

- 训练集：34 张图片 / 34 个标签。
- 验证集：4 张图片 / 4 个标签。
- 测试集：4 张图片 / 4 个标签。

## 关键文件

- `data.yaml`：主数据集配置，`path` 指向 `D:/MelonDataset/watermelon_seg`。
- `classes.txt`：类别清单，目前只有 `watermelon`。
- `README.md`：面向使用者的主数据集说明。
- `external/SOURCES.md`：外部数据来源、许可证、转换结果和未采用候选来源。
- `external/lightly_dataset_fruits_detection/README.md`：Lightly 水果检测数据集原始说明的中文版本。

## 目录约定

- `images/train`、`images/val`、`images/test`：主分割数据集图片。
- `labels/train`、`labels/val`、`labels/test`：主分割数据集标签。
- `raw`：原始采集数据和未筛选素材，不应在没有确认用途时批量删除或覆盖。
- `exports`：标注工具导出结果或格式转换中间产物。
- `external`：外部下载数据、原始压缩包、完整克隆和转换后的候选数据。

## 标签格式

主数据集使用 YOLO 实例分割格式，每张图片应有同名 `.txt` 标签文件。标签行格式为：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

所有坐标均归一化到 `0.0-1.0`。当前项目中 `class_id` 固定为 `0`，含义为 `watermelon`。

## 外部数据边界

- `external/FruitSeg30_watermelon_yolo_seg` 是 FruitSeg30 西瓜掩膜转换后的 YOLO 分割数据，可作为根目录主数据集的来源。
- `external/lightly_watermelon_detection_yolo` 是从 Lightly 水果检测数据集中筛出的西瓜检测数据，标签是检测框，不是分割多边形。
- 不要把检测框标签直接混入根目录的实例分割数据集；只有在明确训练检测模型时才使用 Lightly 筛选子集。
- 外部数据的来源、许可证和处理记录应同步维护在 `external/SOURCES.md`。

## 常用命令

训练 YOLO 分割模型：

```powershell
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=8 device=0
```

验证模型：

```powershell
yolo segment val model=path/to/best.pt data=D:/MelonDataset/watermelon_seg/data.yaml
```

快速查看 Markdown 文件：

```powershell
rg --files -g "*.md"
```

## 后续维护原则

- 修改数据拆分、类别编号或标签格式时，同步更新 `README.md`、`data.yaml`、`classes.txt` 和本文件。
- 新增外部数据时，先记录来源、许可证、下载日期、原始位置和转换位置。
- 对标签做批量转换前，优先保留原始数据和中间结果，避免不可逆覆盖。
- 文档统一使用中文；数据集名、路径、命令、类别英文名和许可证名称可以保留原文。
