# AGENTS.md

## 项目概览

这是 `D:/MelonDataset/watermelon_seg` 下的西瓜数据集工作区。根目录主数据集用于 Ultralytics YOLO 实例分割，目标类别只有 `0: watermelon`。

当前主数据集由两个 Roboflow COCO segmentation 导出数据集和本地 `raw/my_labelme` 标注融合生成，并统一转换为 YOLO 分割多边形格式。

当前拆分为：

- 训练集：845 张图片 / 845 个标签 / 1826 个实例。
- 验证集：105 张图片 / 105 个标签 / 226 个实例。
- 测试集：107 张图片 / 107 个标签 / 196 个实例。

总计：1057 张图片 / 1057 个标签 / 2248 个实例。

## 关键文件

- `data.yaml`：主数据集配置，`path` 指向 `D:/MelonDataset/watermelon_seg`。
- `classes.txt`：类别清单，目前只有 `watermelon`。
- `README.md`：面向使用者的主数据集说明。
- `external/SOURCES.md`：外部数据来源、许可证、转换结果和已移除旧来源。
- `exports/fused_watermelon_yolo_seg/summary.json`：融合数据统计摘要。
- `exports/fused_watermelon_yolo_seg/split_manifest.tsv`：融合数据拆分清单。
- `exports/my_labelme_yolo_seg`：本地 LabelMe 标注的单独 YOLO 分割转换结果。

## 目录约定

- `images/train`、`images/val`、`images/test`：主融合分割数据集图片。
- `labels/train`、`labels/val`、`labels/test`：主融合分割数据集标签。
- `raw`：原始采集数据和本地 LabelMe 标注，不应在没有确认用途时批量删除或覆盖。
- `exports`：标注工具导出结果、格式转换中间产物、融合清单和统计摘要。
- `external`：当前采用的 Roboflow COCO segmentation 原始导出和来源记录。

## 标签格式

主数据集使用 YOLO 实例分割格式，每张图片应有同名 `.txt` 标签文件。标签行格式为：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

所有坐标均归一化到 `0.0-1.0`。当前项目中 `class_id` 固定为 `0`，含义为 `watermelon`。

## 当前外部数据边界

- `external/roboflow_team128_watermelon_detection_v5_coco_segmentation` 是 Roboflow `Watermelon Detection` v5 的 COCO Segmentation 导出。
- `external/roboflow_drago_watermelon_detector_v1_coco_segmentation` 是 Roboflow `Watermelon Detector` v1 的 COCO Segmentation 导出。
- 两个 Roboflow 数据集均为 Instance Segmentation，融合时已统一映射为 `0: watermelon`。
- 不要把检测框标签直接混入根目录的实例分割数据集；只有 polygon segmentation 标签才可进入主数据集。
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

- 修改数据拆分、类别编号或标签格式时，同步更新 `README.md`、`data.yaml`、`classes.txt`、`AGENTS.md` 和 `external/SOURCES.md`。
- 新增外部数据时，先记录来源、许可证、下载日期、原始位置和转换位置。
- 对标签做批量转换前，优先保留原始数据和中间结果，避免不可逆覆盖。
- 文档统一使用中文；数据集名、路径、命令、类别英文名和许可证名称可以保留原文。
