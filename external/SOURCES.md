# 外部西瓜数据来源

下载并整理日期：2026-07-06。

## 1. FruitSeg30 分割数据集与掩膜标注

- 来源：https://data.mendeley.com/datasets/vkht8pfsp3/3
- DOI：10.17632/vkht8pfsp3.3
- 许可证：CC BY 4.0
- 已下载压缩包：`external/FruitSeg30_Mendeley_v3.zip`
- 已解压目录：`external/FruitSeg30_Mendeley_v3/`
- 原始西瓜子集：`external/FruitSeg30_watermelon/`
  - `Images`：42 个 JPG 文件。
  - `Mask`：42 个 PNG 掩膜文件。
- YOLO 分割格式转换结果：`external/FruitSeg30_watermelon_yolo_seg/`
  - 配置文件：`external/FruitSeg30_watermelon_yolo_seg/data.yaml`
  - 训练集：34 张图片 / 34 个标签。
  - 验证集：4 张图片 / 4 个标签。
  - 测试集：4 张图片 / 4 个标签。
- 已同步复制到项目主训练数据集中：
  - `images/train`、`labels/train`：34 对文件。
  - `images/val`、`labels/val`：4 对文件。
  - `images/test`、`labels/test`：4 对文件。
- 说明：掩膜已转换为 YOLO 分割多边形，类别为 `0: watermelon`。

## 2. lightly-ai/dataset_fruits_detection

- 来源：https://github.com/lightly-ai/dataset_fruits_detection
- 下载的提交：`e626559d4a127ea9fc35fedd74c52ee9f378a53d`
- 许可证：CC0-1.0
- 完整克隆目录：`external/lightly_dataset_fruits_detection/`
- 只保留西瓜类别后的检测子集：`external/lightly_watermelon_detection_yolo/`
  - 训练集：746 张图片 / 746 个标签 / 1683 个西瓜检测框。
  - 验证集：107 张图片 / 107 个标签 / 217 个西瓜检测框。
  - 测试集：47 张图片 / 47 个标签 / 76 个西瓜检测框。
- 说明：这个来源是目标检测数据，不是实例分割数据。筛选后的标签已从原始类别 `5: Watermelon` 重映射为 `0: watermelon`；除非计划训练检测模型，否则应与 YOLO 分割数据集分开保存。

## 候选但未下载的数据源

- Roboflow 中 Zane 发布的 `watermelon`：https://universe.roboflow.com/zane-fbnlx/watermelon-yl6is
  - 公开页面显示它是实例分割数据集，许可证为 CC BY 4.0，约 1.1k 张图片；但类别是 `leaf`，与西瓜果实分割的目标不完全一致。
- Roboflow 中 WaterMelon Disease Detection 发布的 `Segmentation 2`：https://universe.roboflow.com/watermelon-disease-detection/segmentation-2-kaplf
  - 公开页面显示它是目标检测数据集，许可证为 CC BY 4.0，约 516 张图片；类别主要是病害类型，适合病害检测，不适合作为西瓜果实分割数据。
