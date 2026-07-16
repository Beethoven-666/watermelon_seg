# AGENTS.md

## 项目概览

这是 `D:/MelonDataset/watermelon_seg` 下的西瓜数据集工作区。根目录主数据集用于 Ultralytics YOLO 实例分割，目标类别只有 `0: watermelon`。

当前主数据集由两个 Roboflow COCO segmentation 导出数据集、本地 `raw/my_labelme` 标注和本地 `test_images` 标注融合生成，并统一转换为 YOLO 分割多边形格式。

当前拆分为（已在人工可视化检查后移除 64 张问题图片及对应标签，并按 `20260707` 种子重新拆分）：

- 训练集：802 张图片 / 802 个标签 / 1798 个实例。
- 验证集：100 张图片 / 100 个标签 / 257 个实例。
- 测试集：101 张图片 / 101 个标签 / 254 个实例。

总计：1003 张图片 / 1003 个标签 / 2309 个实例。

2026-07-07 复查标签后，已删除 2 个极小异常多边形和 5 个重复实例。
2026-07-07 已将 `test_images` 中 10 张本地 LabelMe 标注图片加入主数据集，按 8/1/1 拆入 train/val/test。

## 关键文件

- `data.yaml`：主数据集配置，`path` 指向 `D:/MelonDataset/watermelon_seg`。
- `classes.txt`：类别清单，目前只有 `watermelon`。
- `README.md`：面向使用者的主数据集说明。
- `external/SOURCES.md`：外部数据来源、许可证、转换结果和已移除旧来源。
- `exports/fused_watermelon_yolo_seg/summary.json`：融合数据统计摘要。
- `exports/fused_watermelon_yolo_seg/split_manifest.tsv`：融合数据拆分清单。
- `exports/my_labelme_yolo_seg`：本地 LabelMe 标注的单独 YOLO 分割转换结果。
- `exports/test_images_yolo_seg`：`test_images` 本地 LabelMe 标注的单独 YOLO 分割转换结果和并入记录。
- `exports/graspable_yolo_seg`：面向机械臂抓取候选的派生 YOLO 分割数据集，按实例面积占比 `>=0.02` 保留大目标，主数据集不被覆盖。

## 目录约定

- `images/train`、`images/val`、`images/test`：主融合分割数据集图片。
- `labels/train`、`labels/val`、`labels/test`：主融合分割数据集标签。
- `raw`：原始采集数据和本地 LabelMe 标注，不应在没有确认用途时批量删除或覆盖。
- `exports`：标注工具导出结果、格式转换中间产物、融合清单和统计摘要。
- `external`：当前采用的 Roboflow COCO segmentation 原始导出和来源记录。
- `runs/runs1`、`runs/runs2` ...：按训练批次归档的 YOLO 训练、验证、测试和预测结果；索引见 `runs/RUNS_INDEX.md`。
- `runs/runs6`：加入 `test_images` 后的工业落地目标复训和评估结果，综合报告见 `runs/runs6/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs7`：1024 高分辨率复训、1024 推理复核、阈值/面积后处理分析和最新工业落地复查，综合报告见 `runs/runs7/segment/INDUSTRIAL_READINESS_REPORT.md`。当前仍未达到 Mask Precision 95% / Recall 90%。
- `runs/runs8`：YOLO11m 768 复训、全实例阈值分析和可抓取大目标口径评估，综合报告见 `runs/runs8/segment/INDUSTRIAL_READINESS_REPORT.md`。全实例仍未达标；可抓取大目标口径下 runs6/runs8 均达到 P95/R90。
- `runs/runs9`：基于 runs6 权重的 `mask_ratio=2`、copy-paste/mixup 复训和轻量二阶段候选过滤器复核，综合报告见 `runs/runs9/segment/INDUSTRIAL_READINESS_REPORT.md`。全实例仍未达 P95/R90，当前最佳仍为 runs6。
- `runs/runs10`：切片推理和 train+val 固定轮数训练复核，综合报告见 `runs/runs10/segment/INDUSTRIAL_READINESS_REPORT.md`。全实例仍未达 P95/R90，当前最佳仍为 runs6。
- `runs/runs11`：TTA/多尺度推理复核和 Top 40 复标/补样队列，综合报告见 `runs/runs11/segment/INDUSTRIAL_READINESS_REPORT.md`，队列见 `runs/runs11/analysis/RELABEL_REVIEW_QUEUE.md`。全实例仍未达 P95/R90，当前最佳仍为 runs6。
- `runs/runs12`：实例级标签审计和保守清洗候选影响评估，综合报告见 `runs/runs12/segment/INDUSTRIAL_READINESS_REPORT.md`，审计见 `runs/runs12/analysis/TEST_INSTANCE_AUDIT.md`。清洗候选有提升但仍未达全实例 P95/R90。
- `runs/runs13`：派生可抓取目标数据集并做针对性微调。全实例仍未达标；抓取候选口径下推荐 `runs6` 最佳模型配合 `conf>=0.70`、预测 mask 面积占比 `>=0.02`，test Precision 96.63% / Recall 91.49%。runs13 微调模型未晋升，报告见 `runs/runs13/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs14`：对 `runs6 + conf>=0.70 + mask area>=0.02` 推荐策略做实例级误差审计。全可抓取 GT 计入口径为 Precision 98.88% / Recall 90.72%；旧阈值脚本贴边排除口径为 Precision 96.63% / Recall 91.49%。报告见 `runs/runs14/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs15`：修正 `exports/trainval_lists` UTF-8 BOM 后尝试 train+val 全实例复训和 mask-NMS 后处理。最佳全实例 F1 为 0.8834，仍未达 Precision 95% / Recall 90%，不晋升为推荐工业模型。报告见 `runs/runs15/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs16`：评估 `runs6` 与 `runs15` 双模型共识/并集后处理。双模型共识最佳 F1 为 0.8740，普通并集最佳 F1 为 0.8767，均未超过 runs15，仍未达全实例 P95/R90。报告见 `runs/runs16/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs17`：复核 `runs6` 的 YOLO11s 分支。全实例阈值/面积最佳 F1 为 0.8476，mask-NMS 最佳 F1 为 0.8487，低于当前最佳且仍未达 P95/R90，不晋升。报告见 `runs/runs17/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs18`：构建难例过采样训练清单并从 `runs6` 继续微调。低阈值理论可达召回提高到 0.9528，但误报大幅增加，mask-NMS 最佳 F1 为 0.8716，仍未达 P95/R90，不晋升。报告见 `runs/runs18/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs19`：验证来源专属后处理；只用 val 学习各来源 `conf/NMS/area`，再固定应用到 test。runs18 策略最佳 F1 为 0.8716，runs6 策略最佳 F1 为 0.8599，仍未达 P95/R90。报告见 `runs/runs19/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs20`：将 train+val 合并并做难例过采样，从 runs15 继续微调。低阈值可达召回提高到 0.9567，但 mask-NMS 最佳 F1 为 0.8787，仍低于 runs15 且未达 P95/R90，不晋升。报告见 `runs/runs20/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs21`：从 runs20 的 train/val 误检中挖掘 424 张 hard negative 背景裁剪，并从 runs20 继续微调。标准 test Mask P95.7/R81.5，mask-NMS 最佳 F1 为 0.8740，仍未达 P95/R90，不晋升；机械臂抓取候选仍推荐 runs6 策略。报告见 `runs/runs21/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs22`：基于 runs15 低阈值候选训练轻量二阶段过滤器。val 策略固定到 test 最好约 P94.1/R81.5；test oracle 也无法同时达到 P95/R90，不晋升；机械臂抓取候选仍推荐 runs6 策略。报告见 `runs/runs22/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs23`：对 runs15 做高召回 FP/FN 实例审计、来源专属校准、CNN 候选复核和边缘过滤网格。瓶颈集中在 `roboflow_team128_v5` 叶片密集域，仍未达 P95/R90，不晋升；机械臂抓取候选仍推荐 runs6 策略。报告见 `runs/runs23/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs24`：从 `roboflow_team128_v5` 定向挖掘 hard negative 并从 runs15 继续微调。标准 test Mask P95.1/R81.5，mask-NMS 最佳 F1 为 0.8821，仍未达 P95/R90，不晋升；机械臂抓取候选仍推荐 runs6 策略。报告见 `runs/runs24/segment/INDUSTRIAL_READINESS_REPORT.md`。
- `runs/runs25`：复核 runs6 + runs13 双模型抓取候选部署策略，在 `exports/graspable_yolo_seg` test 集达到 P95.96/R97.94，满足可抓取大目标离线候选检测 P95/R90；已导出策略后 YOLO 标签、完成 4 FP / 2 FN 审计，并新增 `scripts/predict_graspable_two_model_policy.py` 端到端预测入口，可输出 `robot_candidates.csv`、候选 JSONL 和叠图预览；错误集中在 `roboflow_team128_v5` 叶片密集域；全实例口径仍未达标，真实机械臂闭环仍需现场验证。报告见 `runs/runs25/segment/INDUSTRIAL_READINESS_REPORT.md`，验收矩阵见 `runs/runs25/segment/INDUSTRIAL_ACCEPTANCE_MATRIX.md`。
- `runs/runs26`：复核全实例多模型缓存后处理。严格 P95/R90 仍未达标；按用户最新“Precision/Recall 约 90%”口径，来源专属 runs8+runs24 策略在主数据集 test 达到 `TP=227 / FP=28 / FN=27`，Precision 89.02%、Recall 89.37%、F1 89.19%，可作为当前离线全实例近 90/90 策略。报告见 `runs/runs26/segment/INDUSTRIAL_READINESS_REPORT.md`，策略见 `runs/runs26/segment/source_aware_approx_p90r90_policy.json`。
- `runs/runs27`：已完成 `facebookresearch/sam3` 零样本水果分割部署与西瓜 val→test 严格评测。test COCO Mask AP50 87.06%、AP75 75.30%、mAP50-95 70.08%；val 冻结阈值后 test Precision 84.05%、Recall 85.04%、F1 84.54%、正样本实例 Accuracy 73.22%，未达 P90/R90；同源/重复图去重估计 F1 83.91%；8 GB GPU 峰值 allocated 5.86 GiB。报告见 `runs/runs27/sam3/watermelon_zero_shot_benchmark/SAM3_WATERMELON_EVALUATION.md`，部署说明见 `deploy/sam3/README.md`。

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
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=8 device=0 project=D:/MelonDataset/watermelon_seg/runs/runs27/segment name=train
```

验证模型：

```powershell
yolo segment val model=path/to/best.pt data=D:/MelonDataset/watermelon_seg/data.yaml
```

验证可抓取目标派生数据集：

```powershell
yolo segment val model=D:/MelonDataset/watermelon_seg/runs/runs6/segment/industrial_recall_finetune_768/weights/best.pt data=D:/MelonDataset/watermelon_seg/exports/graspable_yolo_seg/data.yaml split=test imgsz=768 device=0
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
