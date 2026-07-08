# 西瓜 YOLO 实例分割数据集

本目录是 Ultralytics YOLO 实例分割数据集，用于训练识别西瓜实例轮廓的分割模型。当前主数据集由两个 Roboflow COCO segmentation 导出数据集、本地 `raw/my_labelme` 标注和本地 `test_images` 标注融合生成。

## 目录结构

- `images/train`、`images/val`、`images/test`：融合后的训练、验证、测试图片。
- `labels/train`、`labels/val`、`labels/test`：与图片一一对应的 YOLO 分割标签。
- `raw`：原始采集数据和本地 LabelMe 标注来源，不应批量删除。
- `exports/fused_watermelon_yolo_seg`：融合拆分清单和统计摘要。
- `exports/my_labelme_yolo_seg`：本地 LabelMe 标注单独转换结果。
- `exports/graspable_yolo_seg`：面向机械臂抓取候选的派生数据集，按实例面积占比 `>=0.02` 保留可抓取大目标，主数据集不被覆盖。
- `external`：当前采用的 Roboflow COCO segmentation 原始导出和来源记录。
- `runs/runs1`、`runs/runs2` ...：按训练批次归档的 YOLO 训练、验证、测试和预测结果，索引见 `runs/RUNS_INDEX.md`。

## 当前主数据集

类别只有一个：

```text
0: watermelon
```

融合来源：

- Roboflow `Watermelon Detection` v5 `raw_annotations`，Instance Segmentation，CC BY 4.0。
- Roboflow `Watermelon Detector` v1，Instance Segmentation，CC BY 4.0。
- 本地 `raw/my_labelme` LabelMe 多边形标注。
- 本地 `test_images` LabelMe 多边形标注，已转换为 `exports/test_images_yolo_seg` 并以 `local_test_images_20260707_*` 文件名加入主数据集。

当前按随机种子 `20260707` 统一重拆分，并在人工可视化检查后移除 64 张问题图片及对应标签，当前为：

- 训练集：802 张图片 / 802 个标签 / 1798 个实例。
- 验证集：100 张图片 / 100 个标签 / 257 个实例。
- 测试集：101 张图片 / 101 个标签 / 254 个实例。

总计：1003 张图片 / 1003 个标签 / 2309 个实例。

2026-07-07 复查标签后，已删除 2 个极小异常多边形和 5 个重复实例。
2026-07-07 已将 `test_images` 中 10 张本地 LabelMe 标注图片加入主数据集，按 8/1/1 拆入 train/val/test。

详细拆分来源见 `exports/fused_watermelon_yolo_seg/split_manifest.tsv`，统计摘要见 `exports/fused_watermelon_yolo_seg/summary.json`。

## 标签格式

每张图片都应在对应的 `labels` 目录中有同名 `.txt` 标签文件。YOLO 分割标签每一行格式为：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

所有坐标均归一化到 `0.0-1.0`。本项目中 `class_id` 固定为 `0`，表示 `watermelon`。

## 训练命令

```powershell
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=8 device=0 project=D:/MelonDataset/watermelon_seg/runs/runs27/segment name=train
```

如果只能使用 CPU：

```powershell
yolo segment train model=yolo11n-seg.pt data=D:/MelonDataset/watermelon_seg/data.yaml imgsz=640 epochs=100 batch=4 device=cpu workers=0 project=D:/MelonDataset/watermelon_seg/runs/runs27/segment name=train
```

## 快速验证

```powershell
yolo segment val model=path/to/best.pt data=D:/MelonDataset/watermelon_seg/data.yaml
```

## 最新训练结果

融合数据集的 GPU 微调结果、参数和评估指标见：

```text
runs/runs5/segment/clean_finetune_768_precision_epoch10/TRAINING_REPORT.md
```

加入 `test_images` 后的 runs6 工业落地复训和评估报告见：

```text
runs/runs6/segment/INDUSTRIAL_READINESS_REPORT.md
```

runs7 对 1024 高分辨率微调、1024 推理和阈值/面积后处理做了复查，当前仍未达到 Mask Precision 95% / Recall 90%，报告见：

```text
runs/runs7/segment/INDUSTRIAL_READINESS_REPORT.md
```

runs8 继续尝试 YOLO11m 768，并新增“全实例”和“可抓取大目标”两套评估口径。全实例仍未达到 Mask Precision 95% / Recall 90%；面积占图至少 2% 的可抓取大目标口径下，runs6 和 runs8 均达到 P95/R90，报告见：

```text
runs/runs8/segment/INDUSTRIAL_READINESS_REPORT.md
```

runs9 在 runs6 基础上尝试 `mask_ratio=2`、copy-paste/mixup 复训，并验证轻量二阶段候选过滤器。全实例仍未达到 Mask Precision 95% / Recall 90%，报告见：

```text
runs/runs9/segment/INDUSTRIAL_READINESS_REPORT.md
```

runs10 尝试了切片推理和 train+val 固定轮数训练，全实例仍未达到 Mask Precision 95% / Recall 90%，报告见：

```text
runs/runs10/segment/INDUSTRIAL_READINESS_REPORT.md
```

runs11 验证了 TTA/多尺度推理无提升，并生成 Top 40 复标/补样队列，报告见：

```text
runs/runs11/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs11/analysis/RELABEL_REVIEW_QUEUE.md
```

runs12 进行了实例级标签审计，并量化了保守移除极小/贴边不可达 GT 的影响；清洗候选仍不足以达到 Mask Precision 95% / Recall 90%，报告见：

```text
runs/runs12/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs12/analysis/TEST_INSTANCE_AUDIT.md
```

runs13 正式派生了 `exports/graspable_yolo_seg` 可抓取目标数据集，并在该口径下复核当前最佳和微调模型。全实例仍未达标；机械臂抓取候选口径下，当前推荐 `runs6` 最佳模型配合 `conf>=0.70` 和预测 mask 面积占比 `>=0.02`，在可抓取 test 集达到 Precision 96.63% / Recall 91.49%。runs13 微调模型未超过该策略，报告见：

```text
runs/runs13/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs13/segment/industrial_readiness_summary.json
```

runs14 对上述推荐策略做了实例级 FP/FN 审计。全可抓取 GT 计入口径下达到 Precision 98.88% / Recall 90.72%；复现旧阈值脚本的贴边 GT 排除口径下为 Precision 96.63% / Recall 91.49%。报告见：

```text
runs/runs14/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs14/analysis/runs6_graspable_policy_error_audit_conf070_area002_all_gt/POLICY_ERROR_REVIEW.md
```

runs15 修正了 trainval 列表文件的 UTF-8 BOM，并继续尝试全实例 train+val 复训和 mask-NMS 后处理。最佳全实例后处理 F1 提升到 0.8834，但仍未达到 Precision 95% / Recall 90%，因此不晋升为推荐工业模型。报告见：

```text
runs/runs15/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs15/analysis/trainval_b4_best_full_instance_mask_nms_area_grid/mask_nms_summary.json
```

runs16 评估了 runs6 与 runs15 的双模型共识/并集后处理。双模型共识最佳 F1 为 0.8740，普通并集最佳 F1 为 0.8767，均低于 runs15 的最佳后处理结果，仍未达到 Precision 95% / Recall 90%。报告见：

```text
runs/runs16/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs16/analysis/runs6_runs15_full_instance_two_model_consensus_focused/two_model_consensus_summary.json
```

runs17 复核了 runs6 中的 YOLO11s 分支。全实例阈值/面积最佳 F1 为 0.8476，mask-NMS 最佳 F1 为 0.8487，均低于当前最佳后处理结果，仍未达到 Precision 95% / Recall 90%，因此不晋升。报告见：

```text
runs/runs17/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs17/analysis/yolo11s_best_full_instance_mask_nms_area_grid/mask_nms_summary.json
```

runs18 构建了不覆盖主数据集的难例过采样训练清单，并从 runs6 最佳权重继续微调。该策略把低阈值候选理论可达召回提高到 0.9528，但误报大幅增加，mask-NMS 最佳 F1 为 0.8716，仍未达到 Precision 95% / Recall 90%，因此不晋升。报告见：

```text
runs/runs18/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs18/segment/industrial_readiness_summary.json
exports/hard_oversample_yolo_seg_train/summary.json
```

runs19 验证了来源专属后处理：只用 val 集学习各来源的 `conf / mask-NMS / min area` 策略，再固定应用到 test。runs18 来源校准最佳 F1 为 0.8716，runs6 来源校准最佳 F1 为 0.8599，仍未达到 Precision 95% / Recall 90%。报告见：

```text
runs/runs19/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs19/segment/industrial_readiness_summary.json
```

runs20 将 train+val 合并并做难例过采样，从 runs15 最佳权重继续微调。低阈值理论可达召回提高到 0.9567，但 mask-NMS 最佳 F1 为 0.8787，仍低于 runs15 的 0.8834，且未达到 Precision 95% / Recall 90%。报告见：

```text
runs/runs20/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs20/segment/industrial_readiness_summary.json
```

runs21 从 runs20 的 train/val 误检中挖掘 424 张 hard negative 背景裁剪，并从 runs20 最佳权重继续微调。标准 test Mask Precision 为 95.7%、Recall 为 81.5%；mask-NMS 最佳 F1 为 0.8740，仍未达到 Precision 95% / Recall 90%，因此不晋升。当前机械臂抓取候选仍推荐 runs6 最佳模型配合 `conf>=0.70` 和预测 mask 面积占比 `>=0.02`。报告见：

```text
runs/runs21/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs21/segment/industrial_readiness_summary.json
```

runs22 基于 runs15 低阈值候选训练轻量二阶段过滤器。验证集策略固定到 test 后最好约 Precision 94.1% / Recall 81.5%；即使使用 test oracle 搜索，也无法同时达到 Precision 95% / Recall 90%。因此不晋升，当前机械臂抓取候选仍推荐 runs6 策略。报告见：

```text
runs/runs22/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs22/analysis/runs15_candidate_filter_mlp_conf003/candidate_filter_summary.json
```

runs23 对 runs15 做高召回 FP/FN 实例审计、来源专属校准、CNN 候选复核和边缘过滤网格。结果显示瓶颈集中在 `roboflow_team128_v5` 的叶片密集域：高召回点 55 个 FP 中 44 个来自该来源；来源校准、CNN 候选复核和边缘过滤均未达到 Precision 95% / Recall 90%。因此不晋升。报告见：

```text
runs/runs23/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs23/analysis/runs15_high_recall_mask_nms_error_audit/MASK_NMS_ERROR_AUDIT.md
```

runs24 基于 runs23 的瓶颈定位，从 `roboflow_team128_v5` 定向挖掘 266 张 hard negative 背景裁剪，并从 runs15 最佳权重继续微调。标准 test Mask Precision 为 95.1%、Recall 为 81.5%；mask-NMS 最佳 F1 为 0.8821，仍未达到 Precision 95% / Recall 90%，因此不晋升。当前机械臂抓取候选仍推荐 runs6 的可抓取大目标策略。报告见：

```text
runs/runs24/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs24/analysis/team128_hardneg_full_instance_mask_nms_area_grid/mask_nms_summary.json
```

runs25 复核了机械臂抓取更贴近现场的“可抓取大目标候选”部署策略：使用已训练的 runs6 与 runs13 双模型候选合并，在 `exports/graspable_yolo_seg` test 集上通过 `conf>=0.15`、预测 mask 面积占比 `>=0.02`、mask-NMS `0.3` 达到 `TP=95 / FP=4 / FN=2`，Precision 95.96%、Recall 97.94%。该策略已导出为可审计 YOLO 标签，并新增端到端预测脚本 `scripts/predict_graspable_two_model_policy.py`，可输出 `robot_candidates.csv`、候选 JSONL 和叠图预览；剩余错误集中在 `roboflow_team128_v5` 叶片密集域。该结论只适用于抓取候选离线检测，不代表全实例口径或真实机械臂闭环已经完成。报告见：

```text
runs/runs25/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs25/segment/industrial_grasp_candidate_summary.json
runs/runs25/analysis/runs6_runs13_graspable_policy_error_audit/POLICY_ERROR_REVIEW.md
runs/runs25/segment/DEPLOYMENT_NOTES.md
runs/runs25/segment/INDUSTRIAL_ACCEPTANCE_MATRIX.md
```

runs26 继续按全实例口径复核多模型缓存后处理。严格 P95/R90 仍未达到；但按最新放宽的“Precision 和 Recall 约 90%”口径，来源专属 runs8+runs24 策略在主数据集 test 上达到 `TP=227 / FP=28 / FN=27`，Precision 89.02%、Recall 89.37%、F1 89.19%。该策略已导出为 YOLO 标签，可作为当前离线全实例近 90/90 操作点；真实工位仍需现场相机域重新校准。报告见：

```text
runs/runs26/segment/INDUSTRIAL_READINESS_REPORT.md
runs/runs26/segment/source_aware_approx_p90r90_policy.json
runs/runs26/segment/source_aware_approx_p90r90_policy_export
runs/runs26/analysis/source_aware_approx_p90r90_policy_export_eval_all_gt
```

## 注意事项

- 当前根目录 `images` 和 `labels` 已替换为融合后的实例分割数据集。
- 不要把检测框标签直接混入本数据集；本项目主数据集只接受实例分割多边形标签。
- `exports/graspable_yolo_seg` 是工业抓取候选口径的派生数据集，不等同于主数据集全实例口径。
- 修改数据来源、拆分、类别编号或标签格式时，请同步更新 `README.md`、`data.yaml`、`classes.txt`、`AGENTS.md` 和 `external/SOURCES.md`。
