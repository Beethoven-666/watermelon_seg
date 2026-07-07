# 融合西瓜数据集 YOLO 实例分割训练报告

生成日期：2026-07-07

## 结论

推荐使用权重：

```text
D:/MelonDataset/watermelon_seg/runs/segment/fused_finetune_640_gpu_from_mylabelme/weights/best.pt
```

推荐推理尺寸为 `imgsz=640`。测试集中 `imgsz=960` 的 `mAP50` 基本持平，但 `mAP50-95` 更低，因此暂不作为默认推理设置。

## 数据集

当前根目录数据集由两个 Roboflow COCO segmentation 数据集和本地 `raw/my_labelme` 融合生成，类别只有：

```text
0: watermelon
```

清理后统计：

| split | images | labels | instances |
| --- | ---: | ---: | ---: |
| train | 845 | 845 | 1826 |
| val | 105 | 105 | 226 |
| test | 107 | 107 | 196 |
| total | 1057 | 1057 | 2248 |

训练后复查结果：图片和标签一一对应，无孤立标签、无缺失标签，标签类别和归一化坐标范围正常。

## 微调策略

这次没有从通用 COCO 权重直接开始，而是从已有的本地西瓜模型继续微调：

```text
D:/MelonDataset/watermelon_seg/runs/segment/my_labelme_finetune/weights/best.pt
```

这样做的原因是该权重已经学到西瓜实例轮廓，比 `yolo11n-seg.pt` 的通用类别初始化更适合当前单类别任务。

主要调整：

- 使用 GPU 环境 `melonvision`：PyTorch `2.8.0+cu128`，GPU 为 NVIDIA GeForce RTX 5060 Laptop GPU 8GB。
- `batch` 从原先较保守的设置提高到 `16`，充分利用显存。
- 优化器使用 `AdamW`，初始学习率设置为 `lr0=0.003`，比常见默认值更保守，适合在已有西瓜模型上继续微调。
- 使用 `cos_lr=True` 让学习率后期平滑下降。
- 使用 `close_mosaic=10`，最后 10 个 epoch 关闭 mosaic，帮助模型回到更真实的图片分布。
- 使用 `cache=disk`，减少反复读取大图和标签的开销。
- 使用 `patience=25` 保留早停机制，但本次完整训练到 100 epoch。

## 训练命令

```powershell
C:\Users\zyh68\.conda\envs\melonvision\Scripts\yolo.exe segment train `
  model=D:/MelonDataset/watermelon_seg/runs/segment/my_labelme_finetune/weights/best.pt `
  data=D:/MelonDataset/watermelon_seg/data.yaml `
  imgsz=640 epochs=100 batch=16 device=0 workers=4 patience=25 seed=20260707 `
  project=D:/MelonDataset/watermelon_seg/runs/segment `
  name=fused_finetune_640_gpu_from_mylabelme exist_ok=True plots=True close_mosaic=10 `
  cache=disk cos_lr=True optimizer=AdamW lr0=0.003 lrf=0.01
```

关键参数：

| parameter | value |
| --- | --- |
| task | segment |
| model | previous watermelon `best.pt` |
| imgsz | 640 |
| epochs | 100 |
| batch | 16 |
| optimizer | AdamW |
| lr0 / lrf | 0.003 / 0.01 |
| patience | 25 |
| close_mosaic | 10 |
| cache | disk |
| seed | 20260707 |
| device | 0 |

## 训练过程指标

训练日志中分割指标最高点：

| metric | best epoch | value |
| --- | ---: | ---: |
| Mask mAP50 | 60 | 0.84986 |
| Mask mAP50-95 | 80 | 0.62949 |
| Mask precision | 70 | 0.95356 |
| Mask recall | 91 | 0.80973 |
| Box mAP50 | 68 | 0.86476 |
| Box mAP50-95 | 81 | 0.70003 |

第 100 轮最后一次验证：

| type | precision | recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Box | 0.92046 | 0.80531 | 0.85220 | 0.68944 |
| Mask | 0.91783 | 0.80088 | 0.83362 | 0.62437 |

## 最终复评

使用 `best.pt` 重新评估 `val`：

| type | precision | recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Box | 0.951 | 0.766 | 0.858 | 0.697 |
| Mask | 0.907 | 0.781 | 0.835 | 0.623 |

使用 `best.pt` 评估 `test`，`imgsz=640`：

| type | precision | recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Box | 0.833 | 0.776 | 0.836 | 0.675 |
| Mask | 0.822 | 0.765 | 0.812 | 0.593 |

使用 `best.pt` 评估 `test`，`imgsz=960`：

| type | precision | recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Box | 0.856 | 0.724 | 0.825 | 0.630 |
| Mask | 0.843 | 0.714 | 0.813 | 0.556 |

由于 `imgsz=960` 的精细阈值指标下降，最终推荐 `imgsz=640`。

## 基线对比

融合数据集验证集上的对比：

| model | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: |
| `yolo11n-seg.pt` 通用权重 | 0.0128 | 0.0086 |
| 旧 `my_labelme_finetune/best.pt` | 0.258 | 0.174 |
| 新融合数据集微调 `best.pt` | 0.835 | 0.623 |

说明：新模型相对旧本地模型对融合数据集的泛化明显提升，尤其是 Roboflow 来源图片和密集多西瓜场景。

## 可视化结果

测试集预测图已生成到：

```text
D:/MelonDataset/watermelon_seg/runs/segment/fused_gpu_best_predict_test_640
```

评估图和曲线在：

```text
D:/MelonDataset/watermelon_seg/runs/segment/fused_finetune_640_gpu_from_mylabelme
D:/MelonDataset/watermelon_seg/runs/segment/fused_gpu_best_val_640
D:/MelonDataset/watermelon_seg/runs/segment/fused_gpu_best_test_640
```

## 推荐使用命令

验证：

```powershell
C:\Users\zyh68\.conda\envs\melonvision\Scripts\yolo.exe segment val `
  model=D:/MelonDataset/watermelon_seg/runs/segment/fused_finetune_640_gpu_from_mylabelme/weights/best.pt `
  data=D:/MelonDataset/watermelon_seg/data.yaml split=test imgsz=640 batch=16 device=0
```

预测图片或文件夹：

```powershell
C:\Users\zyh68\.conda\envs\melonvision\Scripts\yolo.exe segment predict `
  model=D:/MelonDataset/watermelon_seg/runs/segment/fused_finetune_640_gpu_from_mylabelme/weights/best.pt `
  source=你的图片或文件夹路径 imgsz=640 conf=0.25 device=0
```

## 训练后的数据修正

测试集有一张图中存在两个外接框完全相同的重复多边形，Ultralytics 会自动忽略其中一个。已手动删除冗余实例，并同步更新 `README.md`、`AGENTS.md`、`external/SOURCES.md` 和融合数据统计文件。该修正只影响测试集统计，不影响已经完成的训练集学习过程。

