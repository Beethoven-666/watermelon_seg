# 清理标签后微调报告

生成日期：2026-07-07

## 目标和结论

本次按 `val` split 的 YOLO 分割指标 `Mask precision` 判断“准确率 95%”。最终候选权重在独立验证中达到：

```text
D:/MelonDataset/watermelon_seg/runs/runs5/segment/clean_finetune_768_precision_epoch10/weights/best.pt
```

推荐评估和推理尺寸：`imgsz=768`。该权重在 `val` 上达到 `Mask precision=0.953`，满足 95% 目标；在 `test` 上为 `0.926`，未达到 95%，因此测试集泛化仍需继续提升。

## 标签清理

训练前先修正当前数据集中的标签问题：

- 删除 2 个极小异常多边形。
- 删除 5 个重复或近重复实例。
- 复查结果：993 张图片 / 993 个标签 / 2193 个实例，errors=0，warnings=0。
- 复查报告：`exports/yolo_seg_label_check_20260707_clean_after_fix2/report.md`

清理后拆分为：

| split | images | labels | instances |
| --- | ---: | ---: | ---: |
| train | 794 | 794 | 1709 |
| val | 99 | 99 | 244 |
| test | 100 | 100 | 240 |

## 基线

旧最佳权重：

```text
D:/MelonDataset/watermelon_seg/runs/runs4/segment/fused_finetune_640_gpu_from_mylabelme/weights/best.pt
```

在清理后数据上的基线：

| model | split | imgsz | Box P | Box mAP50 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old best | val | 640 | 0.942 | 0.888 | 0.881 | 0.847 | 0.889 | 0.670 |
| old best | test | 640 | 0.922 | 0.942 | 0.913 | 0.879 | 0.929 | 0.714 |

## 微调设置

从旧最佳权重继续训练，使用更高输入尺寸和较低学习率：

```powershell
C:\Users\zyh68\.conda\envs\melonvision\Scripts\yolo.exe segment train `
  model=D:/MelonDataset/watermelon_seg/runs/runs4/segment/fused_finetune_640_gpu_from_mylabelme/weights/best.pt `
  data=D:/MelonDataset/watermelon_seg/data.yaml `
  imgsz=768 epochs=10 batch=8 device=0 workers=4 patience=0 seed=20260707 deterministic=True `
  project=D:/MelonDataset/watermelon_seg/runs/runs5/segment `
  name=clean_finetune_768_precision_epoch10 exist_ok=True plots=True `
  close_mosaic=10 cache=disk cos_lr=True optimizer=AdamW lr0=0.001 lrf=0.01
```

关键变化：

- `imgsz`: 640 -> 768，提高验证时的分割精度。
- `batch`: 16 -> 8，适配更高输入尺寸。
- `lr0`: 0.003 -> 0.001，作为旧权重继续微调时降低学习率。
- 清理标签后重新生成缓存，避免重复实例和异常小多边形进入训练。

## 训练结果

训练共 10 epoch，最后一轮也是本轮最优分割精度：

| epoch | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.939 | 0.759 | 0.872 | 0.712 | 0.958 | 0.751 | 0.869 | 0.639 |

独立复评：

| model | split | imgsz | Box P | Box mAP50 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| new best | val | 768 | 0.940 | 0.874 | 0.953 | 0.753 | 0.870 | 0.638 |
| new best | test | 768 | 0.930 | 0.926 | 0.926 | 0.833 | 0.921 | 0.697 |
| new best | val | 640 | 0.937 | 0.876 | 0.898 | 0.787 | 0.864 | 0.634 |
| new best | test | 640 | 0.927 | 0.935 | 0.918 | 0.841 | 0.924 | 0.696 |

## 使用建议

若目标是验证集 `Mask precision >= 0.95`，使用：

```text
D:/MelonDataset/watermelon_seg/runs/runs5/segment/clean_finetune_768_precision_epoch10/weights/best.pt
```

推理建议：

```powershell
yolo segment predict model=D:/MelonDataset/watermelon_seg/runs/runs5/segment/clean_finetune_768_precision_epoch10/weights/best.pt source=你的图片或文件夹路径 imgsz=768 conf=0.25 device=0
```

如果目标要求测试集也达到 95%，当前模型尚未达成，需要继续清理 test 中的漏标/误标样本，或增加更接近测试分布的训练数据。
