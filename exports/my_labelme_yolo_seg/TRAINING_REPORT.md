# my_labelme 训练报告

## 数据转换

- 来源：`D:/MelonDataset/watermelon_seg/raw/my_labelme`
- 输出：`D:/MelonDataset/watermelon_seg/exports/my_labelme_yolo_seg`
- 格式：YOLO 实例分割，类别 `0: watermelon`
- 拆分：train 37 张 / val 4 张 / test 6 张
- 实例数：train 408 个 / val 40 个 / test 63 个
- 拆分清单：`split_manifest.tsv`

## 训练结果

- 初始权重：`D:/MelonDataset/watermelon_seg/runs/runs1/segment/train/weights/best.pt`
- 最终权重：`D:/MelonDataset/watermelon_seg/runs/runs2/segment/my_labelme_finetune/weights/best.pt`
- 训练轮数：190 轮早停
- 训练尺寸：`imgsz=640`

验证集（`split=val`, `imgsz=640`）：

- Box: P 0.991 / R 0.875 / mAP50 0.964 / mAP50-95 0.841
- Mask: P 0.991 / R 0.875 / mAP50 0.939 / mAP50-95 0.729

测试集（`split=test`, `imgsz=960`，推荐评估/推理尺寸）：

- Box: P 0.981 / R 0.825 / mAP50 0.916 / mAP50-95 0.754
- Mask: P 0.981 / R 0.825 / mAP50 0.916 / mAP50-95 0.680

## 常用命令

训练：

```powershell
yolo segment train model=D:/MelonDataset/watermelon_seg/runs/runs1/segment/train/weights/best.pt data=D:/MelonDataset/watermelon_seg/exports/my_labelme_yolo_seg/data.yaml imgsz=640 epochs=200 batch=4 device=cpu workers=0 patience=50 seed=20260706 project=D:/MelonDataset/watermelon_seg/runs/runs2/segment name=my_labelme_finetune exist_ok=True plots=True close_mosaic=20 cache=True
```

测试集评估：

```powershell
yolo segment val model=D:/MelonDataset/watermelon_seg/runs/runs2/segment/my_labelme_finetune/weights/best.pt data=D:/MelonDataset/watermelon_seg/exports/my_labelme_yolo_seg/data.yaml split=test imgsz=960 batch=2 device=cpu workers=0
```

测试图预测可视化：

```powershell
yolo segment predict model=D:/MelonDataset/watermelon_seg/runs/runs2/segment/my_labelme_finetune/weights/best.pt source=D:/MelonDataset/watermelon_seg/exports/my_labelme_yolo_seg/images/test imgsz=960 conf=0.25 device=cpu
```
