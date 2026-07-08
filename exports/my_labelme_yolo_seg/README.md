# my_labelme YOLO 分割数据集

该目录由 `raw/my_labelme` 中的 LabelMe 多边形标注转换生成，类别只有 `0: watermelon`。

- 随机种子：`20260706`
- train：37 张图片 / 37 个标签 / 408 个实例
- val：4 张图片 / 4 个标签 / 40 个实例
- test：6 张图片 / 6 个标签 / 63 个实例

训练命令示例：

```powershell
yolo segment train model=D:/MelonDataset/watermelon_seg/runs/runs1/segment/train/weights/best.pt data=D:/MelonDataset/watermelon_seg/exports/my_labelme_yolo_seg/data.yaml imgsz=640 epochs=200 batch=4 device=cpu workers=0 patience=50 seed=20260706 project=D:/MelonDataset/watermelon_seg/runs/runs2/segment name=my_labelme_finetune exist_ok=True plots=True close_mosaic=20 cache=True
```
