# YOLO26 双 ROI 运行

本流程只包含 YOLO26s-seg 西瓜实例分割、YOLO26s Detect 果托检测和 ByteTrack 二维跟踪。
运行环境使用 `deploy/yolo26/requirements-windows.txt` 中已在本机验证的版本。

```powershell
python scripts/audit_watermelon_dataset.py
python scripts/audit_tray_dataset.py
python scripts/train_yolo26_watermelon_seg.py
python scripts/train_yolo26_tray_detect.py
python scripts/run_yolo26_dual_demo.py --config configs/yolo26_dual_runtime.yaml --source video.mp4
python scripts/predict_yolo26_watermelon_seg.py --weights path/to/watermelon_best.pt --source image.jpg --roi 0 0 .55 1 --no-window
python scripts/track_yolo26_trays.py --weights path/to/tray_best.pt --source video.mp4 --roi .45 0 1 1 --no-window
```

`--diagnose` 不运行模型，仅报告环境、数据、权重和输入源就绪状态。果托训练必须等真实审核标签，官方 COCO 权重不能冒充自定义果托权重。
