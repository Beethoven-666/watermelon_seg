# SAM 3 零样本水果分拣部署

本目录提供 `facebookresearch/sam3` 的 Windows 原生部署基线。第一阶段使用 SAM 3 图像模型和英文概念提示做离线图片验证；SAM 3.1 的主要改进是视频多目标跟踪，待单图精度、显存和延迟通过后再接传送带视频。

## 当前状态（2026-07-13）

已完成：

- Conda 环境：`C:\Users\zyh68\.conda\envs\sam3`，Python 3.12.13。
- PyTorch 2.10.0 + CUDA 12.8 可用，GPU 为 RTX 5060 Laptop 8 GB，支持 bfloat16。
- 官方源码已克隆到 `D:\MelonDataset\sam3`，固定核对提交为 `5dd401d1c5c1d5c3eedff06d41b77af824517619`。
- `sam3` editable 安装、CUDA 张量、TorchVision CUDA NMS 和运行时导入均已通过。
- Windows 补充依赖已锁定在 `requirements-windows.txt`，其中 `triton-windows` 是社区 Windows 移植，不是 Meta 官方 Windows 发行物。
- 多水果批量预测入口：`scripts/predict_sam3_zero_shot_fruits.py`。
- 西瓜 val→test 严格评测入口：`scripts/benchmark_sam3_watermelon.py`。
- Hugging Face gated 权限已批准；官方权重已下载到 `D:\MelonDataset\sam3\sam3.pt`，SHA-256 为 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
- 官方固定 `resolution=1008` 的单图 CUDA 前向已通过，峰值 allocated 5.41 GiB、reserved 6.02 GiB。
- 完整 val→test 评估已完成：test Mask AP50 87.06%、AP75 75.30%、mAP50-95 70.08%；冻结 val 阈值后 P84.05/R85.04/F1 84.54/正样本实例 Accuracy 73.22，未达到 P90/R90。后验拆分审计估计同源/重复图使 F1 高估约 0.63 个百分点，去重后约 83.91%。

## 新机器只需完成一次的授权

1. 登录 [facebook/sam3 模型页](https://huggingface.co/facebook/sam3)，申请并等待访问批准。
2. 在 Anaconda Prompt 中执行以下命令。令牌只输入 Hugging Face CLI，不要发到聊天、代码或日志中。

```powershell
conda activate sam3
hf auth login
hf auth whoami
hf download facebook/sam3 config.json
```

最后一条只下载小型配置并验证 gated 权限。成功后即可由脚本自动下载约 3.45 GB 的 `sam3.pt`；也可提前执行：

```powershell
hf download facebook/sam3 sam3.pt
```

## 环境诊断

```powershell
conda activate sam3
cd D:\MelonDataset\watermelon_seg
python scripts\predict_sam3_zero_shot_fruits.py --diagnose
```

`ready_for_inference=true` 表示 Python、CUDA、SAM3 导入和权重来源已就绪。本机 Windows Triton、峰值显存与真实前向已经由下面的单图/完整评测验证；换机器后仍需重做诊断。

## 西瓜准确率测试

正式基准会：

1. 仅使用英文提示 `watermelon`，不让 apple/orange 等提示词竞争西瓜 mask。
2. 在 `images/val` 的 100 张 / 257 个实例上选择置信度。
3. 冻结该阈值后，只在 `images/test` 的 101 张 / 254 个实例上评估一次。
4. 使用原图尺寸 mask 和官方 `pycocotools.COCOeval` 计算 Mask AP50、AP75、mAP50-95、AR100。
5. 使用 mask IoU≥0.50 的最大基数一对一匹配计算 TP、FP、FN、Precision、Recall 和 F1，避免预测遍历顺序改变 TP 数。

先检查数据和参数，不加载模型：

```powershell
python scripts\benchmark_sam3_watermelon.py --dry-run
```

权限就绪后执行完整基准：

```powershell
python scripts\benchmark_sam3_watermelon.py `
  --prompt watermelon `
  --resolution 1008 `
  --max-image-side 1024 `
  --candidate-floor 0 `
  --min-area-ratio 0 `
  --max-detections 100 `
  --checkpoint D:\MelonDataset\sam3\sam3.pt `
  --output-dir runs\runs27\sam3\watermelon_zero_shot_benchmark
```

输出包括：

- `SAM3_WATERMELON_EVALUATION.md`：中文结论与验收判断。
- `benchmark_summary.json`：机器可读完整指标，并记录数据集组合 SHA256、checkpoint SHA256 和官方源码提交。
- `val_threshold_curve.csv`：仅用于 val 选阈值。
- `test_threshold_curve_oracle_only.csv`：仅作诊断，禁止据此挑部署阈值。
- `per_image_metrics_at_val_threshold.csv`：冻结阈值下的逐图结果。
- `raw_score_iou_cache.json`：逐图分数和预测/真值 IoU 矩阵，便于不重复推理地复查阈值与匹配。

当前 test 的 101 张图全部含西瓜，没有空传送带、纯背景或其他水果负样本。因此，即使本次 AP/P/R 很高，也不能证明模型不会把苹果、叶片或机械结构误报成西瓜。上线前必须再冻结一套现场“多水果 + 空背景 + 遮挡 + 反光 + 运动模糊”测试集。

## 多水果零样本预测

默认配置在 `fruit_prompts.yaml`。提示词应使用简短英文名词短语；默认启用 watermelon、apple、orange、banana、pear，其他水果按需启用。`fruit` 通用提示只能发现“某种水果”，不能自动给出水果亚类。

先做不加载模型的扫描：

```powershell
python scripts\predict_sam3_zero_shot_fruits.py `
  --dry-run `
  --source test_images `
  --recursive
```

真实批量推理：

```powershell
python scripts\predict_sam3_zero_shot_fruits.py `
  --source test_images `
  --recursive `
  --output-dir runs\runs27\sam3\zero_shot_fruit_predict
```

只查询一种临时水果时，可不用改 YAML：

```powershell
python scripts\predict_sam3_zero_shot_fruits.py `
  --source path\to\images `
  --prompt dragon_fruit="dragon fruit" `
  --output-dir runs\runs27\sam3\dragon_fruit_predict
```

主要输出：

- `robot_candidates.csv`：兼容现有分拣候选风格，增加 class、prompt、mask 尺寸和抓取候选点。
- `candidates.jsonl`：归一化几何、原图像素坐标、mask 元数据和有损 polygon 标记。
- `masks/`：推理尺寸下的无损二值 PNG；`mask_width/mask_height` 与原图尺寸同时写入结果。
- `overlays/`：人工可视化复核图。
- `prediction_manifest.tsv` / `prediction_summary.json`：逐图状态、延迟、显存与错误。

为了避免 4032×3024 原图上的大量候选 mask 在 GPU 上直接上采样导致 OOM，配置会先把传给 SAM 的图片最长边限制为 1024。所有归一化坐标会映射回原图，mask PNG 保留推理尺寸并明确记录尺寸；任何评测器都不得把它误当成原图尺寸 mask。

`grasp_point` 是二值 mask 距离变换得到的最深内部点，保证位于 mask 内，但仍只是二维图像候选。真实机械臂必须叠加深度有效性、相机内外参、末端执行器尺寸、遮挡、碰撞和急停逻辑。

## 分辨率与显存策略

当前官方图像 checkpoint 的旋转位置编码固定为 `resolution=1008`；改成 560 或 756 会触发形状断言，不能把 `resolution` 当作显存旋钮。本机 8 GB 显存采用以下策略：

1. 固定 `resolution=1008` 与 `amp_dtype=bfloat16`。
2. 用 `max_image_side=1024` 控制候选 mask 上采样尺寸；坐标仍映射回原图。
3. 单图实测峰值 allocated 约 5.41 GiB、reserved 约 6.02 GiB。
4. 若 Windows Triton JIT 或显存仍不稳定，再迁移到 WSL2/Linux 或更大显存 GPU；不要先投入相机实时流。

## 数据边界与许可证

- SAM 3 多水果预测属于 run-local 推理结果，不得写入根目录 `labels/`，也不得修改主数据集固定的 `0: watermelon`。
- 零样本输出不是人工真值，未经抽检和转换不得混入训练集。
- SAM 3 使用自定义 [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)，不是 MIT。工业产品分发源码、权重或衍生物前应归档许可证并由法务复核。
- 官方资料：[SAM 3 README](https://github.com/facebookresearch/sam3)、[论文](https://arxiv.org/abs/2511.16719)、[模型权重](https://huggingface.co/facebook/sam3)。
