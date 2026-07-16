# SAM 3 西瓜微调数据集（sam3_watermelon_finetune_v1）

本目录由 `scripts/build_sam3_finetune_dataset.py` 确定性生成。图片和 YOLO 标签仍保留在项目根目录，未复制、未覆盖；COCO `file_name` 均相对 `D:/MelonDataset/watermelon_seg`。

## 拆分

| 拆分 | 图片 | 实例 | 背景图 | 场景键 |
| --- | ---: | ---: | ---: | ---: |
| train_clean | 651 | 1616 | 1 | 556 |
| val_clean | 95 | 250 | 1 | 93 |
| test_frozen | 101 | 254 | 0 | 97 |
| test_scene_unique（敏感性视图） | 97 | 249 | 0 | 97 |

类别固定为 COCO `1: watermelon`，文本提示词为 `watermelon`。空标签图片被保留，SAM 3 的 `COCO_FROM_JSON(include_negatives=true)` 会把它们作为负查询。

## 防泄漏策略

1. 源 `images/test` 与 `labels/test` 完全冻结，指纹必须为 `c5fd1505c69860ca359ee3f9affc60e6a9dab08734fc888b4ab05014561b4f9b`。
2. scene key 为 `source + Roboflow 文件名 .rf. 前的原始资产名`；非 Roboflow 文件使用完整 stem。
3. val 中与 test scene/SHA 冲突的图片被隔离。
4. train 中与 test 或 clean val scene/SHA 冲突的图片被隔离。
5. clean train 内字节相同的图片仅在实例可一对一匹配且 mask IoU ≥ 0.99 时去重；冲突时构建直接失败。

该策略消除了已知的同原图增强与字节重复泄漏，但不能证明采集会话独立。本地 `test_images` 是连续拍摄批次，`my_labelme` 也包含短时间连续帧；最终工业验收仍应使用新采集、独立场景的现场 test。

## 文件

- `annotations/instances_train_clean.json`
- `annotations/instances_val_clean.json`
- `annotations/instances_test_frozen.json`
- `annotations/instances_test_scene_unique.json`
- `manifests/images.tsv`：所有基础派生拆分图片及 SHA-256、场景键、COCO ID。
- `manifests/instances.tsv`：逐实例 bbox/area、源标签行和 mask/polygon SHA-256。
- `manifests/scene_groups.tsv`：原始场景组及派生去向。
- `manifests/exclusions.tsv`：所有隔离或去重记录。
- `summary.json`：数量、来源和生成文件摘要。
- `integrity.json`：冻结 test、重复语义和交叉拆分检查。

## SAM 3 数据路径

Hydra 数据配置应使用：

```yaml
img_folder: D:/MelonDataset/watermelon_seg
ann_file: D:/MelonDataset/watermelon_seg/exports/sam3_watermelon_finetune_v1/annotations/instances_train_clean.json
coco_json_loader:
  _target_: sam3.train.data.coco_json_loaders.COCO_FROM_JSON
  include_negatives: true
  category_chunk_size: 1
  _partial_: true
```

实例分割训练还必须同时启用 `load_segmentation`、`with_seg_masks`、模型 `enable_segmentation`、`DecodeRle` 和 `sam3.train.loss.loss_fns.Masks`。`test_frozen` 不得进入训练、选 epoch、阈值选择或超参数调优。
