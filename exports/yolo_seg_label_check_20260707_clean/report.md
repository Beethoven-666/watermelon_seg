# YOLO segmentation label check

## Summary

- Images: 993
- Labels: 993
- Instances: 2200
- Errors: 0
- Warnings: 2
- Visualizations: 0

## Split counts

| split | images | labels | instances | errors | warnings | visuals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 794 | 794 | 1715 | 0 | 2 | 0 |
| val | 99 | 99 | 244 | 0 | 0 | 0 |
| test | 100 | 100 | 241 | 0 | 0 | 0 |

## Issue counts

- warning:tiny_polygon: 2

## First issues

| severity | split | line | code | image | message |
| --- | --- | ---: | --- | --- | --- |
| warning | train | 3 | tiny_polygon | roboflow_team128_v5_IMG_0377_jpg.rf.616677059599ed412b138e9a00e14bed.jpg | Normalized polygon area 0.00000066 is very small. |
| warning | train | 4 | tiny_polygon | roboflow_team128_v5_IMG_0606_jpg.rf.5ee996e9b1028c58e13a8dae82f30432.jpg | Normalized polygon area 0.00000049 is very small. |
