# tray_video_capture 使用文档（彩色视频采集）

本脚本实现果托采集阶段的彩色视频录制，使用 `pyorbbecsdk` 读取 Gemini 435Le 彩色流，输出与后续 `scripts/extract_tray_frames.py` 兼容的独立视频文件与元数据。

## 1. 功能范围

- ✅ 采集彩色视频（仅 RGB/BGR 原始图像流）
- ✅ 生成输出视频、逐帧 CSV、同名元数据 JSON、片段日志
- ✅ 支持连续分段、按时长自动停止、键盘分段退出
- ✅ 自动文件名递增、固定临时文件写入与校验后原子重命名
- ✅ 支持固定测试视频专用目录
- ✅ 支持 manifest 持久化（`raw/tray_detect/capture_manifest.csv`）
- ❌ 不采集深度、不做 RGB-D 对齐、不做 3D、抽帧、标注、YOLO/ByteTrack
- ❌ 不支持一次录制多个相机

> 已按指定解释器核对：`D:\Programming Software\Python\python.exe` 为 Python 3.13.14，已安装 `pyorbbecsdk2 2.1.2`（导入名 `pyorbbecsdk`；SDK 运行时版本 2.9.3）、OpenCV 4.13.0 和 NumPy。本脚本按该 SDK 的 `Pipeline(device)`、`get_default_video_stream_profile()` 与 `Config.enable_stream(profile)` 接口实现；当前执行 `--list-devices` 未发现 Gemini 435Le（退出码 3），因此真实硬件录制测试未执行。

## 2. 安装与调用前提

- Python 可执行文件：`D:\Programming Software\Python\python.exe`
- 依赖：`opencv-python`、`numpy`、`pyorbbecsdk`（采集时必须可导入）
- 运行前确保 `raw/`、`raw/tray_detect/` 可写

## 3. 命令行参数（完整）

```text
--output-root PATH
--batch-id TEXT
--serial-number TEXT
--list-devices
--list-profiles
--width INT
--height INT
--fps FLOAT
--format TEXT
--duration-seconds FLOAT
--segment-seconds FLOAT
--warmup-seconds FLOAT
--no-preview
--fixed-test
--planned-split {unassigned,train,val,test,fixed_test}
--lighting TEXT
--scene-tags TEXT
--tray-count TEXT
--camera-moved {yes,no,unknown}
--conveyor-running {yes,no,mixed,unknown}
--notes TEXT
--codec TEXT
--container {mp4,avi}
--frame-timeout-ms INT
--log-level {DEBUG,INFO,WARNING,ERROR}
```

## 4. 关键行为

- `--batch-id`：普通采集中必填；`--fixed-test` 时可省略
- `--fixed-test`：视频输出进入 `raw/tray_detect/fixed_test_videos/`，`planned_split` 会强制为 `fixed_test`
- `--list-devices / --list-profiles`：仅列出信息，不启动录制
- 多设备且未传 `--serial-number`：返回错误，要求显式指定序列号
- 指定宽高/帧率/格式但参数不完整：返回错误；三项同时给出使用显式配置
- 同时未指定宽高/帧率：使用设备默认配置（记录实际参数）
- 任何已有文件不会被覆盖；输出名会自动递增
- 文件写入采用 `*.partial.mp4`（或 `*.partial.avi`）中转，验证可读后改为正式名
- `q` 或 `Esc` 正常结束整个采集；`n` 正常结束当前片段并立即创建下一段
- `--no-preview --duration-seconds 0` 时没有键盘窗口输入，只能使用 Ctrl+C 结束
- 发生相机、超时或写盘错误时，已写入的内容会保留为 partial/incomplete 记录，并在 JSON/manifest 标注非 `completed` 状态；不会标为训练可用视频

退出码：`0` 正常；`2` 参数错误；`3` 未发现设备；`4` 多设备选择不明确；`5` 不支持的彩色流；`6` Pipeline 启动失败；`7` 编码或写盘失败；`8` 连续超时或相机断开；`9` 输出校验失败；`10` 未处理异常。

## 5. 输出目录与文件清单

### 5.1 普通采集

```text
raw/tray_detect/
├─ capture_manifest.csv
└─ <batch-id>/
   ├─ 20260820T120000_Gemini_435Le_SNxxxx_normal_light_001.mp4
   ├─ 20260820T120000_Gemini_435Le_SNxxxx_normal_light_001.json
   ├─ 20260820T120000_Gemini_435Le_SNxxxx_normal_light_001.frames.csv
   └─ 20260820T120000_Gemini_435Le_SNxxxx_normal_light_001.log
```

### 5.2 固定测试

```text
raw/tray_detect/
├─ capture_manifest.csv
└─ fixed_test_videos/
   ├─ 20260820T120000_Gemini_435Le_SNxxxx_fixed_test_001.mp4
   ├─ 20260820T120000_Gemini_435Le_SNxxxx_fixed_test_001.json
   ├─ 20260820T120000_Gemini_435Le_SNxxxx_fixed_test_001.frames.csv
   └─ 20260820T120000_Gemini_435Le_SNxxxx_fixed_test_001.log
```

## 6. 文件格式

- 视频：原始彩色帧；不包含叠加文字
- `.frames.csv` 字段：
  - `video_frame_index`, `sdk_frame_index`, `device_timestamp_ms`, `host_monotonic_ns`, `host_time_iso`, `inter_frame_delta_ms`, `conversion_ok`, `write_ok`
- JSON（同名 `.json`）包含：
  - capture/schema/device/stream/encoding/statistics/scene/integrity/software/resolved_arguments
- manifest：`raw/tray_detect/capture_manifest.csv`（UTF-8/utf-8-sig 写入）

## 7. 典型命令

### 7.1 普通训练候选视频

```powershell
$Py = 'D:\Programming Software\Python\python.exe'

& $Py scripts\capture_tray_videos.py `
  --output-root raw\tray_detect `
  --batch-id 20260820_normal_light `
  --serial-number <SERIAL> `
  --duration-seconds 120 `
  --segment-seconds 30 `
  --planned-split unassigned `
  --lighting normal `
  --scene-tags empty_tray,loaded_tray,entry_exit `
  --tray-count 0-3 `
  --camera-moved no `
  --conveyor-running yes `
  --notes "normal light"
```

### 7.2 固定测试视频

```powershell
$Py = 'D:\Programming Software\Python\python.exe'

& $Py scripts\capture_tray_videos.py `
  --output-root raw\tray_detect `
  --serial-number <SERIAL> `
  --duration-seconds 30 `
  --fixed-test `
  --lighting normal `
  --scene-tags entry_exit,multi_tray,short_occlusion `
  --tray-count 0-3 `
  --camera-moved no `
  --conveyor-running yes `
  --notes "ByteTrack only"
```

### 7.3 仅列表接口

```powershell
$Py = 'D:\Programming Software\Python\python.exe'
& $Py scripts\capture_tray_videos.py --list-devices
& $Py scripts\capture_tray_videos.py --list-profiles --serial-number <SERIAL>
```

## 8. 常用测试命令

- 无硬件单元测试（已通过）：
  - `& 'D:\Programming Software\Python\python.exe' -m pytest tests/test_capture_tray_videos.py -q`
- 合成冒烟（脚本内测试用例中包含 30~60+ 帧 NumPy 合成帧录制）

## 9. 已知限制（当前）

- 尚未连接 Gemini 435Le，因此硬件录制测试未执行；连接后应先执行 `--list-devices` 和按序列号执行 `--list-profiles`
- `--format` 与 SDK 常量映射遵循当前脚本兼容实现（RGB/BGR/MJPG/YUYV，未知格式直接报错）
- 多机并行采集未实现（按要求仅支持单机、单序列号）
- 固定测试视频仅用于连续视频/ByteTrack 测试，禁止抽帧进入任何 train、val 或 test 图像训练数据集
