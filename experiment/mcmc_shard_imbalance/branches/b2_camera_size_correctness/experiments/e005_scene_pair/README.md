# 真实 apartment 12k 尺寸配对

**身份：** `MCMC-B2-E005` · **阶段：** 历史索引，代码未迁移。

## 为什么做

检查正确性修复能否改善真实场景中的退化与分片恢复。

## 已有结论

两臂完成 12k；修复未消除死亡总量和恢复差异，较弱 owner 从 rank1 变为 rank0。

## 不能据此推断

只覆盖 12k，不是旧 15k 以后百万级退化的完整复现；监控面板结束标记缺失保持原记录。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_camera_size_scene_pair.py](../../../../../../scripts/run_camera_size_scene_pair.py)

### 原测试

- [tests/test_camera_size_scene_pair.py](../../../../../../tests/test_camera_size_scene_pair.py)

### 原分析代码（仅导航，不执行下载内容）

- [outputs/analysis/camera-size-scene-pair-results-20261021-v1/analyze.py](../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/analyze.py)

## 执行与证据

- [camera-size-scene-pair-results-20261021-v1](records/camera-size-scene-pair-results-20261021-v1.json)：执行 `completed_monitor_terminal_marker_missing`；验收 `verified_not_promoted`。
  - [historical_assessment](../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/report.md)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/camera-size-scene-pair-results-20261021-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b2)
- 执行／证据依赖：[MCMC-B2-E004](../e004_native_lifecycle/README.md)

整理不改变旧 Gate，不构成新执行授权。

- 审核摘要：[真实 12k 配对：尺寸修复不是最终答案](reports/camera-size-scene-pair-results-20261021-v1.md)
