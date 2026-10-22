# 尺寸修复的原生生命周期门禁

**身份：** `MCMC-B2-E004` · **阶段：** 历史索引，代码未迁移。

## 为什么做

在真实 trainer 的短合成生命周期中验证原库与隔离修复的加载、更新和保存链路。

## 已有结论

原库与修复的 12 步生命周期门禁通过；本地保留 protocol、summary 与逐文件核验。

## 不能据此推断

本目录没有统一 assessment.json，以实际 verification.json/summary.json 为证据，不补造旧摘要。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_camera_size_lifecycle.py](../../../../../../scripts/run_camera_size_lifecycle.py)

### 原测试

- [tests/test_camera_size_lifecycle.py](../../../../../../tests/test_camera_size_lifecycle.py)

## 执行与证据

- [camera-size-lifecycle-results-20261021-v1](records/camera-size-lifecycle-results-20261021-v1.json)：执行 `completed`；验收 `short_lifecycle_passed`。
  - [historical_assessment](../../../../../../outputs/analysis/camera-size-lifecycle-results-20261021-v1/verification.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/camera-size-lifecycle-results-20261021-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b2)
- 执行／证据依赖：[MCMC-B2-E003](../e003_size_repair/README.md)

整理不改变旧 Gate，不构成新执行授权。
