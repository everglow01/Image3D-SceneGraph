# 原库异尺寸单卡／双卡等价性诊断

**身份：** `MCMC-B2-E001` · **阶段：** 历史索引，代码未迁移。

## 为什么做

检查相机尺寸处理是否引入上游计算差异。

## 已有结论

异尺寸出现差异，但首轮同尺寸对照也失败，需先排除排序混杂。

## 不能据此推断

原始诊断受控制组阻塞，不能将所有差异直接归因于尺寸。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/smoke_distributed_camera_sizes.py](../../../../../../scripts/smoke_distributed_camera_sizes.py)

### 原测试

- [tests/test_distributed_camera_sizes.py](../../../../../../tests/test_distributed_camera_sizes.py)

## 执行与证据

- [camera-size-equivalence-results-20261019-v1](records/camera-size-equivalence-results-20261019-v1.json)：执行 `diagnostic_completed_with_failure`；验收 `control_blocked`。
  - [historical_assessment](../../../../../../outputs/analysis/camera-size-equivalence-results-20261019-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/camera-size-equivalence-results-20261019-v1/incoming/camera-size-equivalence-20261019-v1/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b2)

整理不改变旧 Gate，不构成新执行授权。
