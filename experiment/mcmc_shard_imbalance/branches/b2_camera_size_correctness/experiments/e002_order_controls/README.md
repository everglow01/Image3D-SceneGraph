# 统一 Gaussian 顺序的尺寸对照

**身份：** `MCMC-B2-E002` · **阶段：** 历史索引，代码未迁移。

## 为什么做

从单卡／双卡比较中剥离拼接顺序和深度排序影响。

## 已有结论

排序控制通过，异尺寸问题仍在，修复目标收敛为每目标相机的 width/height。

## 不能据此推断

执行退出码 1 保留；控制通过不意味着未修复库的所有比较通过。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/smoke_distributed_camera_sizes.py](../../../../../../scripts/smoke_distributed_camera_sizes.py)

### 原测试

- [tests/test_distributed_camera_sizes.py](../../../../../../tests/test_distributed_camera_sizes.py)

## 执行与证据

- [camera-size-order-results-20261019-v2](records/camera-size-order-results-20261019-v2.json)：执行 `diagnostic_completed_with_failure`；验收 `order_controls_passed_original_not_equivalent`。
  - [historical_assessment](../../../../../../outputs/analysis/camera-size-order-results-20261019-v2/assessment.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/camera-size-order-results-20261019-v2/incoming/reports/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b2)
- 研究动机来自：[MCMC-B2-E001](../e001_native_equivalence/README.md)

整理不改变旧 Gate，不构成新执行授权。
