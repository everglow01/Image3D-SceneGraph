# 目标相机尺寸隔离修复门禁

**身份：** `MCMC-B2-E003` · **阶段：** 历史索引，代码未迁移。

## 为什么做

只改投影使用的目标尺寸，保持原 CUDA 二进制，验证计算契约。

## 已有结论

12 个合成用例、每例 5 步通过原容差。

## 不能据此推断

固定拓扑、fresh Adam 的合成等价性，不是 MCMC 长期恢复或真实场景质量通过。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/prepare_gsplat_camera_sizes.py](../../../../../../scripts/prepare_gsplat_camera_sizes.py)
- [scripts/smoke_distributed_camera_sizes.py](../../../../../../scripts/smoke_distributed_camera_sizes.py)

### 原测试

- [tests/test_prepare_gsplat_camera_sizes.py](../../../../../../tests/test_prepare_gsplat_camera_sizes.py)
- [tests/test_distributed_camera_sizes.py](../../../../../../tests/test_distributed_camera_sizes.py)

## 执行与证据

- [camera-size-repair-results-20261019-v1](records/camera-size-repair-results-20261019-v1.json)：执行 `completed`；验收 `synthetic_equivalence_passed`。
  - [historical_assessment](../../../../../../outputs/analysis/camera-size-repair-results-20261019-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/camera-size-repair-results-20261019-v1/incoming/reports/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b2)
- 执行／证据依赖：[MCMC-B2-E002](../e002_order_controls/README.md)

整理不改变旧 Gate，不构成新执行授权。
