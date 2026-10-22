# 容量约束候选的小型 CUDA 门禁

**身份：** `MCMC-B3-E003` · **阶段：** 历史索引，代码未迁移。

## 为什么做

真实复制数量与分份公式一致，容量不足时延期，而不是强制复活全部目标。

## 已有结论

小型 CUDA 门禁通过；保留真实复制计数与候选范围。

## 不能据此推断

不等于真实场景质量或长期存活通过，也不提供跨 owner 来源。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/mcmc_capacity_ops.py](../../../../../../scripts/mcmc_capacity_ops.py)
- [scripts/prepare_gsplat_mcmc.py](../../../../../../scripts/prepare_gsplat_mcmc.py)
- [scripts/smoke_mcmc_stability.py](../../../../../../scripts/smoke_mcmc_stability.py)

### 原测试

- [tests/test_mcmc_capacity.py](../../../../../../tests/test_mcmc_capacity.py)
- [tests/test_prepare_gsplat_mcmc.py](../../../../../../tests/test_prepare_gsplat_mcmc.py)

## 执行与证据

- [mcmc-capacity-results-20261017-v1](records/mcmc-capacity-results-20261017-v1.json)：执行 `completed`；验收 `small_cuda_gate_passed_quality_not_evaluated`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-capacity-results-20261017-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/mcmc-capacity-results-20261017-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b3)
- 研究动机来自：[MCMC-B3-E002](../e002_mechanism_details/README.md)

整理不改变旧 Gate，不构成新执行授权。
