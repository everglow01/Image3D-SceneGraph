# 完整 B0／B1 反向边界定位

**身份：** `MCMC-B1-E003` · **阶段：** 历史索引，代码未迁移。

## 为什么做

区分传入图像梯度变化与 rasterizer backward 自身的变化。

## 已有结论

完整 B0 与图像梯度位级一致，B1 已不同；首差位于 rasterizer backward 内、反向通信之前。

## 不能据此推断

B1 在容差内不豁免后续 P2/P3 超容差；没有证明唯一微观原因或长期退化因果。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_mcmc_backward_boundary.py](../../../../../../scripts/run_mcmc_backward_boundary.py)

### 原测试

- [tests/test_mcmc_backward_boundary.py](../../../../../../tests/test_mcmc_backward_boundary.py)
- [tests/test_mcmc_backward_boundary_remote.py](../../../../../../tests/test_mcmc_backward_boundary_remote.py)

### 原分析代码（仅导航，不执行下载内容）

- [outputs/analysis/backward-boundary-results-20261022-v1/analyze.py](../../../../../../outputs/analysis/backward-boundary-results-20261022-v1/analyze.py)
- [outputs/analysis/backward-boundary-results-20261022-v1/verify_boundary_metrics.py](../../../../../../outputs/analysis/backward-boundary-results-20261022-v1/verify_boundary_metrics.py)

## 执行与证据

- [backward-boundary-results-20261022-v1](records/backward-boundary-results-20261022-v1.json)：执行 `completed`；验收 `downstream_numerically_inconsistent`。
  - [historical_assessment](../../../../../../outputs/analysis/backward-boundary-results-20261022-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/backward-boundary-results-20261022-v1/report.md)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/backward-boundary-results-20261022-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b1)
- 执行／证据依赖：[MCMC-B1-E002](../e002_first_divergence/README.md)

整理不改变旧 Gate，不构成新执行授权。
