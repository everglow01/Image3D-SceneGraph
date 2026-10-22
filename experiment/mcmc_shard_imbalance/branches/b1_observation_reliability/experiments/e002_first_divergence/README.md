# 首步 P0—P4 分歧定位

**身份：** `MCMC-B1-E002` · **阶段：** 历史索引，代码未迁移。

## 为什么做

确定恢复、forward、backward、Adam 和噪声中最早出现差异的边界。

## 已有结论

P0/P1 位级相同；首差在 P2，少量梯度和下游参数超容差。

## 不能据此推断

标量 Adam 公式复核不是实际内核重执行；该实验尚不能定位 rasterizer 内部或长期恢复机制。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_mcmc_first_divergence.py](../../../../../../scripts/run_mcmc_first_divergence.py)
- [scripts/mcmc_first_divergence.py](../../../../../../scripts/mcmc_first_divergence.py)

### 原测试

- [tests/test_mcmc_first_divergence.py](../../../../../../tests/test_mcmc_first_divergence.py)
- [tests/test_mcmc_first_divergence_remote.py](../../../../../../tests/test_mcmc_first_divergence_remote.py)

### 原分析代码（仅导航，不执行下载内容）

- [outputs/analysis/first-divergence-t1-results-20261021-v1/analyze.py](../../../../../../outputs/analysis/first-divergence-t1-results-20261021-v1/analyze.py)

## 执行与证据

- [first-divergence-t1-results-20261021-v1](records/first-divergence-t1-results-20261021-v1.json)：执行 `completed`；验收 `numerically_inconsistent`。
  - [historical_assessment](../../../../../../outputs/analysis/first-divergence-t1-results-20261021-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/first-divergence-t1-results-20261021-v1/report.md)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/first-divergence-t1-results-20261021-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b1)
- 执行／证据依赖：[MCMC-Q1-E002](../../../../questions/q1_post_relocation_survival/experiments/e002_lifecycle_trace/README.md)
- 研究动机来自：[MCMC-B1-E001](../e001_native_aa/README.md)

整理不改变旧 Gate，不构成新执行授权。
