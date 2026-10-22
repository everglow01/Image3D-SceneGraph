# 原生控制 A/A 准入比较

**身份：** `MCMC-B1-E001` · **阶段：** 历史索引，代码未迁移。

## 为什么做

在完整 observer 执行前，检查共同最小审计下的两个控制是否可比较。

## 已有结论

使用 Q1 生命周期实验的同一次执行：12001 第一步起参数/Adam 不同，严格门禁失败。

## 不能据此推断

这是共享执行的门禁视角，不是又做了一组独立训练；不重复建立运行记录。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_mcmc_lifecycle_trace.py](../../../../../../scripts/run_mcmc_lifecycle_trace.py)

### 原测试

- [tests/test_mcmc_lifecycle_trace.py](../../../../../../tests/test_mcmc_lifecycle_trace.py)

## 执行与证据

- [mcmc-lifecycle-results-20261021-v1](../../../../questions/q1_post_relocation_survival/experiments/e002_lifecycle_trace/records/mcmc-lifecycle-results-20261021-v1.json)：执行 `controls_completed_panel_terminal_marker_missing`；验收 `gate1_failed_gate2_not_run`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/report.md)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b1)
- 执行／证据依赖：[MCMC-Q1-E002](../../../../questions/q1_post_relocation_survival/experiments/e002_lifecycle_trace/README.md)

整理不改变旧 Gate，不构成新执行授权。
