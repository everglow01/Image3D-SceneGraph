# 容量候选多周期恢复检查

**身份：** `MCMC-B3-E004` · **阶段：** 历史索引，代码未迁移。

## 为什么做

小门禁之后检验反复搬迁、可见／不可见和分片不平衡条件下的能力边界。

## 已有结论

受控研究完成但门禁阻塞：可见例 PSNR 差约 −1.2601 dB，无监督／不平衡条件有来源耗尽；未晋级长训。

## 不能据此推断

首次搬迁前已有极小数值差异，不能把终点差异全部归因于 capacity；也不能追认旧质量门槛通过。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_mcmc_capacity_cycles.py](../../../../../../scripts/run_mcmc_capacity_cycles.py)

### 原测试

- [tests/test_mcmc_capacity_cycles.py](../../../../../../tests/test_mcmc_capacity_cycles.py)

## 执行与证据

- [mcmc-capacity-cycles-results-20261017-v1](records/mcmc-capacity-cycles-results-20261017-v1.json)：执行 `completed_with_gate_failure`；验收 `gate_blocked_no_long_training`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-capacity-cycles-results-20261017-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/mcmc-capacity-cycles-results-20261017-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b3)
- 执行／证据依赖：[MCMC-B3-E003](../e003_capacity_gate/README.md)

整理不改变旧 Gate，不构成新执行授权。
