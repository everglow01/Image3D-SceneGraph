# 图像恢复梯度与尺度下溢机制

**身份：** `MCMC-B3-E002` · **阶段：** 历史索引，代码未迁移。

## 为什么做

区分恢复信号不足、重复分份和 floor 修改产生的数值失效。

## 已有结论

有效可见微型例图像恢复梯度可远大于正则；floor 压力例出现尺度下溢。

## 不能据此推断

证明受控例失效路径，不等于重演旧 24.9k 点的 lineage；无新真实场景训练。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/smoke_mcmc_degeneration.py](../../../../../../scripts/smoke_mcmc_degeneration.py)

### 原测试

- [tests/test_mcmc_degeneration_gate.py](../../../../../../tests/test_mcmc_degeneration_gate.py)

## 执行与证据

- [mcmc-mechanism-details-results-20261017-v1](records/mcmc-mechanism-details-results-20261017-v1.json)：执行 `completed`；验收 `stability_not_passed`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-mechanism-details-results-20261017-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/mcmc-mechanism-details-results-20261017-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b3)
- 研究动机来自：[MCMC-B3-E001](../e001_degeneration_probe/README.md)

整理不改变旧 Gate，不构成新执行授权。
