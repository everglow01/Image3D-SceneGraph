# 原始 apartment 退化与分片状态核验

**身份：** `MCMC-Q2-E001` · **阶段：** 历史索引，代码未迁移。

## 为什么做

核实长期质量异常和分片低透明度是否真实存在，排查未跑完或漏更新等简单解释。

## 已有结论

原始 MCMC 30k 的 rank1 低透明度占比约 96.164%，rank0 约 17.4182%；参数有限、两 rank Adam step 一致。

## 不能据此推断

Project/MCMC 训练链不同，不是单因素 A/B；原始一次运行的 best=3k 不能推广为所有基线。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原测试

- [tests/test_mcmc_diagnostics.py](../../../../../../tests/test_mcmc_diagnostics.py)

本研究项没有已确认的独立 `scripts/` 执行入口；保留原分析/证据位置，不创建虚假 run.py。

## 执行与证据

- [apartment-training-audit-20261012-v1](records/apartment-training-audit-20261012-v1.json)：执行 `analysis_completed`；验收 `historical_observation`。
  - [historical_assessment](../../../../../../outputs/analysis/apartment-training-audit-20261012-v1/assessment.json)（本地已核 SHA；远端未重查）。
- [mcmc-instability-verification-20261012-v1](records/mcmc-instability-verification-20261012-v1.json)：执行 `analysis_completed`；验收 `state_checked_not_causal`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-instability-verification-20261012-v1/assessment.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#q2)

整理不改变旧 Gate，不构成新执行授权。
