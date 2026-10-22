# 低透明度退化受控检查

**身份：** `MCMC-B3-E001` · **阶段：** 历史索引，代码未迁移。

## 为什么做

将长训失败缩为有图像信号／无信号下的 opacity 与优化器动态。

## 已有结论

完成各臂和独立 unit/监控核验；外层执行面板结束标记缺失，不能伪造 finished。

## 不能据此推断

受控动态不是旧真实点的逐点历史重演；不存在“正则一定杀死所有搬迁点”的结论。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/smoke_mcmc_degeneration.py](../../../../../../scripts/smoke_mcmc_degeneration.py)

### 原测试

- [tests/test_mcmc_degeneration_gate.py](../../../../../../tests/test_mcmc_degeneration_gate.py)

## 执行与证据

- [mcmc-degeneration-results-20261017-v1](records/mcmc-degeneration-results-20261017-v1.json)：执行 `unit_completed_panel_terminal_marker_missing`；验收 `controlled_probe_not_long_term_proof`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-degeneration-results-20261017-v1/assessment.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b3)
- 研究动机来自：[MCMC-Q3-E002](../../../../questions/q3_causal_mechanism/experiments/e002_stability_suite/README.md)

整理不改变旧 Gate，不构成新执行授权。
