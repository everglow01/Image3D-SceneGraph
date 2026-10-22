# 反向源码与 ULP／累加规模只读分析

**身份：** `MCMC-B1-E005` · **阶段：** 历史索引，代码未迁移。

## 为什么做

在已有快照上缩小微观候选，不额外执行 forward/backward。

## 已有结论

源码和差异分布最支持浮点原子累加顺序候选；近零大 ULP 不等于大绝对误差。

## 不能据此推断

没有动态累加顺序追踪、确定性干预或独占因果证明；两次只读分析不能算新独立重复。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原分析代码（仅导航，不执行下载内容）

- [outputs/analysis/first-divergence-backward-readonly-20261021-v1/analyze.py](../../../../../../outputs/analysis/first-divergence-backward-readonly-20261021-v1/analyze.py)
- [outputs/analysis/first-divergence-backward-readonly-20261021-v1/inspect_single_camera.py](../../../../../../outputs/analysis/first-divergence-backward-readonly-20261021-v1/inspect_single_camera.py)
- [outputs/analysis/first-divergence-backward-readonly-20261021-v1/inspect_snapshots.py](../../../../../../outputs/analysis/first-divergence-backward-readonly-20261021-v1/inspect_snapshots.py)
- [outputs/analysis/rasterizer-internal-readonly-20261022-v1/analyze.py](../../../../../../outputs/analysis/rasterizer-internal-readonly-20261022-v1/analyze.py)
- [outputs/analysis/rasterizer-internal-readonly-20261022-v1/inspect_existing.py](../../../../../../outputs/analysis/rasterizer-internal-readonly-20261022-v1/inspect_existing.py)

本研究项没有已确认的独立 `scripts/` 执行入口；保留原分析/证据位置，不创建虚假 run.py。

## 执行与证据

- [first-divergence-backward-readonly-20261021-v1](records/first-divergence-backward-readonly-20261021-v1.json)：执行 `analysis_completed`；验收 `not_causal_isolation`。
  - [historical_assessment](../../../../../../outputs/analysis/first-divergence-backward-readonly-20261021-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/first-divergence-backward-readonly-20261021-v1/report.md)（本地已核 SHA；远端未重查）。
- [rasterizer-internal-readonly-20261022-v1](records/rasterizer-internal-readonly-20261022-v1.json)：执行 `analysis_completed`；验收 `atomic_order_candidate_not_proven`。
  - [historical_assessment](../../../../../../outputs/analysis/rasterizer-internal-readonly-20261022-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/rasterizer-internal-readonly-20261022-v1/report.md)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#b1)
- 执行／证据依赖：[MCMC-B1-E002](../e002_first_divergence/README.md)
- 执行／证据依赖：[MCMC-B1-E003](../e003_backward_boundary/README.md)

整理不改变旧 Gate，不构成新执行授权。
