# 12k 后逐点生命周期观测与门禁

**身份：** `MCMC-Q1-E002` · **阶段：** 历史索引，代码未迁移。

## 为什么做

追踪来源、目标、未参与点及贡献，解释恢复不对称；先验证控制重复性。

## 已有结论

原库 control_a/control_b 各完成 200 步，但第一步参数/Adam 已不同；Gate1 失败，完整 observer 与 repair 分支未运行。

## 不能据此推断

没有获得完整生命周期因果证据，不能把 A/A 失败归咎于尚未运行的完整 observer。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_mcmc_lifecycle_trace.py](../../../../../../scripts/run_mcmc_lifecycle_trace.py)
- [scripts/mcmc_lifecycle_observer.py](../../../../../../scripts/mcmc_lifecycle_observer.py)

### 原测试

- [tests/test_mcmc_lifecycle_trace.py](../../../../../../tests/test_mcmc_lifecycle_trace.py)
- [tests/test_mcmc_lifecycle_trace_remote.py](../../../../../../tests/test_mcmc_lifecycle_trace_remote.py)

### 原分析代码（仅导航，不执行下载内容）

- [outputs/analysis/mcmc-lifecycle-results-20261021-v1/analyze.py](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/analyze.py)

## 执行与证据

- [mcmc-lifecycle-results-20261021-v1](records/mcmc-lifecycle-results-20261021-v1.json)：执行 `controls_completed_panel_terminal_marker_missing`；验收 `gate1_failed_gate2_not_run`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/report.md)（本地已核 SHA；远端未重查）。
  - [frozen_protocol_copy](../../../../../../outputs/analysis/mcmc-lifecycle-results-20261021-v1/incoming/protocol.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#q1)
- 执行／证据依赖：[MCMC-B2-E005](../../../../branches/b2_camera_size_correctness/experiments/e005_scene_pair/README.md)
- 研究动机来自：[MCMC-Q1-E001](../e001_asymmetry_onset/README.md)

整理不改变旧 Gate，不构成新执行授权。
