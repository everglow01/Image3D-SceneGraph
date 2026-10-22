# 死亡起点与搬迁后恢复的时序审计

**身份：** `MCMC-Q1-E001` · **阶段：** 历史索引，代码未迁移。

## 为什么做

先回答 relocation 是初始死亡的来源，还是未能持续恢复已有死亡。

## 已有结论

两条 12k 轨迹中死亡先于首次 relocation；固定点数阶段搬迁即时缩小分片死亡数差，但后续训练又增加死亡。

## 不能据此推断

只有总量日志，无法把新死亡归到上一轮目标、来源或未参与点；不是逐点再死亡率。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原分析代码（仅导航，不执行下载内容）

- [outputs/analysis/mcmc-asymmetry-trigger-readonly-20261021-v1/analyze.py](../../../../../../outputs/analysis/mcmc-asymmetry-trigger-readonly-20261021-v1/analyze.py)

本研究项没有已确认的独立 `scripts/` 执行入口；保留原分析/证据位置，不创建虚假 run.py。

## 执行与证据

- [mcmc-asymmetry-trigger-readonly-20261021-v1](records/mcmc-asymmetry-trigger-readonly-20261021-v1.json)：执行 `analysis_completed`；验收 `aggregate_timing_only`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-asymmetry-trigger-readonly-20261021-v1/assessment.json)（本地已核 SHA；远端未重查）。
  - [report.md](../../../../../../outputs/analysis/mcmc-asymmetry-trigger-readonly-20261021-v1/report.md)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#q1)
- 执行／证据依赖：[MCMC-B2-E005](../../../../branches/b2_camera_size_correctness/experiments/e005_scene_pair/README.md)

整理不改变旧 Gate，不构成新执行授权。
