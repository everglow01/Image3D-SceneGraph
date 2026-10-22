# 初始候选机制与参考实现审查

**身份：** `MCMC-Q3-E001` · **阶段：** 历史索引，代码未迁移。

## 为什么做

把颜色裁剪、opacity/scale 一致性、目标动量和位置日程变成可检验候选。

## 已有结论

形成四个单因素候选；目标旧矩保留也是参考实现行为，不能直接定性为本项目独有 bug。

## 不能据此推断

只读源码、既有状态和有界机制检查不代替真实场景训练，也不证明候选修法有效。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

本研究项没有已确认的独立 `scripts/` 执行入口；保留原分析/证据位置，不创建虚假 run.py。

## 执行与证据

- [mcmc-fix-research-20261012-v1](records/mcmc-fix-research-20261012-v1.json)：执行 `analysis_completed`；验收 `candidate_hypotheses_only`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-fix-research-20261012-v1/assessment.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#q3)
- 研究动机来自：[MCMC-Q2-E001](../../../q2_rank_accumulation/experiments/e001_initial_audit/README.md)

整理不改变旧 Gate，不构成新执行授权。
