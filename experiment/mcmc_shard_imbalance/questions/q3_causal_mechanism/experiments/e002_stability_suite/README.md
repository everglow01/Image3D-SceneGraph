# 五臂真实场景稳定性对照

**身份：** `MCMC-Q3-E002` · **阶段：** 历史索引，代码未迁移。

## 为什么做

分别检验四个候选，避免一次混改后无法归因。

## 已有结论

五臂均实际执行，baseline/raw_loss 完成，floor、目标矩清零、位置日程三臂失败；raw_loss 改善颜色但未消除分片低透明度失衡。

## 不能据此推断

非有限 scale、CUDA OOM 和显存保护停止是不同失败；单场景单 seed，不自动推广任何候选。

## 设计与代码

参数、容差、预算以各运行记录指向的原协议为准；无协议副本时标明缺失，不补造。当前文件可能晚于历史运行版本，不能仅凭同名脚本复现旧结果。

### 原执行／辅助代码

- [scripts/run_mcmc_stability_ablation.py](../../../../../../scripts/run_mcmc_stability_ablation.py)
- [scripts/run_mcmc_stability_suite.py](../../../../../../scripts/run_mcmc_stability_suite.py)
- [scripts/prepare_gsplat_mcmc.py](../../../../../../scripts/prepare_gsplat_mcmc.py)
- [scripts/smoke_mcmc_stability.py](../../../../../../scripts/smoke_mcmc_stability.py)
- [scripts/smoke_mcmc_validation_memory.py](../../../../../../scripts/smoke_mcmc_validation_memory.py)

### 原测试

- [tests/test_mcmc_stability_ablation.py](../../../../../../tests/test_mcmc_stability_ablation.py)
- [tests/test_mcmc_stability_suite.py](../../../../../../tests/test_mcmc_stability_suite.py)
- [tests/test_prepare_gsplat_mcmc.py](../../../../../../tests/test_prepare_gsplat_mcmc.py)

## 执行与证据

- [mcmc-suite-results-20261017-v1](records/mcmc-suite-results-20261017-v1.json)：执行 `completed_with_failures`；验收 `two_arms_completed_three_failed`。
  - [historical_assessment](../../../../../../outputs/analysis/mcmc-suite-results-20261017-v1/assessment.json)（本地已核 SHA；远端未重查）。

## 关联

- [所属问题／支线](../../../../README.md#q3)
- 研究动机来自：[MCMC-Q3-E001](../e001_candidate_research/README.md)

整理不改变旧 Gate，不构成新执行授权。

- 审核摘要：[五臂对照：颜色改善不等于恢复失衡已解决](reports/mcmc-suite-results-20261017-v1.md)
