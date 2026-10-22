# 固定 B0 五次重复与六点敏感性

**MCMC-B1-E004** · [所属支线 B1](../../../../README.md#b1) · [身份与来源索引](experiment.json)

## 目的、证据与限制

测量固定输入下原 backward 的重复性，解释少量既有异常点的 Adam 敏感性。

- [v1](records/fixed-b0-results-20261022-v1.json)：首次导入改变 Python RNG，前置失败，真实回放未开始。
- [v2](records/fixed-b0-results-20261022-v2.json)：审计初始化后，两相机各五次，80 项比较均在原容差内，无非有限。
- 有限重复不证明全局界，也不豁免下游差异。六点 P2/P3 来自历史 A/A，**不是本次 B1 的连通传播**；旧 Gate 不追认，长期因果未证。

原协议仍是参数、预算、容差与绑定权威。同名文件可能晚于历史运行，复现须核原身份；前置依赖为 [B0/B1 边界实验](../e003_backward_boundary/README.md)。

## 现役入口

### 历史 v2 文本分析

[analyze.py](analyze.py) 合并原两份分析，限定 [analysis-source.json](analysis-source.json) 绑定的 33 份文本或逐字副本，不是通用报告器。输入只读，输出须是本实验 runs 下全新目录；下例若存在，换新 ID，不覆盖。

```bash
python3 -I experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e004_fixed_b0/analyze.py \
  --input-root outputs/analysis/fixed-b0-results-20261022-v2 \
  --output-dir experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e004_fixed_b0/runs/local-v2-review-001
```

不加载数组库、模型或图像。原脚本留作历史版本，见 [维护退役记录](../../../../history/README.md)。[本机测试](tests/test_fixed_b0_analysis_migration.py) 与 [迁移验收](reports/analysis-migration-20261022-v1.md) 已完成；此入口**不能处理新 GPU profile 的结果**。

### 新 GPU 入口：尚未远端验收

[run.py](run.py) 负责 prepare/run/worker/compare，[tensor_analysis.py](tensor_analysis.py) 负责统计。仅查看参数可在仓库根执行：

```bash
python -m experiment.mcmc_shard_imbalance.branches.b1_observation_reliability.experiments.e004_fixed_b0.run --help
```

| 身份／边界 | 值 |
|---|---|
| profile | mcmc_fixed_b0_five_repeats_workspace_v1 |
| worker unit | image3d-fixed-b0-workspace-v1.service |
| 输出 | 本实验 runs/<new-run-id>/，不允许旧输出根或覆盖已有 prepare 目录 |
| 真实环境 | i-94B8D131 的 /usr/local/3dgs_new/Image3D-SceneGraph |
| 远端前置 opt-in | IMAGE3D_AUTHORIZE_FIXED_B0_WORKSPACE_TESTS=1；旧变量不生效 |
| 授权 | 显式阶段授权及协议 SHA；worker 另需继承 GPU lease |
| 状态 | 本机 NoTorch 通过；远端前置、真实回放、同源数值验收均 not_run |

[legacy.py](legacy.py) 按 [gpu-source.json](gpu-source.json) 核验后复用旧来源验证和 snapshot/schema/RNG 能力；不是旧 main 的转发壳，但尚未完全解耦。原 scripts、测试及 core 不删，资源保护不变，新入口拒绝旧 v2 协议。

- [本机 GPU 迁移报告](reports/gpu-migration-local-20261022-v1.md)
- [NoTorch 测试](tests/test_fixed_b0_gpu_runner.py) / [远端前置测试](tests/test_fixed_b0_gpu_remote.py)

<a id="next"></a>
## 下一步：先验收迁移，不扩大科学范围

1. 明确提交范围、经 Git 提交后由用户手动同步；核对远端主机/目录、提交、新旧源码 SHA、历史来源及环境。源码没有同步或来源缺失时停止，不上传源码包绕过。
2. 单独授权新 opt-in 微型前置；检查当时 GPU 型号/数量/空闲、lease 和资源条件。前置结果单列，不能当作真实回放。
3. 前置通过后，按已授权范围 prepare 新目录/协议，核协议 SHA，再以唯一执行和只读监控运行五次回放及 compare；保留失败记录，不自动重试。
4. 对同源新旧统计做数值验收，分别记录执行、数据有效性、数值结果。当前没有新 prepare 协议或 GPU 任务，不因本机测试或历史 v2 通过而晋级。

以上是迁移验收顺序，不是执行授权。真正连通 B1→P2→P3、独立 A/A 基线和后续生命周期是 [另待冻结的科学研究](../../../../README.md#下一步与执行状态)，不得用本实验历史六点分析替代。
