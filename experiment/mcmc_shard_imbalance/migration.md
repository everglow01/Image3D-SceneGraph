# MCMC 迁移与保留边界

更新日期：2026-10-22。固定 B0 历史文本分析已验收，新 GPU 代码已迁入但远端验收未运行；其他 GPU runner 尚未迁移。

## 代码归属与当前状态

各实验 experiment.json 的 legacy_entrypoints、legacy_tests、legacy_analysis_scripts 登记原路径；[课题首页](README.md) 提供实验索引，这里不再复制 55 条路径表。indexed_legacy_not_moved 仅表示尚未迁移，不能推断文件无用。

- [E004 本地分析](branches/b1_observation_reliability/experiments/e004_fixed_b0/README.md)：替代两份旧 v2 分析的维护职责，只接受原 v2 文本；原文件仍受 analysis-source.json 绑定，保留原字节。
- [E004 GPU 入口](branches/b1_observation_reliability/experiments/e004_fixed_b0/run.py)：新 profile/模块/输出路径；不是转发旧 main。冻结桥仍绑定 13 个旧 scripts 和必要 core/helper/测试，共 32 文件，清单见 [gpu-source.json](branches/b1_observation_reliability/experiments/e004_fixed_b0/gpu-source.json)。尚未彻底解耦，旧依赖不可删。
- [历史归档](history/README.md)：72 条决策、24 节研究合同逐字保存；原章节跳转、冻结约束及历史验收记录保持。
- 本次收尾只删除合并到课题首页的 Q1/Q2/Q3/B1/B2/B3 六份导航 README，不删除运行代码、测试或证据。历史 C 批次的“删除 0 个”不改写。

## 删除与复现边界

只有无现役消费者、无冻结路径要求且替代已验收的文件才可退役。检查包括动态 import、subprocess 和协议 SHA；软链或 wrapper 不能冒充原身份。研究专用、历史失败、不常用均不是删除理由。

旧 outputs、Job、锁、远端服务、个人文档和用户原导航不动。根 CLAUDE.md 不强制入库。研究源码/审核文档可进入 Git，runs 等产物仍忽略；pytest 默认发现 tests 与 experiment。新远端验证和科学实验分别申请授权，步骤见 [E004](branches/b1_observation_reliability/experiments/e004_fixed_b0/README.md#next)。

## 产品共享模块


| 原文件 | 核实情况 | 处置 |
|---|---|---|
| [src/image3d_scenegraph/gaussian/absgrad_resources.py](../../src/image3d_scenegraph/gaussian/absgrad_resources.py) | trainer.py/evaluation.py/正式训练入口使用；冻结 core 成员 | production_shared_keep，不迁移、不改名 |
| [src/image3d_scenegraph/gaussian/absgrad_experiment.py](../../src/image3d_scenegraph/gaussian/absgrad_experiment.py) | AbsGrad 专用编排；当前属于另一个研究主题 | 登记为后续独立审查，不塞入 MCMC，不删除 |
| [src/image3d_scenegraph/gpu_lease.py](../../src/image3d_scenegraph/gpu_lease.py) | 产品与研究的 GPU 租约能力 | 保留并复用，不复制到 experiment |
| [src/image3d_scenegraph/file_integrity.py](../../src/image3d_scenegraph/file_integrity.py) | 共享完整性能力 | 保留并复用 |
| [scripts/run_gaussian_training.py](../../scripts/run_gaussian_training.py) | 正式训练 CLI | 留在 scripts，不改变行为 |

## 其他已有辅助资料

这些目录只登记位置，不新建虚构运行记录、不宣称其结论已在本轮重新核验。尤其早期前置资源失败不能与后续正式成功合并。2026-09 的 MCMC 产品观察属于前史，不冒充本次主课题的独立对照。

- [apartment-vs-num4-readonly-20261019-v1](../../outputs/analysis/apartment-vs-num4-readonly-20261019-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-capacity-implementation-20261017-v1](../../outputs/analysis/mcmc-capacity-implementation-20261017-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-degeneration-audit-20261017-v1](../../outputs/analysis/mcmc-degeneration-audit-20261017-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-degeneration-repair-20261017-v1](../../outputs/analysis/mcmc-degeneration-repair-20261017-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-final-fit-train-only-20260915](../../outputs/analysis/mcmc-final-fit-train-only-20260915)：`supporting_evidence_only`，保留原址。
- [mcmc-investigation-retrospective-20261022-v1](../../outputs/analysis/mcmc-investigation-retrospective-20261022-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-memory-repair-20261013-v2](../../outputs/analysis/mcmc-memory-repair-20261013-v2)：`supporting_evidence_only`，保留原址。
- [mcmc-multilayer-review-20261019-v1](../../outputs/analysis/mcmc-multilayer-review-20261019-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-native-low-view-attribution-20260911](../../outputs/analysis/mcmc-native-low-view-attribution-20260911)：`supporting_evidence_only`，保留原址。
- [mcmc-render-audit-20260910](../../outputs/analysis/mcmc-render-audit-20260910)：`supporting_evidence_only`，保留原址。
- [mcmc-stability-implementation-20261013-v1](../../outputs/analysis/mcmc-stability-implementation-20261013-v1)：`supporting_evidence_only`，保留原址。
- [mcmc-unlimited-host-memory-20261016-v3](../../outputs/analysis/mcmc-unlimited-host-memory-20261016-v3)：`supporting_evidence_only`，保留原址。
- [mcmc-validation-stall-20261013-v1](../../outputs/analysis/mcmc-validation-stall-20261013-v1)：`supporting_evidence_only`，保留原址。
