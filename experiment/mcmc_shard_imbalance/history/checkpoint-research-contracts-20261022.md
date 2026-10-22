# Checkpoint 研究历史边界

以下 5 节来自 `docs/gaussian-checkpoint-contract.md` 的研究区段，按原顺序逐字归档。**迁移文档位置不取消冻结条款**；“仅实现”“待验证”等保留为原时点陈述，不能当作现行执行许可或把后续结果倒写入旧合同。

原路径仍以仓库根解释；现行问题见 [主课题](../README.md)，各实验 protocol 和 [归档清单](archive-manifest.json) 保留原身份。产品合同在原位置保留同名章节入口。

<a id="checkpoint-001"></a>
<!-- archive:checkpoint-001:begin -->
## 2026-10-21 bounded research execution

An explicit native MCMC execution endpoint may precede the nominal configured schedule. Normal budget completion still publishes a complete `final` checkpoint at the actual update, never at the nominal future step. Its opaque scheduler adds `nominal_iterations` and `requested_stop_iteration`; without an explicit budget the historical iteration-only representation remains. Public schema and mandatory components are unchanged. Learning-rate scheduling still uses the unchanged effective configuration. A checkpoint is final for this execution budget, not proof that the nominal schedule completed. Candidate selection iteration remains distinct from checkpoint iteration. Camera-size overlay identity is hashed into environment provenance, preventing cross-renderer resume; source/core changes still require fresh matched evidence.


<!-- archive:checkpoint-001:end -->

<a id="checkpoint-002"></a>
<!-- archive:checkpoint-002:begin -->
## 2026-10-21 observer resume research boundary (implementation only)

The lifecycle observer does not change any hashed training-core file, rewrite parent provenance, or waive code/environment matching. Each future branch copies the immutable source `attempt.json` and complete 12,000 checkpoint into a new workspace and creates a distinct `trace-001` resume attempt. Parent component bytes/hashes and lineage remain unchanged; there are no writable aliases into the old attempt. All three branches of an environment restore that same parent, not one another. A separate frozen observer manifest binds the extra executable observer/runner code and hooks: an unchanged core hash is not a claim that no additional code executes.

The bounded resume ends normally at 12,200 on the unchanged nominal 30,000 schedule. Its scheduler records the new execution endpoint while the parent retains its original 12,000 endpoint. All mandatory components and the atomic writer remain unchanged. Gate 1 compares canonical tensor/optimizer/strategy/camera/RNG contents and sampling order, not container bytes, timestamps or file paths; each checkpoint still independently passes normal byte-integrity verification. The existing native resume resets Validation candidate selection, so the new merged candidate represents only the resumed window, not the historical 3k best. No historical best-selection continuity is claimed or synthesized.

Actual native checkpoint restoration and CUDA observer equivalence are **not run** in this implementation phase. Local opaque-state/AST tests cannot promote those gates to passed. Research stage approval does not authorize Gate 3, an optimizer intervention, or a GPU run.


<!-- archive:checkpoint-002:end -->

<a id="checkpoint-003"></a>
<!-- archive:checkpoint-003:begin -->
## 2026-10-21 one-step first-divergence probe (implementation only)

The old lifecycle Gate 1 A/A comparison failed after update 12,001; full observer equivalence and Gate 2 were not accepted. The independent T1 probe retains the same frozen native core and original gsplat environment, copies the same immutable 12k parent into each independent a/b workspace, and uses native resume without weakening provenance. Its separate protocol binds the probe executable/dependency hashes and stage definitions. T2/T3 are not automatically authorized or implemented.

P0 is captured after restoration but before camera sampling and forward, unlike the old first_render record. P0–P4 snapshots are synchronized CPU-owned arrays in non-pickle NPY files plus typed JSON manifests; they are diagnostic evidence, not native checkpoint components. Required tensors, scalar types, shapes, source layout and canonical/file hashes are verified. Adam state is named by the six Gaussian parameter groups and separately preserves moments, step and parameter groups. Copies never alias the parent checkpoint or live parameter storage.

The probe exits with a narrowly caught completion signal after P4, before the native final Validation/checkpoint/merge path. It does not publish a final checkpoint, model, best candidate or normal TrainingResult; probe_complete means exactly one diagnostic update and all five stages per rank were recorded. Other exceptions remain failures. Snapshot numerical equivalence with predeclared atol=2e-6/rtol=2e-4 does not change or retroactively pass the old strict Gate 1. P0 mismatch invalidates stage attribution, and a synchronized run that does not reproduce the difference does not establish original-run determinism. Local lightweight tests do not validate real restoration, CUDA staging or tensor equivalence; remote tests and T1 are not_run in this implementation task.


<!-- archive:checkpoint-003:end -->

<a id="checkpoint-004"></a>
<!-- archive:checkpoint-004:begin -->
## 2026-10-22 backward-boundary probe

The level-one boundary probe uses the same immutable baseline 12k parent and the unchanged T1 native restoration path. Its separate profile binds the additional executable and B0/B1 schemas, leaving old protocol-bound files untouched. B0 saves all rasterizer backward context tensors, incoming image gradients and settings; B1 saves the returned gradients before communication. The worker and completion records bind both boundary manifests as well as the original P0–P4 records. These are CPU diagnostic snapshots, never replacement checkpoint components. Attribution requires bitwise-identical complete B0 inputs; tolerance-only input agreement is insufficient. No loss-only or fixed-input replay is automatically authorized, and no native final checkpoint or Validation candidate is produced.


<!-- archive:checkpoint-004:end -->

<a id="checkpoint-005"></a>
<!-- archive:checkpoint-005:begin -->
## 2026-10-22 固定B0独立回放

新`mcmc_fixed_b0_five_repeats_v1`绑定原一级边界a臂B0及两臂历史P0/P1/P2/P3、完成/比较记录和全部快照SHA、包/core/脚本依赖、固定五次预算及预选六个owner/slot；旧协议及checkpoint不改、不复制、不反序列化训练状态。B0只重建原backward所需上下文和图像梯度，CPU→GPU回读核内容/布局身份；源文件变更、缺trial/rank、重复/重排、输入/RNG变化、输出别名或非有限均失败。每rank五份非pickle B1快照由worker→complete→comparison逐层绑定，附逐元素频次/range的NPY及SHA；它们不是checkpoint或新模型。

无forward/optimizer/MCMC/Validation，也无Test图像读取；准备只核既有元数据、快照及checkpoint文件完整性，不调用replay图像字节验证。选点先于真实回放冻结，历史Adam状态分析和本轮B1重复数据分开标识来源；不产生新的P2/P3，不宣称长期退化因果或全局数值误差上界。当前是实现合同，不表示远端前置或真实五次观测已经完成。


固定B0首轮实测补记（2026-10-22）：前置在首份微型B1快照后因Python RNG变化退出，未生成真实实验protocol或complete；单独存在的trial-0 manifest不是完成证据。不得以该快照补发replay_complete，不得将新尝试覆盖到此失败前置目录。失败报告与初始化副作用定位保留，真实B0/P2/P3及原checkpoint均未重算或改写。


固定B0 v2初始化边界（2026-10-22，仅实现）：冻结输入/五次预算不变，新增每rank的`initialization.json`，记录源协议所钉住的已安装二进制身份、显式导入前后Python/NumPy/TorchCPU/CUDA RNG摘要及变化列表、零forward/backward热身声明；worker以SHA绑定，完成校验要求首个trial RNG等于初始化后RNG，缺审计/篡改/边界不符拒绝。不得通过恢复RNG消除首次导入记录，不把旧v1单份微型B1升级成完成；修复后的远端验证仍未执行。


固定B0 v2结果补记（2026-10-22）：独立协议ff6e21e7…的两个worker均完成五次、初始化审计及全部trial绑定有效；真实10次原backward与前置10次分开记账，无额外热身/训练更新。源输入、包、回放快照file/canonical SHA及32份频次/range数组已独立复核，未生成新checkpoint/model/P2/P3；历史源状态仍只读。实测数值容差内不是旧严格Gate通过或全局有界证明，v1失败目录和历史记录保留。
<!-- archive:checkpoint-005:end -->
