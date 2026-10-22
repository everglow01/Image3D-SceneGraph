# Gaussian checkpoint and attempt contract

R2.5 defines the project-owned persistence boundary consumed by later Gaussian trainers and workers. It does not define a trainer serialization format: model, optimizer, scheduler, densification, and RNG state are mandatory opaque byte components at this layer.

## Layout

```text
outputs/jobs/{job_id}/
└── attempts/
    └── {attempt_id}/
        ├── attempt.json
        └── checkpoints/
            └── iteration_000001000/
                ├── checkpoint.json
                ├── model.bin
                ├── optimizer.bin
                ├── scheduler.bin
                ├── densification.bin
                ├── rng.bin
                └── metrics.json
```

Attempt IDs are limited to 1–64 ASCII letters, digits, `_`, and `-`, beginning with a letter or digit. Existing attempt and checkpoint destinations are immutable and are never overwritten.

## Attempt semantics

Attempt schema version 1 distinguishes three operations:

- `fresh`: the first attempt; it has no parent and loads no checkpoint.
- `retry`: a new attempt linked to an earlier attempt. It loads no state and starts from iteration zero. Dataset and effective-config hashes must match the parent, while code or environment may change to correct a failure.
- `resume`: a new attempt linked to a fully committed checkpoint from the parent attempt. Dataset, effective config, code, and environment hashes must all match exactly.

`attempt.json` records provenance, parent lineage, the resume checkpoint path/hash when applicable, and a hash of its own complete descriptor. R2.6 will add job/worker status separately; R2.5 does not add queued/running/failed fields to current geometry manifests.

## Checkpoint contents and provenance

Every checkpoint records:

- attempt ID and completed iteration;
- purpose: `periodic`, `best_validation`, or `final`;
- a finite validation score for `best_validation` only;
- R2.3 dataset hash;
- R2.4 effective-config hash;
- caller-supplied code and environment hashes;
- mandatory component path, byte count, and SHA-256;
- metric-history JSON;
- a hash over the complete metadata index.

The loader validates exact schema fields, fixed relative component paths, sizes, component hashes, metadata hash, attempt identity, provenance, and finite JSON metric values before returning state. It does not deserialize opaque model state and therefore does not execute checkpoint content.

R2.7 must serialize all RNG sources needed by its trainer, including Python, NumPy, Torch CPU, and CUDA state where used. It also owns the concrete Gaussian parameter and optimizer formats.

R2.7 concrete trainer checkpoints place the active topology-strategy state in the opaque densification component. `default_v1` stores gsplat `DefaultStrategy` accumulators; `mcmc_v1` stores its binomial relocation state. Both also store rank-local optimizer topology, camera order/cursor, and RNG. The effective-config hash binds the strategy identity and complete method package, so a Default checkpoint cannot resume as MCMC. Distributed checkpoints remain sharded and reject a changed world size; MCMC's global 3,000,000 cap is revalidated after merged candidate publication.

## Atomic publication

A checkpoint is built under a uniquely named hidden temporary directory in the final checkpoint parent. Each component and metadata file is flushed and `fsync`ed; metadata is written last. The temporary directory is `fsync`ed, renamed once to the final iteration directory on the same filesystem, and then its parent is `fsync`ed.

Loaders address only the final `iteration_NNNNNNNNN` directory. A crash before rename therefore leaves a hidden temporary directory that is not loadable; a crash after rename leaves the complete committed directory. Cleanup of stale temporary directories belongs to R2.6 restart recovery.

The contract assumes the job directory and its temporary checkpoint directory share one local filesystem. Stage 3 must define the equivalent database/object-storage transaction order separately.

## Retention

The trainer does not write periodic or best-Validation full checkpoints during a fresh run. Validation candidate selection uses one overwrite-in-place model snapshot without Adam or densification state, so candidate improvements do not accumulate process checkpoints.

A successful run atomically publishes one complete checkpoint at the final iteration, then removes any older committed checkpoint directories from that same attempt. Hidden temporary directories and unrelated files are never treated as committed checkpoints. This bounds completed-job checkpoint storage to one full terminal model/optimizer state rather than multiplying it by a training cadence. Cancellation before the final iteration retains progress diagnostics but no new full checkpoint; resume remains available only from an explicitly retained final parent checkpoint.

## Reproducibility boundary

The CPU reference test checkpoints a deterministic optimizer-like state machine and proves that resumed final state and metric history exactly match an uninterrupted run. This verifies persistence and RNG continuity, not real Gaussian/CUDA numerical equivalence.

R2.0 does not require bitwise equality across GPU or CUDA environments. R2.7 and R2.15 must add fixed-environment trainer evidence showing rendering metrics and Gaussian statistics remain within predeclared tolerances.


## 2026-10-09 streaming publication

The public schema-1 directory, mandatory opaque components, SHA checks, fsync ordering and atomic rename are unchanged. The writer additionally accepts regular non-symlink component files and copies/hashes them in bounded chunks. Exactly one of in-memory state and file sources is required. Source changes during copying reject publication.

New distributed trainer components use an `IMAGE3D_SHARDS_V1` prefix followed by an uncompressed ZIP container with `manifest.json` (version, world size, rank byte counts and SHA-256) and ordered `rank-N` members. This is an internal opaque-component format change, not byte-identical serialization; legacy Torch rank-shard containers remain readable. Rank count/identity and per-rank hashes are checked. No extraction to arbitrary paths is used.

Each rank stages its components on the same filesystem; collectives exchange only small status messages. Rank 0 packs components by streaming and invokes the existing atomic publisher. Failures never expose a committed partial checkpoint; hidden staging is not resumable. Successful publication removes only its own staging. Fresh cancellation no longer invokes full-state publication, matching the retention contract above; final checkpoints still retain all optimizer/topology/camera/RNG state. TrainingResult takes the returned checkpoint metadata instead of reloading the full checkpoint solely to read its hash.

## 2026-10-09 model snapshot I/O follow-up

`gaussian/model_io.py` now owns native model payload loading and file writes. Best-Validation snapshots, terminal single-rank snapshots, distributed model shards, merged models and checkpoint model components share this implementation. Production save paths serialize directly into a file, not a complete `BytesIO` buffer. The byte-based helpers remain only for legacy checkpoint decoding and equivalence diagnostics.

A snapshot is written to a unique temporary file in its destination directory, flushed and fsynced. Best-candidate replacement explicitly uses atomic replace; immutable outputs use no-clobber hard-link publication. The parent directory is fsynced and only the writer's own temporary file is removed. A serialization failure leaves the previous best candidate intact. This uses the existing local same-filesystem assumption; no object-storage semantics are implied. CPU Path loads use Torch mmap; this removes the redundant `read_bytes()` buffer but does not claim that constructing, validating or merging a model uses no additional CPU memory.

Public checkpoint schema, mandatory components, committed-directory atomic rename, rank-container compatibility and cancellation retention are unchanged. The implementation/core hash changes; old evidence is not relabeled as validation of this revision. Real model/RNG equivalence tests were added but have not been run in this local-only repair task.

## 2026-10-21 bounded research execution

An explicit native MCMC execution endpoint may precede the nominal configured schedule. Normal budget completion still publishes a complete `final` checkpoint at the actual update, never at the nominal future step. Its opaque scheduler adds `nominal_iterations` and `requested_stop_iteration`; without an explicit budget the historical iteration-only representation remains. Public schema and mandatory components are unchanged. Learning-rate scheduling still uses the unchanged effective configuration. A checkpoint is final for this execution budget, not proof that the nominal schedule completed. Candidate selection iteration remains distinct from checkpoint iteration. Camera-size overlay identity is hashed into environment provenance, preventing cross-renderer resume; source/core changes still require fresh matched evidence.


## 2026-10-21 observer resume research boundary (implementation only)

The lifecycle observer does not change any hashed training-core file, rewrite parent provenance, or waive code/environment matching. Each future branch copies the immutable source `attempt.json` and complete 12,000 checkpoint into a new workspace and creates a distinct `trace-001` resume attempt. Parent component bytes/hashes and lineage remain unchanged; there are no writable aliases into the old attempt. All three branches of an environment restore that same parent, not one another. A separate frozen observer manifest binds the extra executable observer/runner code and hooks: an unchanged core hash is not a claim that no additional code executes.

The bounded resume ends normally at 12,200 on the unchanged nominal 30,000 schedule. Its scheduler records the new execution endpoint while the parent retains its original 12,000 endpoint. All mandatory components and the atomic writer remain unchanged. Gate 1 compares canonical tensor/optimizer/strategy/camera/RNG contents and sampling order, not container bytes, timestamps or file paths; each checkpoint still independently passes normal byte-integrity verification. The existing native resume resets Validation candidate selection, so the new merged candidate represents only the resumed window, not the historical 3k best. No historical best-selection continuity is claimed or synthesized.

Actual native checkpoint restoration and CUDA observer equivalence are **not run** in this implementation phase. Local opaque-state/AST tests cannot promote those gates to passed. Research stage approval does not authorize Gate 3, an optimizer intervention, or a GPU run.


## 2026-10-21 one-step first-divergence probe (implementation only)

The old lifecycle Gate 1 A/A comparison failed after update 12,001; full observer equivalence and Gate 2 were not accepted. The independent T1 probe retains the same frozen native core and original gsplat environment, copies the same immutable 12k parent into each independent a/b workspace, and uses native resume without weakening provenance. Its separate protocol binds the probe executable/dependency hashes and stage definitions. T2/T3 are not automatically authorized or implemented.

P0 is captured after restoration but before camera sampling and forward, unlike the old first_render record. P0–P4 snapshots are synchronized CPU-owned arrays in non-pickle NPY files plus typed JSON manifests; they are diagnostic evidence, not native checkpoint components. Required tensors, scalar types, shapes, source layout and canonical/file hashes are verified. Adam state is named by the six Gaussian parameter groups and separately preserves moments, step and parameter groups. Copies never alias the parent checkpoint or live parameter storage.

The probe exits with a narrowly caught completion signal after P4, before the native final Validation/checkpoint/merge path. It does not publish a final checkpoint, model, best candidate or normal TrainingResult; probe_complete means exactly one diagnostic update and all five stages per rank were recorded. Other exceptions remain failures. Snapshot numerical equivalence with predeclared atol=2e-6/rtol=2e-4 does not change or retroactively pass the old strict Gate 1. P0 mismatch invalidates stage attribution, and a synchronized run that does not reproduce the difference does not establish original-run determinism. Local lightweight tests do not validate real restoration, CUDA staging or tensor equivalence; remote tests and T1 are not_run in this implementation task.


## 2026-10-22 backward-boundary probe

The level-one boundary probe uses the same immutable baseline 12k parent and the unchanged T1 native restoration path. Its separate profile binds the additional executable and B0/B1 schemas, leaving old protocol-bound files untouched. B0 saves all rasterizer backward context tensors, incoming image gradients and settings; B1 saves the returned gradients before communication. The worker and completion records bind both boundary manifests as well as the original P0–P4 records. These are CPU diagnostic snapshots, never replacement checkpoint components. Attribution requires bitwise-identical complete B0 inputs; tolerance-only input agreement is insufficient. No loss-only or fixed-input replay is automatically authorized, and no native final checkpoint or Validation candidate is produced.


## 2026-10-22 固定B0独立回放

新`mcmc_fixed_b0_five_repeats_v1`绑定原一级边界a臂B0及两臂历史P0/P1/P2/P3、完成/比较记录和全部快照SHA、包/core/脚本依赖、固定五次预算及预选六个owner/slot；旧协议及checkpoint不改、不复制、不反序列化训练状态。B0只重建原backward所需上下文和图像梯度，CPU→GPU回读核内容/布局身份；源文件变更、缺trial/rank、重复/重排、输入/RNG变化、输出别名或非有限均失败。每rank五份非pickle B1快照由worker→complete→comparison逐层绑定，附逐元素频次/range的NPY及SHA；它们不是checkpoint或新模型。

无forward/optimizer/MCMC/Validation，也无Test图像读取；准备只核既有元数据、快照及checkpoint文件完整性，不调用replay图像字节验证。选点先于真实回放冻结，历史Adam状态分析和本轮B1重复数据分开标识来源；不产生新的P2/P3，不宣称长期退化因果或全局数值误差上界。当前是实现合同，不表示远端前置或真实五次观测已经完成。


固定B0首轮实测补记（2026-10-22）：前置在首份微型B1快照后因Python RNG变化退出，未生成真实实验protocol或complete；单独存在的trial-0 manifest不是完成证据。不得以该快照补发replay_complete，不得将新尝试覆盖到此失败前置目录。失败报告与初始化副作用定位保留，真实B0/P2/P3及原checkpoint均未重算或改写。


固定B0 v2初始化边界（2026-10-22，仅实现）：冻结输入/五次预算不变，新增每rank的`initialization.json`，记录源协议所钉住的已安装二进制身份、显式导入前后Python/NumPy/TorchCPU/CUDA RNG摘要及变化列表、零forward/backward热身声明；worker以SHA绑定，完成校验要求首个trial RNG等于初始化后RNG，缺审计/篡改/边界不符拒绝。不得通过恢复RNG消除首次导入记录，不把旧v1单份微型B1升级成完成；修复后的远端验证仍未执行。
