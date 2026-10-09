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
