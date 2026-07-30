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

## Atomic publication

A checkpoint is built under a uniquely named hidden temporary directory in the final checkpoint parent. Each component and metadata file is flushed and `fsync`ed; metadata is written last. The temporary directory is `fsync`ed, renamed once to the final iteration directory on the same filesystem, and then its parent is `fsync`ed.

Loaders address only the final `iteration_NNNNNNNNN` directory. A crash before rename therefore leaves a hidden temporary directory that is not loadable; a crash after rename leaves the complete committed directory. Cleanup of stale temporary directories belongs to R2.6 restart recovery.

The contract assumes the job directory and its temporary checkpoint directory share one local filesystem. Stage 3 must define the equivalent database/object-storage transaction order separately.

## Retention selection

R2.0 freezes algorithm-development retention at:

- latest three periodic checkpoints;
- best two validation checkpoints, ranked by score and then iteration;
- latest final checkpoint.

R2.5 returns this deterministic selection but deletes nothing. R2.6 may apply cleanup only after preserving the selected immutable checkpoints and diagnostics.

## Reproducibility boundary

The CPU reference test checkpoints a deterministic optimizer-like state machine and proves that resumed final state and metric history exactly match an uninterrupted run. This verifies persistence and RNG continuity, not real Gaussian/CUDA numerical equivalence.

R2.0 does not require bitwise equality across GPU or CUDA environments. R2.7 and R2.15 must add fixed-environment trainer evidence showing rendering metrics and Gaussian statistics remain within predeclared tolerances.
