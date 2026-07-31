# Manifest schema

`outputs/jobs/{job_id}/manifest.json` is the stable interface between the backend and frontend. Paths in the manifest are relative to the job directory; consumers must not depend on model-internal directories that are not listed here.

## Top-level fields

Newly completed jobs contain:

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Local job identifier. |
| `status` | string | Current terminal value is `done`. |
| `stage` | string | Last completed pipeline stage. |
| `progress` | number | Completion fraction. |
| `mode` | string | Requested input mode. |
| `input_type` | string | Normalized input type; panoramas use `equirectangular_panorama`. |
| `geometry_backend` | string | Effective geometry backend. |
| `output_type` | string | Requested geometry asset type. |
| `created_at` | string | UTC creation timestamp. |
| `inputs` | array | Stored input filename, relative path, content type, and byte count. |
| `assets` | object | Available output assets keyed by stable role. |
| `metrics` | object | Backend and postprocess diagnostics suitable for display and audit. |
| `gaussian_config` | object | Optional resolved 3DGS configuration provenance for jobs that have explicitly entered the project-owned Gaussian lifecycle. |
| `mesh_variants` | array | Optional generated mesh alternatives. |

Historical manifests can omit fields added after that job was created. Readers must tolerate absent optional fields and must not manufacture an effective policy that the old job did not record.

## Local lifecycle schema v1

R2.6 submissions add `lifecycle_schema_version: 1` and are atomically persisted before execution. Their status transitions are:

```text
queued -> running -> done
                  -> failed
       -> cancelled
failed/cancelled -> queued (new bounded retry attempt)
```

Lifecycle manifests additionally record `updated_at`, `queued_at`, nullable `started_at`/`completed_at`/`cancel_requested_at`, `active_attempt_id`, an `attempts` history, and nullable structured `error: {code, message}`. Attempt history records immutable IDs, `fresh` or `retry` kind, parent attempt, timestamps, status, and error. A retry starts clean, never loads checkpoint state, and is capped at three total attempts.

`POST /api/jobs` returns HTTP 202 with the durable `queued` manifest. One local filesystem worker processes jobs FIFO and holds a filesystem lease, so only one GPU-heavy job runs for an output root. Queued work remains queued across API restart. A stale `running`/`exporting` job discovered at worker startup becomes explicit `failed` with `worker_interrupted` (or `cancelled` when a cancellation request was already durable); it is never inferred to be successful.

A running attempt writes under `lifecycle/attempts/{attempt_id}/workspace`. Complete outputs are moved to stable job paths before the terminal `done` manifest is published. Failed/cancelled partial output is retained under the attempt for diagnostics but is not listed in `assets`. Inputs and completed diagnostics are not deleted by cancellation. R2.6 attempt history is lifecycle metadata; R2.5 checkpoint descriptors remain separate and are created only when a trainer has real dataset/config/code/environment provenance.

Legacy terminal-only manifests remain valid and readable. They do not gain synthetic attempt history and cannot use the R2.6 retry operation.

## Gaussian effective configuration

`gaussian_config` is present only when a trusted project-owned 3DGS caller supplies a resolved configuration. It contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Gaussian configuration schema version; trainer topology repair defines version `2`. |
| `requested_profile` | string | Versioned profile selected before resolution; the only public profile is currently `standard_v1`. |
| `effective_config` | object | Complete validated training configuration after trusted internal overrides. It is authoritative over request fields or environment variables. |
| `effective_config_hash` | string | SHA-256 of `effective_config` serialized as sorted, compact JSON. The requested profile is provenance and is not part of this hash. |

The same schema version, requested profile, effective configuration, and hash are written to `logs/run.log`. Public job submission does not expose raw Gaussian hyperparameters; a future Gaussian-specific public route may select only an allow-listed quality profile. Internal research callers may supply validated overrides, and a recorded ablation must differ from its baseline in exactly one effective leaf field.

Training configuration contains validation cadence but no test cadence. Schema v2 adds screen-normalized densification, duplicate/split controls, and explicitly disabled-by-default normalized screen-size pruning. Held-out test views remain isolated until the candidate, effective configuration, and model hashes are frozen. `standard_v1` remains the only public profile. The measured `rtx4060_8gb_development_v1` is internal-only and exists for reproducible Stage 2D development/smoke runs; it is not a public quality promise.

Jobs without `gaussian_config`, including all historical geometry jobs and imported-splat fixtures, remain valid. Readers must not add a default profile or infer effective parameters for them.

## Assets

Common asset roles include:

- `point_cloud`, `point_cloud_aligned`, and `cameras`
- `alignment_diagnostics`, `fusion_diagnostics`, `visibility_graph`, `scale_disagreement_diagnostics`, and `consistency_diagnostics`
- `mesh` and `mesh_diagnostics`
- `scene_splat`, `scene_graph`, and `log`
- `gaussian_model`, `gaussian_training_result`, `gaussian_progress`, and `gaussian_dataset` for complete project-owned training jobs
- `gaussian_evaluation`, `gaussian_test_evaluation`, `gaussian_test_decision`, `gaussian_export_metadata`, `gaussian_canonical`, `gaussian_camera_path`, and `gaussian_bundle` for complete Stage 2D delivery

`scene_splat` is the versioned browser derivative; `gaussian_canonical` is the project-owned deterministic PLY contract. The Gaussian dataset/evaluation/export roles are per-attempt hash-bound records. Incomplete or failed training/evaluation/export files must not be added to `assets`.

Only roles present in `assets` are available. Generic postprocessing can add existing alignment or mesh assets when an older manifest is loaded; this does not rerun reconstruction.

## COLMAP+VGGT effective-policy metrics

A completed `colmap_vggt` job records effective runtime settings parsed from `logs/run.log`. Relevant fields include:

| Metric | Stable G1.26 fallback | Notes |
| --- | --- | --- |
| `fusion_mode` | `points` | `tsdf` remains experimental. |
| `vggt_batch_size` | `4` | Hardware-sensitive. |
| `vggt_grouping` | `sequential` | `covisibility` remains experimental. |
| `vggt_overlap_size` | `2` | Requested overlap. |
| `overlap_size` | `0` for sequential grouping | Effective consecutive overlap; may differ from the request. |
| `conf_percentile` | `50.0` | Percentile, not a metric confidence guarantee. |
| `confidence_threshold_scope` | `global` | `per_frame` remains experimental. |
| `consistency_support_policy` | `any_support` | `adaptive_two` remains experimental. |
| `max_points` | `2000000` | Final point-cloud cap only. |
| `point_budget_policy` | `random` | `spatial_balanced` remains experimental. |
| `point_budget_input_points` | run-dependent | Accepted points before the cap. |
| `point_budget_output_points` | run-dependent | Points written after the cap. |
| `point_budget_applied` | run-dependent | Whether point-budget policy actually ran. |

API form fields are optional. When omitted, the adapter resolves each setting from its environment variable and then its stable fallback. Therefore manifest metrics and runner diagnostics—not a request form alone—are authoritative for effective behavior. Historical jobs may lack some or all policy metrics and remain valid.

## Coordinates and scale

Raw geometry uses the reconstruction backend's source coordinate system. An aligned point cloud is a separate asset with its transform recorded in alignment diagnostics. Viewer axis flips are display-only.

COLMAP+VGGT world units are arbitrary unless an independently evaluated scale source is present. The manifest and ETH3D's evaluation-time camera Sim(3) alignment do not establish true metres or metric accuracy.
