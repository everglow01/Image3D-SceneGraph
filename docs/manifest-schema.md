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
| `inputs` | array | Stored input filename, relative path, content type, byte count, and SHA-256. |
| `assets` | object | Available output assets keyed by stable role. |
| `metrics` | object | Backend and postprocess diagnostics suitable for display and audit. |
| `gaussian_config` | object | Optional resolved 3DGS configuration provenance for jobs that have explicitly entered the project-owned Gaussian lifecycle. |
| `gaussian_trainer` | object | Optional selected trainer identity, label, pinned revision, and license. Current IDs are `graphdeco`, `project`, and experimental native `mcmc`; historical manifests may omit it. |
| `gaussian_geometry_source` | string | Optional requested Gaussian geometry source: stable `colmap` or experimental video-only `vggt_ba`. Historical Gaussian jobs may omit it. |
| `gaussian_geometry_effective_source` | string or null | Geometry actually used by a completed Gaussian job: `colmap` or `vggt_ba`; queued jobs use `null`. Historical jobs may omit it rather than receiving a synthetic value. |
| `gaussian_geometry_fallback_applied` | boolean | Whether a requested `vggt_ba` job completed using the explicit ordinary-COLMAP fallback. |
| `gaussian_geometry_fallback_reason` | string or null | One allow-listed geometry-quality reason when fallback occurred; otherwise `null`. |
| `gaussian_postprocess` | string | Optional requested Gaussian derivative: `none` or experimental `vggt_visibility_v1`. |
| `gaussian_postprocess_status` | string | Optional derivative lifecycle: `pending`, `not_requested`, `available`, or `unavailable`. Failure is fail-soft and does not invalidate Original Gaussian assets. |
| `gaussian_postprocess_reason` | string or null | Optional diagnostic reason when the requested derivative is unavailable. |
| `gaussian_sor_filter` | string | Optional SOR floater-cleanup request for project Gaussian jobs: `on` (default) or `off`. Absent on historical manifests, which never ran the filter. |
| `gaussian_sor_filter_status` | string | Optional SOR lifecycle: `pending`, `disabled`, `available`, or `unavailable`. Failure is fail-soft and the unfiltered model remains a successful result. |
| `gaussian_sor_filter_reason` | string or null | Optional diagnostic reason when the requested SOR cleanup is unavailable. |
| `gaussian_recovery_prune` | string | Optional experimental recovery-gated prune request for project Gaussian jobs: `on` or `off` (default). When `on`, the effective Gaussian config enables `opacity_reset.recovery_prune`. Absent on historical manifests. |
| `navigation_status` | string | Optional Gaussian-navigation lifecycle: `pending`, `not_generated`, `queued`, `generating`, `available`, or `unavailable`. Navigation failure never changes a completed Gaussian job from `done`. |
| `navigation_reason` | string or null | Stable machine-readable reason when navigation is unavailable. |
| `navigation_details` | object or null | Hashes, budgets, coordinate semantics, and Train-only publication evidence for an available navigation asset set. |
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
| `schema_version` | integer | Gaussian configuration schema version; new Project/MCMC jobs use version `10`, while historical versions remain immutable evidence. |
| `requested_profile` | string | Versioned profile selected before resolution; `standard_v1` is public and `mcmc_v1` is selected only through the trusted `gaussian_trainer=mcmc` method identity. |
| `effective_config` | object | Complete validated training configuration after trusted internal overrides. It is authoritative over request fields or environment variables. |
| `effective_config_hash` | string | SHA-256 of `effective_config` serialized as sorted, compact JSON. The requested profile is provenance and is not part of this hash. |

The same schema version, requested profile, effective configuration, and hash are written to `logs/run.log`. Public job submission does not expose raw Gaussian hyperparameters; a future Gaussian-specific public route may select only an allow-listed quality profile. Internal research callers may supply validated overrides, and a recorded ablation must differ from its baseline in exactly one effective leaf field.

Training configuration contains Validation cadence but no Test cadence. Schema v10 adds an explicit strategy, method-specific initialization/loss/LR fields, and an optional global Gaussian cap while preserving all `standard_v1/default_v1` numerical behavior. `mcmc_v1` is a frozen experimental method package: gsplat `MCMCStrategy`, 30,000 iterations, initial opacity `0.5`, frozen 3NN scale multiplied by `0.1`, opacity/scale regularization `0.01`, position-LR delay `0.01`, opacity LR `0.05`, relocation/growth from iteration 500 through 25,000 every 100 iterations, and a 3,000,000-Gaussian cap across all ranks. It disables Default pruning, opacity reset, and recovery-prune; requesting `gaussian_recovery_prune=on` with `mcmc` is rejected. Raw MCMC tuning leaves are not public form controls. New Project and MCMC frontend jobs export the Validation-selected model without loading Test; `gaussian_test_evaluation` and `gaussian_test_decision` remain absent unless a separately authorized frozen-candidate evaluation produces them. MCMC is runnable but remains experimental pending remote quality/resource evidence; Graphdeco remains the default.

Jobs without `gaussian_config`, including all historical geometry jobs and imported-splat fixtures, remain valid. Readers must not add a default profile or infer effective parameters for them.

## Assets

Common asset roles include:

- `point_cloud`, `point_cloud_aligned`, and `cameras`
- `sfm_sparse_point_cloud` and `sfm_diagnostics` for Project Gaussian jobs that publish the final accepted COLMAP sparse PLY/cameras and frontend feature/match diagnostics
- `alignment_diagnostics`, `fusion_diagnostics`, `visibility_graph`, `scale_disagreement_diagnostics`, and `consistency_diagnostics`
- `mesh` and `mesh_diagnostics`
- `scene_splat`, `scene_graph`, and `log`
- `gaussian_raw_model`, `gaussian_model`, `gaussian_training_result`, `gaussian_progress`, and `gaussian_dataset` for complete project-owned training jobs; the raw role is the immutable Validation-selected trainer output before SOR, while `gaussian_model` is the model passed to common evaluation/export
- `gaussian_replay_dataset` and `gaussian_replay_record` for a self-contained frozen trainer input rooted at `gaussian/replay/`; it preserves dataset/image/camera/initialization hashes and includes the registered undistorted images without publishing the COLMAP database or matches
- `gaussian_evaluation`, `gaussian_export_metadata`, `gaussian_canonical`, `gaussian_camera_path`, and `gaussian_bundle` for complete Stage 2D delivery; optional `gaussian_test_evaluation` and `gaussian_test_decision` appear only after an authorized frozen-candidate Test evaluation
- `collision_mesh`, `navigation`, and `navigation_diagnostics` for a complete Train-only first-person navigation set
- `video_probe`, `video_frame_selection`, `video_keyframe_timing`, `video_registration_diagnostics`, and `video_keyframe_contact_sheet` for a completed bounded video attempt; explicit ordinary-COLMAP `standard_v2` attempts additionally publish `video_initial_registration_expansion`, `video_registration_recovery`, and `colmap_timing`
- `vggt_ba_diagnostics` and `vggt_ba_window_graph` for a completed experimental VGGT-BA attempt, including one that used the explicit COLMAP fallback; `vggt_ba_initialization_diagnostics` appears only when VGGT-BA remained the effective source
- `gaussian_vggt_filtered_model`, `gaussian_vggt_filter_diagnostics`, `gaussian_vggt_filter_mask`, `gaussian_vggt_filtered_evaluation`, `gaussian_vggt_filtered_export_metadata`, `gaussian_vggt_filtered_canonical`, `scene_splat_vggt_filtered`, and `gaussian_vggt_filtered_bundle` for a complete experimental filtered derivative

`collision_mesh` is a low-poly invisible local-physics GLB, not the customer-visible `mesh` or `scene_splat`. `navigation` is the versioned normalized-coordinate boundary/spawn/player contract; `navigation_diagnostics` is its Train-only quality record. The three roles are published together only after source path containment, source/model/config/export hashes, split isolation, schema, topology, triangle/size/time budgets, and GLB integrity pass. Validation/Test IDs must be empty and selected render IDs must be a subset of Train.

New Gaussian jobs attempt navigation after export. A failed attempt records `navigation_status: unavailable` and a stable reason while the Gaussian job remains `status: done` with `scene_splat`. `POST /api/jobs/{job_id}/navigation-assets` queues generation for an old successful Gaussian job, is idempotent while queued/generating/available, never retrains, and runs through the same single-worker filesystem lease. Generation writes under `lifecycle/navigation/attempt-NNN/workspace`; one directory rename publishes the complete set, while cancellation/interruption preserves partial output outside stable asset paths. A retry increments `navigation_attempt` and does not overwrite a published set.

Navigation assets and their navigation lifecycle diagnostics are excluded from the existing job download ZIP. The ZIP contains a sanitized manifest without navigation roles/status so it cannot reference omitted files. This preserves the existing delivery contract until explicitly revised; direct manifest asset URLs remain the product path.

The desktop Gaussian viewer enables Walk only when `scene_splat`, `collision_mesh`, and `navigation` are all present and the navigation set passes client-side schema, Train-only, normalized/arbitrary-unit, range, byte-count, SHA-256, triangle-count, and spawn checks. It uses the collision GLB only to construct a Three.js `Octree`; the mesh is hidden by default and the Gaussian splat remains the customer-visible scene. Missing, generating, unavailable, or invalid navigation leaves Orbit mode usable. Completed old Gaussian jobs expose the idempotent generation action in the frontend, while Walk provides Pointer Lock, WASD/arrows, hard-boundary enforcement, safe-position preservation, and spawn reset. Player dimensions and speed are relative to estimated eye height `H`, not metres.

`scene_splat` is the versioned browser derivative; `gaussian_canonical` is the project-owned deterministic PLY contract. The Gaussian dataset/evaluation/export roles are per-attempt hash-bound records. Incomplete or failed training/evaluation/export files must not be added to `assets`.

Only roles present in `assets` are available. Generic postprocessing can add existing alignment or mesh assets when an older manifest is loaded; this does not rerun reconstruction.

## SfM frontend diagnostics

New `project_3dgs + gaussian_splat` jobs publish the final accepted sparse model as `sfm_sparse_point_cloud` (`geometry/points.ply`) and its camera source as `cameras`. The dedicated role keeps raw `colmap_world` arbitrary-unit geometry independently addressable, while generic postprocessing also aligns that same sparse cloud into `point_cloud_aligned` with `alignment_diagnostics`; the frontend defaults to aligned geometry but preserves Raw/Aligned and display-axis controls. Because a fitted plane normal has an ambiguous sign, viewers use transformed camera-up evidence to choose the initial ±Z display orientation. Nearest-input-view records are separately transformed into the Gaussian normalized frame and never claim metric scale.

After the final Mapper/registration/recovery/BA result and dataset split are frozen, the adapter attempts a fail-soft frontend export from that same workspace's COLMAP database. Success publishes `sfm_diagnostics=diagnostics/sfm/manifest.json` and `sfm_diagnostics_status=available`; failure omits the role, records `sfm_diagnostics_status=unavailable` plus `sfm_diagnostics_reason`, and does not fail Gaussian training. Data from failed/replaced `lifecycle/partial` attempts is never combined with the accepted model. Historical manifests may omit all these roles and metrics.

`sfm_diagnostics` schema 1 (`profile=sfm_frontend_diagnostics_v1`) records:

- normalized OpenCV camera axes and arbitrary-unit semantics;
- dataset hash, default run ID, one or more detector/matcher run records, and aggregate image/keypoint/pair/match/inlier counts;
- every feature-extracted source image's stable frame UID, job-relative path/hash/dimensions, COLMAP image ID, optional source video time, final registration/split state, and—only when registered—normalized center/forward/up and horizontal/vertical FOV;
- per-run detector/matcher/geometric-verification provenance plus job-relative feature-index and pair-index paths.

Large run children are deterministic gzip-compressed JSON shards. Their filenames end in `.json.gz`; the asset endpoint returns them as `application/json` with `Content-Encoding: gzip`, so browser consumers use ordinary JSON decoding. Feature records contain only rounded `(x,y)` coordinates in original upright feature-input pixels. Pair records store each tentative correspondence exactly once, partitioned into geometrically verified `inliers` and rejected `outliers`; a pair missing from the index means it was not tested, while an indexed pair with zero counts was tested with no correspondences. COLMAP descriptors, the mutable database, and failed-attempt internals are not published. Pixel mosaic stitching and final SfM 3D-track observations are outside schema 1.

Metrics on success include `sfm_diagnostics_image_count`, `sfm_diagnostics_registered_image_count`, `sfm_diagnostics_keypoint_count`, `sfm_diagnostics_pair_count`, `sfm_diagnostics_match_count`, `sfm_diagnostics_inlier_count`, and `sfm_diagnostics_bytes`.

## Bounded video contract

`mode=video` is implemented only with `project_3dgs + gaussian_splat`. The API accepts `video_keyframe_profile=standard_v1|standard_v2` and temporarily defaults to v1; the frontend continues to submit v1 until the v2 geometry gates pass. Both profiles accept one MP4/MOV/M4V/WebM, 10–600 seconds (606 seconds is the technical container-tolerance limit), at most 2 GiB, and analyze no more than 3,636 candidates at 6 fps. Historical v1 selects at most 1,000 keyframes. Explicit v2 selection schema 2 records an immutable 4 fps uniform base, deterministic sparse Lucas–Kanade motion/descriptor fallback, at most one adaptive frame per second up to 5 fps, stable candidate-index/source-PTS filenames, and base/adaptive selection reasons and counts. Upload staging uses bounded chunks; the original video remains under `input/` and retry deterministically regenerates keyframes from it.

`colmap_matcher` defaults to `exhaustive`. The experimental video-only `sequential` option applies to both ordinary COLMAP and `vggt_ba` geometry, uses COLMAP sequential matching with vocab-tree loop detection, requires the pinned local vocab tree, and fails before geometry when the tree is unavailable. For v2, sequential overlap is `clamp(ceil(effective_fps × 4), 16, 24)` so initial matching covers approximately four seconds. To bound large-sequence Mapper cost without changing v1 replay, explicit v2 raises intermediate global-BA growth triggers to 1.5, raises frame/point frequencies to 1,000/1,000,000, and limits each global refinement cycle to one; final BA remains enabled. VGGT-BA seeded registration and any classified ordinary-COLMAP fallback reuse the same feature database, matching policy, and v2 Mapper policy.

`video_probe` records the normalized ffprobe result, source hash, source/display dimensions, and applied quarter-turn without exposing location values. `video_frame_selection` is the authoritative final source-PTS, quality, rejection, output hash, dimensions, minimal generated-EXIF, and selection-reason record; accepted v2 recovery frames are atomically added to this same sidecar before final undistortion and dataset construction. `video_keyframe_timing` records the complete probe/analyze/materialize elapsed time and profile; `video_keyframe_contact_sheet` is display-only. For ordinary-COLMAP v2, Mapper starts from at most 1,000 uniformly spaced `base` frames while feature extraction and matching still cover the complete selected set. `video_initial_registration_expansion` records up to two subsequent selected-frame `image_registrator` + non-clearing `point_triangulator` passes, including registered-camera and sparse-point retention. `colmap_timing` records ordinary-COLMAP feature extraction, matching, seed mapping, initial registration expansion, gap recovery, undistortion, conversions, seed count, frozen v2 Mapper options, and total elapsed time. `video_registration_diagnostics` maps the final selected source timestamps to final COLMAP registrations and records registration rate, temporal coverage, largest gap, and `gap_violations`—adjacent registered-frame intervals exceeding `maximum_registered_gap_threshold_seconds` (2 seconds). Gap violations are a soft warning: the job continues, and when present the manifest publishes their count/total/excess and `run.log` records the intervals.

For explicit v2, `video_registration_recovery` is schema 1 and records the frozen incremental-COLMAP policy, initial/final selected and registered counts, registered-camera retention, sparse-point counts, pre/post gap count/total/excess, up to two registration rounds, candidates/materialized frames/local pair counts, per-substage elapsed time, acceptance/rejection reasons, one final-BA record, and fail-soft status. Each round adds at most 25% of the initial selection and all rounds add at most 50%; only candidates in a gap plus two-second bridge margins are eligible. The first round extracts/matches only new local evidence. Every round runs `image_registrator`; after an accepted triangulation, the second round may run in `propagation` mode with zero new candidates so the new 3D points can register frames deeper inside the same gap. A round proceeds to non-clearing triangulation only when registration strictly improves gap count, maximum gap, or total gap excess; registering only out-of-gap cameras is not sufficient. Accepted rounds lose no existing registered camera, retain at least 90% of sparse points, and continue to pass the 12/70%/80% product gates. Initial registration expansion and accepted recovery rounds share one final bundle adjustment, preferring CUDA and retrying once on CPU when CUDA BA is unavailable. Recovery command/model failure preserves the previously accepted model and selection and continues to final soft-gap diagnostics.

Completed v2 metrics include `video_keyframe_elapsed_seconds`, `colmap_geometry_elapsed_seconds`, the serialized COLMAP stage timings, initial-registration expansion status/pass count/registered gain, `video_initial_selected_count`, `video_base_selected_count`, `video_adaptive_selected_count`, `video_recovery_selected_count`, recovery status/reason/attempted and accepted round counts, pair/time totals, final-BA timing/fallback state, registered gain, and pre/post timeline and sparse-point summaries. Existing `video_selected_count` and `video_registration_*` fields always describe the final geometry input and final model. Historical jobs and v1 jobs may omit v2-only fields.

`scripts/evaluate_video_v2_promotion.py` is the frozen geometry promotion evaluator. It requires same-source v1 baseline timing, two independent v2 selection records, v2 initial-expansion/timing/recovery diagnostics, and exits nonzero unless selector evidence is identical, registration is at least 95%, every registered gap is at most two seconds, no Mapper-registered or expansion-registered camera is lost, at least 90% of the Mapper-input sparse points survive the complete expansion/recovery pipeline, recovery stays within two rounds/50%, and extraction plus COLMAP time is at most twice the v1 baseline. These are promotion gates, not product hard-failure semantics.

Generated keyframe JPEGs are physically upright and use EXIF Orientation `1` plus a Software tag. Container creation time/device tags are not fabricated as per-frame photographic EXIF; focal length, ISO, shutter, GPS, and original-photo timestamps are not invented. Estimated camera parameters remain in `geometry/cameras.json`. COLMAP must register at least 12 selected frames, at least 70% of selected frames, and at least 80% of their selected temporal span before Gaussian training starts.

Registered video frames are assigned to deterministic two-second temporal groups. A group belongs wholly to Train, Validation, or Test; the trainer still loads only Train and Validation. Held-out groups are never chosen from the groups adjacent to a registration gap above the 2-second threshold, so holdouts do not compound registration holes; if that avoidance leaves too few eligible groups, dataset contract construction fails. Video extraction, registration gates, navigation, and model selection cannot load Test, and video ingestion does not create a `*.test-consumed.json` record.

This contract is bounded offline reconstruction, not realtime SLAM, guaranteed loop closure, metric scale, or a drift-free long-horizon claim. Missing FFmpeg/ffprobe marks only Project video ingestion unavailable; image jobs remain available.

## Experimental Gaussian geometry and filtered derivative

New `project_3dgs + gaussian_splat` requests may independently select:

```text
gaussian_geometry_source = colmap | vggt_ba
gaussian_postprocess = none | vggt_visibility_v1
```

Both request fields default to the historical behavior (`colmap` and `none`). `vggt_ba` is video-only and research-only. Its local windows classify a frame as strong at 32 or more reliable observations; weak frames do not enter local BA or Sim(3) edge evidence. Each adjacent disconnect receives at most one deterministic recovery window of no more than eight frames, with at least three reliable frames from each side. Existing bounded DINOv2 nonlocal bridge windows remain independent.

After recovery, only a connected component with at least 12 reliable cameras, at least 70% reliable-camera coverage, at least 80% index-span temporal coverage, finite Sim(3), and non-worsening pose-graph optimization may seed geometry. The seeded path is:

```text
partial VGGT camera model
-> COLMAP point_triangulator
-> COLMAP image_registrator
-> COLMAP bundle_adjuster
-> final registration gates
```

SIFT extraction and the selected exhaustive/sequential matching policy run once for all selected keyframes. Ordinary COLMAP Mapper may reuse that database and those matches only for these three classified quality states:

```text
vggt_graph_unusable_after_recovery
vggt_seed_geometry_insufficient
vggt_registration_gate_failed
```

There is no broad exception fallback. OOM/CUDA, dependency or checkpoint, non-finite, corrupted-input, I/O, cancellation, unexpected-code, and unclassified subprocess failures remain failed Jobs. A fallback model must pass the same 12-camera/70%/80% gates or the Job fails. Missing verified nonlocal edges remains `open_trajectory_unverified`; it does not trigger fallback and is not loop-closure or bounded-drift evidence.

Completed manifests preserve `gaussian_geometry_source=vggt_ba` as the request and publish `gaussian_geometry_effective_source`, `gaussian_geometry_fallback_applied`, and the nullable allow-listed reason. Fallback Jobs remain viewable and retain VGGT diagnostics, but they are not successful VGGT-BA A/B evidence. Metrics also include profile, supported cameras/points, elapsed time, trajectory status, and verified nonlocal-edge count.

When VGGT-BA remains the effective source, the global camera/point solution may use all registered images, matching the existing camera-estimation contract. The Gaussian sparse initializer receives only points with at least two Train observations, recolored from Train observations. A COLMAP fallback follows ordinary-COLMAP initialization semantics instead. `vggt_ba_initialization_diagnostics` records accepted, heldout-only rejected, insufficient-Train-support, mixed-track, and recoloring counts only for effective VGGT-BA initialization. This does not authorize Validation/Test RGB in training.

`vggt_visibility_v1` is a fail-soft derivative after the immutable Original model has completed Validation and export. It recomputes VGGT depth from at most 64 Train images, aligns each usable depth map to final sparse geometry, and conservatively removes only multi-view free-space contradictions with no surface support or unsupported oversized Gaussians outside every Train capture envelope. The mask records row-aligned keep/reason/support arrays. It does not use Validation/Test to derive deletion decisions, fill holes, create walls, alter Original, retrain, or change navigation geometry.

When filtering succeeds, `gaussian_postprocess_status` is `available` and all filtered roles are published together. Original and filtered models receive separate Validation/export records and hashes; filtered bundles include `postprocess/diagnostics.json` and `postprocess/filter-mask.npz`. `scene_splat` remains Original for backward compatibility, while `scene_splat_vggt_filtered` is the optional Viewer A/B derivative. If filtering, filtered Validation, or filtered export fails, status is `unavailable`, a reason is recorded, no partial filtered roles enter `assets`, and Original remains a successful result.

Both capabilities remain experimental/research-only pending real CUDA jobs, cross-scene Validation, resource measurement, and license review. They retain normalized arbitrary units and make no metric-scale, complete-room, guaranteed-loop, or drift-free claim.

## Gaussian SOR floater cleanup (default pipeline step)

New `project_3dgs + gaussian_splat` jobs run a post-hoc statistical outlier removal (SOR) cleanup on the selected model snapshot before Validation evaluation and export. The request field defaults to the historical behavior plus cleanup:

```text
gaussian_sor_filter = on | off          (default on)
```

An operator may disable it per job through the API/frontend, or server-wide with `IMAGE3D_GAUSSIAN_SOR_FILTER=off` when the request does not override it. Only the Stage-1 render-lossless conservative band preset is used (`nb_neighbors=30`, `std_ratio=2.0`, `band_opacity=0.05`, 50% removal safety gate); no other tuning is exposed.

On success `gaussian_sor_filter_status` is `available`, the filtered snapshot replaces the training snapshot for Validation evaluation and export (so `scene_splat` is the cleaned model), and the export bundle records `postprocess/filter-record.json` and `postprocess/filter-mask.npz`. Asset roles `gaussian_sor_filter_record` and `gaussian_sor_filter_mask` point to the per-attempt record and mask. Metrics include `gaussian_sor_filter_input_count`, `gaussian_sor_filter_kept_count`, and `gaussian_sor_filter_removed_count`.

Failure is fail-soft: the status is `unavailable` with a reason, the unfiltered snapshot continues through Validation/export unchanged, and the job remains a successful result. Setting the field to `off` records status `disabled` and skips the filter entirely.

This step is independent of `gaussian_postprocess`. When both are requested, the SOR cleanup runs first on the selected snapshot, and any `vggt_visibility_v1` derivative is then derived from the SOR-filtered model; each lifecycle status is reported separately.

## Gaussian recovery-gated prune (experimental option)

```text
gaussian_recovery_prune = on | off      (default off)
```

When requested `on` (API form field, or server-wide `IMAGE3D_GAUSSIAN_RECOVERY_PRUNE=on` when the request does not override it), the resolved Gaussian config enables the schema-9 `opacity_reset.recovery_prune` leaf, and the project trainer fires a one-shot prune of gaussians still below the recovery threshold 500 iterations after each opacity reset (progress events carry `recovery_prune` with before/after counts). The request is echoed as the `gaussian_recovery_prune` manifest field, and the effective leaf is visible in the `gaussian_config` record. This is a research-only option: the room A/B passed its gates, but promotion to a default requires floater-rich video-job confirmation. No frontend switch exists yet.

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

COLMAP+VGGT world units are arbitrary unless an independently evaluated scale source is present. Gaussian navigation is likewise `coordinate_frame: normalized` and `world_units: arbitrary`; eye height `H`, capsule dimensions, movement speed, steps, and boundary coordinates are scene-relative rather than metres. The manifest and ETH3D's evaluation-time camera Sim(3) alignment do not establish true metres or metric accuracy.
