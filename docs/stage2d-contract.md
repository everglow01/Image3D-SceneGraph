# Stage 2D evaluation, export, and delivery contract

Stage 2D turns the project-owned Stage 2C model into evaluated and downloadable browser assets. It does not change the arbitrary-scale geometry claim and does not introduce an external trainer.

## Evaluation schema v1

`image3d_scenegraph.gaussian.evaluation` is shared by cadence validation and standalone evaluation. The trainer still calls `load_training_views`, which accepts only `train` and `validation`. The standalone evaluator accepts `validation` or `test`.

A held-out `test` run requires a frozen candidate record binding:

- candidate ID;
- dataset hash;
- effective-config hash;
- immutable model SHA-256;
- `selection_split: validation` and `test_metrics_seen: false`.

The evaluator atomically creates a sibling `*.test-consumed.json` with exclusive creation before loading test views. A candidate cannot be evaluated twice, and changed dataset/config/model hashes are rejected. Test output is terminal evidence only; no code path feeds it to training, checkpoint selection, or profile resolution.

Each result records per-view and mean/min/p10/p50/p90/max PSNR, SSIM, render milliseconds, aggregate FPS, failures, Gaussian count, opacity/scale distributions, duplicate/split and reason-specific prune/reset counts, and peak CUDA allocation/reservation. Training images remain CPU-resident and only the active view is transferred to CUDA. The frozen model is selected by best Validation PSNR while the final checkpoint remains lifecycle/resume evidence. Fixed rendered PNGs and JSONL rows are emitted. Rendering metrics are explicitly separate from geometry metrics.

LPIPS remains `not_run: pretrained_weight_license_and_hash_not_audited`. `lpips` is not installed. Upstream package code is BSD-2-Clause, but default AlexNet uses a torchvision ImageNet checkpoint downloaded at runtime and no approved local weight hash/license record exists. Stage 2D does not download it or substitute another metric.

## Canonical Gaussian export v1

`canonical.ply` is deterministic binary little-endian PLY in normalized coordinates and arbitrary units. Its vertex layout is:

1. `x y z`;
2. zero `nx ny nz` compatibility fields;
3. `f_dc_0..2`;
4. `f_rest_0..44`, ordered channel-major over SH bands 1–15;
5. opacity logit;
6. three natural-log scales;
7. normalized `wxyz` quaternion.

`scene.ply` is the versioned INRIA-v1 browser derivative. Today its bytes equal the canonical PLY; consumers must still use the separate role because future browser optimization cannot redefine the canonical format.

`export.json` records schema/format, coordinate frame, arbitrary units, camera axes, `world_from_normalized`, Gaussian/SH layout, checkpoint/model/dataset/config/evaluation hashes, browser opacity threshold/health summary, and media state. The browser reads its SH degree and alpha threshold from this metadata; missing `Content-Length` remains unknown rather than displaying zero. Export rejects a model with no effective-opacity rows. `camera_path.json` contains validation-camera keyframes in normalized coordinates. Every center must remain within radius 2 of the normalized origin. Stage 2 baseline retains this descriptor; video is `not_generated` rather than silently invoking an unpinned encoder.

`result.zip` uses sorted names, fixed timestamps, fixed permissions, and project-relative paths. It includes canonical/browser PLY, export/camera metadata, dataset/effective config, and validation evaluation. `bundle.json` records the archive SHA-256 externally to avoid a self-referential archive hash. No absolute local paths or optimizer internals enter the bundle.

## Worker and browser delivery

A successful `project_3dgs` attempt progresses through geometry, Gaussian training, validation, export, and atomic workspace publication. Stable roles are:

- `gaussian_model`, `gaussian_training_result`, `gaussian_progress`;
- `gaussian_dataset` at the real per-attempt preparation path;
- `gaussian_evaluation`, `gaussian_test_evaluation`, and `gaussian_test_decision`;
- `gaussian_export_metadata`, `gaussian_canonical`, `gaussian_camera_path`, `gaussian_bundle`;
- `scene_splat` for browser loading.

Failed/cancelled partial evaluation/export stays in lifecycle attempt diagnostics and is not advertised. The frontend progressively loads `scene_splat`, aborts its metadata request and disposes the viewer on switch/unmount, reports load/error/size state, labels the view as normalized arbitrary units, and retains point/mesh fallback when Gaussian is absent.

## Measured RTX 4060 public/development profile v1

Public `standard_v1` and internal `rtx4060_8gb_development_v1` resolve to the repaired schema-v3 room settings with effective-config hash `f50ba69533396ebe097cb60afd7d3b1a87292a9051a66704fe3fd4d2b2d117f5`. The training values match the measured schema-v2 configuration; only the obsolete checkpoint cadence field was removed:

- 3,000 iterations, longest edge 640;
- 20,000 sparse initial points in the measured room run, 250,000 Gaussian cap;
- SH degree increments every 1,000 iterations;
- densification iterations 200–1,500 every 100;
- opacity reset every 500 while densification is active;
- Validation every 500 iterations; only the final iteration writes a full checkpoint.

The 32-view retained room run on 2026-07-31 completed in 83.17 s with 417,403,904 peak allocated and 1,094,713,344 peak reserved bytes. Topology duplicated 218,548 rows, split 15,476 parents into 30,952 children, pruned 4,024 low-opacity rows, performed no screen-size prune, and reset opacity twice before densification ended. Validation reached PSNR 16.7151/SSIM 0.5803. After freezing, the one consumed Test evaluation recorded PSNR 8.8663/SSIM 0.4438; Test was not used to change the model/config. The 250,000-row canonical/browser PLY is 62,001,587 bytes and the deterministic bundle is 78,845,012 bytes; two exports matched hashes and the audited browser loader decoded all rows.

Training writes no periodic full checkpoint and a successful job retains only its final checkpoint. Validation candidates use one overwrite-in-place model-only snapshot. This avoids cadence-multiplied model/Adam snapshots. The profile is a measured 8GB-safe development result in arbitrary units, not a final quality optimum or metric-accuracy claim.

## Integrated evidence

Ignored final job `outputs/experiments/r2_15/integrated-room-v2/20260730_092826_4c1497f3` completed on its first attempt with the final validation→frozen-test→export worker route. It used 32 uploaded room images, COLMAP sparse reconstruction, 100-iteration project training, validation, one consumed held-out test evaluation, export, API retrieval, deterministic bundle, and viewer PLY parsing. It produced 3,608 Gaussians and an 896,369-byte browser PLY; the audited viewer loader decoded all 3,608 rows. Every advertised asset returned HTTP 200.

The earlier diagnostic job `outputs/experiments/r2_15/integrated-room-v1/20260730_091731_0ac5a833` retained two failed attempts that exposed and fixed workspace input handling:

1. relative input symlink resolved from the workspace to the wrong target;
2. absolute symlink fixed discovery but violated dataset-root containment.

The root-cause fix copies immutable uploaded inputs into the disposable attempt workspace. Original job inputs remain preserved, retries stay isolated, and the completed workspace is removed after atomic publication.
