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

Each result records per-view and mean/min/p10/p50/p90/max PSNR, SSIM, render milliseconds, aggregate FPS, failures, Gaussian count, opacity/scale distributions, duplicate/split and reason-specific prune/reset counts, normalized screen-radius and maximum-axis-scale health distributions, and peak CUDA allocation/reservation. Training images remain CPU-resident and only the active view is transferred to CUDA. The final 15,000-step model is published after the configured cleanup/recovery phase, while the final checkpoint remains lifecycle/resume evidence. Fixed rendered PNGs and JSONL rows are emitted. Rendering metrics are explicitly separate from geometry metrics.

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

## RTX 4060 public/development profiles

Public `standard_v1` and internal `rtx4060_8gb_development_v1` resolve to the same floater-resistant schema-v4 settings with effective-config hash `9287f537144bfef7c24503452db826ceb9e4b01201eda361070d6f7ebbde8efb`:

- 15,000 iterations, longest edge 1280 without upscaling smaller inputs;
- up to 75,000 sparse initial points and a hard 600,000 Gaussian cap;
- SH degree increments every 3,000 iterations;
- densification runs from iteration 500 through 7,500 every 100 iterations;
- normalized screen radius above 0.05 is eligible for early splitting, while radius above 0.15 and maximum normalized 3D scale above 0.1 are cleanup conditions;
- saturated growth reserves 25% of available capacity for split candidates before assigning the remainder to duplication;
- opacity resets run every 3,000 iterations, Validation every 1,500 iterations, and refinement pauses briefly after each reset;
- cleanup runs at the 7,500 densification boundary and at iteration 13,500, followed by 1,500 recovery iterations;
- only iteration 15,000 writes a full checkpoint, and the recovered final model is published.

These values replace the fast 3,000/640/250k development default after a 225-image user job completed in 59.24 s with only 1.076 GB peak reserved VRAM, reached the Gaussian cap by iteration 1,000, and produced incomplete Validation views. That job registered 176/225 uploaded images with sequential matching and initialized from 20,000/20,741 sparse points. Project 3DGS therefore also defaults to exhaustive COLMAP matching for registration coverage; the environment override remains available for explicitly ordered captures. This change is based on training/Validation/resource evidence only, not its consumed held-out Test.

The original higher-capacity profile was validated on the frozen 32-view room Validation split on 2026-08-01. It completed in 774.79 s with 962,405,888 peak allocated and 3,221,225,472 peak reserved bytes, selected iteration 13,500 at Validation PSNR 25.5475/SSIM 0.8244, retained only final checkpoint `iteration_000015000`, and produced 600,000 Gaussians. The 8,000/960/350k profile on the same split recorded 298.57 s, 1.508 GB peak reserved, and Validation PSNR 22.8994/SSIM 0.7884; the earlier 3,000/640/250k profile recorded 83.17 s, 1.095 GB, and 16.7151/0.5803.

The floater-resistant profile was then evaluated Validation-only on the retained 225-image scene. Against the original 15,000-step result, final Validation PSNR changed from 25.2788 to 24.8390 dB (-0.4398 dB), P10 PSNR from 20.2775 to 20.2025 dB (-0.0750 dB), and mean SSIM from 0.8697 to 0.8791 (+0.0094). Visible screen-radius violations above 0.15 fell from 827 to 81 (-90.2%), including high-opacity violations from 586 to 76 (-87.0%). The trainer performed 58,950 split replacements after the old capped strategy had starved splitting, pruned 2,979 screen-oversized rows, retained 591,511 Gaussians, completed in 903.00 s, and peaked at 3,135,242,240 reserved bytes. The remaining 81 violations appeared during the accepted 1,500-step post-cleanup recovery; this is an explicit quality/artifact compromise rather than a claim of complete floater removal. No held-out Test view was loaded or evaluated, and all coordinates and size thresholds remain normalized arbitrary units.

Training writes no periodic full checkpoint and a successful job retains only its final checkpoint. Validation candidates use one overwrite-in-place model-only snapshot. New evaluations and exports expose screen-radius, maximum-scale, and robust-distance health diagnostics without hiding rows in the browser. Export also records a robust median scene center and 95th-percentile radius; the browser uses that frame for stable scene-centered CAD-style orbit, bounded zoom, coherent presets, and click-to-focus while remaining backward-compatible with older export metadata.

## Integrated evidence

Ignored final job `outputs/experiments/r2_15/integrated-room-v2/20260730_092826_4c1497f3` completed on its first attempt with the final validation→frozen-test→export worker route. It used 32 uploaded room images, COLMAP sparse reconstruction, 100-iteration project training, validation, one consumed held-out test evaluation, export, API retrieval, deterministic bundle, and viewer PLY parsing. It produced 3,608 Gaussians and an 896,369-byte browser PLY; the audited viewer loader decoded all 3,608 rows. Every advertised asset returned HTTP 200.

The earlier diagnostic job `outputs/experiments/r2_15/integrated-room-v1/20260730_091731_0ac5a833` retained two failed attempts that exposed and fixed workspace input handling:

1. relative input symlink resolved from the workspace to the wrong target;
2. absolute symlink fixed discovery but violated dataset-root containment.

The root-cause fix copies immutable uploaded inputs into the disposable attempt workspace. Original job inputs remain preserved, retries stay isolated, and the completed workspace is removed after atomic publication.
