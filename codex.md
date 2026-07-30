# codex.md

> This file is the project working plan and maintenance guide for Codex and future development.
> It should evolve with the project. When scope changes, update this file first or in the same commit.

---

## 1. Project Name

**Image3D-SceneGraph**

Working description:

> A calibration-free image-to-semantic-3D reconstruction system intended for deployment through a company portal.
> The target static product accepts one or more images, reconstructs cameras and geometry on company GPUs, trains and evaluates a project-integrated 3D Gaussian scene, and returns browser-viewable and downloadable results. Static semantics follows that product loop; video, long-horizon mapping, and 360 panorama support are separately gated later extensions.

Important wording:

- The user does **not** provide camera intrinsics, extrinsics, depth, or pose.
- The system may still estimate camera parameters internally through geometry models.
- Do not claim true metric accuracy unless scale recovery has been evaluated.

---

## 2. Motivation

The previous `Mono3D-Grounding` project proved several useful pieces:

- monocular metric depth can run in real time after TensorRT optimization;
- webcam-to-point-cloud preview is feasible;
- SAM/segmentation experiments can provide object masks;
- pure monocular depth point clouds are unstable for real objects and can distort geometry.

This new project changes the center of gravity:

> From realtime monocular depth point clouds to full image/video/panorama-based semantic 3D scene reconstruction.

The goal is not to build a generic wrapper around many models. The goal is to build a coherent computer vision system with clear algorithmic contributions and a demonstrable frontend.

---

## 3. Target User Experience

A user opens the company portal and can:

1. Upload a single image or multiple images for the static product baseline.
2. Start a reconstruction job and receive a job ID without waiting for GPU work.
3. Watch persisted queue/stage/progress state and actionable failures.
4. View point cloud, mesh, and evaluated 3D Gaussian results in the browser when available.
5. Inspect semantic objects, evidence, relations, and physical diagnostics after the static semantic stage lands.
6. Download authorized, integrity-checked scene assets and result bundles, or delete retained data.

The public portal should expose a small versioned quality profile, not raw research/ablation controls. Internal CLI/admin paths retain the complete auditable configuration. Video and panorama are not assumed to share the static perspective-image baseline and remain separately gated extensions.

The system should feel like a usable product, not a loose collection of scripts.

---

## 4. Core Problem Definition

Inputs by roadmap stage:

- Static baseline: one RGB image or multiple unordered RGB images.
- Later gated extensions: video converted to selected frames, and an equirectangular 360 panorama through a panorama-aware adapter.

Unavailable from the user by default:

- camera intrinsics;
- camera extrinsics;
- camera trajectory;
- depth map;
- RGBD sensor data;
- manual object annotations.

Outputs:

- reconstructed geometry: point cloud, mesh, and project-integrated 3DGS assets when their gates pass;
- estimated camera parameters or trajectory when available;
- training/validation/test split metadata, effective configuration, checkpoints, render evaluation, and provenance for 3DGS jobs;
- semantic object instances, evidence, object-level 3D positions, spatial relations, and physical consistency diagnostics in the static semantic stage;
- authorized browser visualization and versioned export bundles.

---

## 5. Anti-Goals

Do not turn this into a universal 3D reconstruction platform.

Explicitly avoid:

- supporting every reconstruction model;
- supporting every viewer format early;
- training a large image-to-3D foundation model from scratch;
- using Nerfstudio or another complete external training platform as the Stage 2 runtime;
- exposing arbitrary backends, filesystem paths, checkpoints, or ablation hyperparameters through the public portal;
- overbuilding multi-node infrastructure before one durable bounded GPU-worker product loop is verified;
- forcing video or equirectangular panorama through the static perspective-image path;
- claiming centimeter-level accuracy without a benchmark;
- hiding model limitations behind a polished frontend.

Prefer one reliable baseline and one clear algorithmic improvement path.

---

## 6. High-Level Architecture

```text
Company Portal Frontend
  authenticated static image/multi-image upload
  persisted queue/stage/progress and failure UX
  point cloud / mesh / Gaussian viewer
  later semantic evidence and scene-graph UI
  authorized export and deletion

Public Backend API
  validate and stream uploads
  create a durable job and return its ID quickly
  authorize status, manifest, asset, export, cancel/retry, and delete operations
  expose only versioned public quality profiles

Metadata / Queue / Artifact Layer
  Stage 2: filesystem/manifest state, one local worker, one GPU-heavy job at a time
  persist job, attempt, stage, effective config, hashes, and checkpoint state across local restarts
  Stage 3 handoff: production database, queue, object storage, retention, auth, and multi-user policy

Geometry Worker
  estimate cameras and point geometry through project adapters
  preserve Raw/Aligned, arbitrary-scale, and provenance contracts
  export point cloud / cameras / optional mesh initialization

Project-integrated 3DGS Worker
  own dataset/camera/split/normalization contracts
  own Gaussian model and training lifecycle, hyperparameter validation, and effective config
  own densification/pruning, checkpoint/resume/attempts, validation/test evaluation, and export
  call only R2.0-approved narrow rasterizer/optimizer primitives; never a full external trainer

Static Semantic Worker (later)
  produce 2D evidence and geometry-grounded 3D instances
  infer spatial/support/physical relations with confidence and provenance
  publish a versioned evidence-aware scene graph

Optional Extension Workers (later)
  short-video tracking/keyframes/submaps
  loop closure/pose graph/scale/dynamics
  panorama-aware reconstruction behind a separate gate
```

The current repository still has a synchronous local `JobStore`. Stage 2 only targets a minimal filesystem/manifest-based async lifecycle for one RTX 4060 8GB and one serial GPU worker. Production database, distributed queue, object storage, retention, authentication, tenant isolation, company-server concurrency, deployment, and operations are Stage 3 handoff concerns, not Gate G2 requirements.

---

## 7. Recommended Tech Stack

Frontend:

- React + Vite + TypeScript.
- Keep the current research UI while Stage 2 is developed; Stage 3 replaces the public flow with upload, task timeline, viewer, evaluation summary, export, and deletion.
- Public users receive versioned quality profiles; internal CLI/admin retains experimental controls.
- A browser Gaussian renderer/decoder is allowed only if `R2.0` approves its pinned version, format, transitive dependencies, and license.

Backend and operations:

- Python environment management: `uv`.
- FastAPI remains a thin local API during Stage 2.
- Current local filesystem and synchronous `JobStore` are development baselines; Stage 2 adds only a filesystem/manifest-based async job/attempt lifecycle, one local worker, serial GPU-heavy execution, restart recovery, cancellation, and retry.
- Stage 2 explicitly does not select or implement a production database, Redis/distributed queue, object storage, data-retention/delete/backup policy, authentication, tenant/RBAC, signed downloads, company-server concurrency, deployment topology, or operations stack.
- Those product/deployment decisions are frozen by the responsible deployment work in Stage 3 and are not Gate G2 blockers.

Algorithm stack:

- Geometry fallback: current COLMAP+VGGT `sequential + points + global + any_support + random`; this is reproducible but not a newly improved Gate G1 baseline.
- Integrated 3DGS: this repository owns dataset/camera/coordinate contracts, Gaussian model and trainer orchestration, hyperparameter/effective-config validation, checkpoint/resume, training validation, isolated test evaluation, export, diagnostics, and tests.
- Nerfstudio/Splatfacto or another complete external training platform is prohibited as the Stage 2 runtime. Existing exported-splat registration remains legacy/reference-only.
- PyTorch/CUDA is allowed only in a separate optional GPU dependency group. Stage 2A selected the official binary `gsplat==1.5.3+pt23cu121` (tag `v1.5.3`, commit `937e29912570c372bed6747a5c9bf85fed877bae`, Apache-2.0) as the narrow differentiable rasterizer for the frozen CPython 3.10 / Linux x86_64 / PyTorch 2.3 CUDA 12.1 matrix. Its RTX 4060 sm_89 forward/backward/finite-difference check passes. The current browser renderer remains `@mkkellogg/gaussian-splats-3d==0.4.7` (MIT, lockfile-pinned) for display only. Neither dependency owns project training or canonical export.
- Static semantics: segmentation/open-vocabulary/VLM proposals fused into 3D and validated through geometry, provenance, scale uncertainty, and physical checks.
- Video, long-horizon mapping, and panorama are optional later extensions rather than prerequisites for the static portal.

---

## 8. Repository Layout

Suggested initial layout:

```text
Image3D-SceneGraph/
  codex.md
  README.md
  .gitignore
  frontend/
  backend/
  src/
    image3d_scenegraph/
      __init__.py
      jobs.py
      geometry/
      semantics/
      fusion/
      physics/
      scene_graph/
      export/
  scripts/
  docs/
  examples/
    inputs/
    outputs/
  outputs/          # ignored
  checkpoints/      # ignored
  external/         # ignored
```

Keep algorithm code in `src/`. Keep one-off runnable entrypoints in `scripts/`. Keep frontend and backend separate enough that either can be replaced.

---

## 9. Data and Output Contract

Each job evolves through versioned durable state and one or more immutable attempts. A completed static 3DGS job should converge on a layout like:

```text
outputs/jobs/{job_id}/
  input/
    images/
  geometry/
    cameras.json
    points.ply
    points_aligned.ply
    mesh.glb
  gaussian/
    scene.<canonical-format>
    previews/
  checkpoints/
    attempt-{attempt_id}/
  evaluation/
    validation.jsonl
    test.json
  diagnostics/
  semantic/
    masks/
    objects.json
  scene_graph/
    scene.json
  logs/
    attempt-{attempt_id}.log
  exports/
    bundle-manifest.json
  manifest.json
```

`manifest.json` remains the frontend's stable entry point. Stage 2 must version it beyond today's terminal-only contract while preserving old-job reads. At minimum it records:

- schema/job/attempt identity, status, stage, progress, timestamps, and structured failure/cancellation state;
- source input hashes and dataset/camera/split/Raw-Aligned/normalization contracts;
- requested profile, effective internal configuration and hash, code/dependency/environment provenance;
- complete assets only, with relative paths, roles, hashes, coordinate system, units/scale state, and integrity status;
- checkpoint/resume metadata and validation/test metric history without allowing test-set tuning;
- export inventory.

Allowed lifecycle values include `queued`, `running`, `exporting`, `done`, `failed`, and `cancelled`; detailed stages are separate fields. A retry creates a new attempt and does not overwrite a valid attempt. Partial checkpoints/exports are never published as successful assets. Existing manifests that only contain terminal `done` fields remain valid as documented in `docs/manifest-schema.md`. Owner/tenant/retention fields are reserved for Stage 3 rather than implemented speculatively in Stage 2.

All generated inputs, datasets, checkpoints, rendered views, evaluator outputs, previews, point clouds, and job diagnostics stay in ignored artifact storage and must not enter Git.

---

## 10. API Draft

The API has two phases. Stage 2 only needs a local research API that decouples requests from one serial GPU worker and persists filesystem/manifest state. Stage 3 owns the authorized public portal contract after deployment requirements are known.

```text
POST /api/jobs
  Stage 2: validate a local static image/multi-image upload, persist the job,
  enqueue it behind at most one running GPU-heavy job, and return job ID/status quickly.
  Stage 3: add public quality-profile and authorization boundaries.

GET /api/jobs/{job_id}
  Return authorized status, attempt, stage, progress, timestamps, and safe error details.

GET /api/jobs/{job_id}/manifest
  Return the versioned stable output contract, including only complete assets.

POST /api/jobs/{job_id}/cancel
  Idempotently request cancellation.

POST /api/jobs/{job_id}/retry
  Create a new bounded attempt without overwriting a valid attempt.

GET /api/jobs/{job_id}/assets/{path}
GET /api/jobs/{job_id}/download
DELETE /api/jobs/{job_id}
  Stage 2: local integrity-checked asset access/export; production authorization,
  signed download, retention, and deletion semantics wait for Stage 3.

GET /api/backends
  Internal/admin capability endpoint; never expose setup commands, local paths, or arbitrary backends to public users.
```

Internal CLI/admin may select experimental geometry factors and full versioned 3DGS configurations, but every choice must resolve to an effective config/hash in the job record. The future ordinary-user API must not accept paths, checkpoints, arbitrary backend names, or raw training hyperparameters.

Do not expand the Stage 2 local API beyond the algorithm workflow. Production auth/storage/retention/deployment endpoints and semantics wait for Stage 3.

---

## 11. Frontend MVP

The current research frontend remains useful during Stage 2. The public company-portal rewrite belongs to Stage 3 and should include:

1. Authorized static image/multi-image upload with consent and hard limits.
2. One or a small number of versioned quality profiles; no public ablation panel.
3. Recoverable queue/stage/progress, cancel, retry, and actionable failure states.
4. Point cloud / mesh / Gaussian viewing with coordinate and arbitrary-scale disclosure.
5. Evaluation summary that distinguishes render quality from geometry quality.
6. Authorized export, retention, and deletion controls.
7. Static semantic object/evidence/relation panels after Stage 4.

Internal CLI/admin continues to expose full experimental configuration. Hiding public controls must not delete research reproducibility.

Viewer priorities:

1. Preserve stable manifest-driven loading and current point/camera Raw/Aligned diagnostics.
2. Load the Stage 2 canonical/browser Gaussian export only after format and license approval.
3. Handle progressive loading, cancellation, malformed/missing assets, and bounded large-scene behavior.
4. Keep point, mesh, Gaussian, cameras, and later object anchors in an explicit shared coordinate contract.
5. Add semantic evidence and relation overlays only after the corresponding stable schema exists.

Do not build a marketing landing page before the actual authorized product loop works.

---

## 12. Algorithmic Contributions

This project must not be only model integration. The resume value should come from measurable algorithmic modules.

Primary algorithm modules:

### 12.1 Scale Recovery

Problem:

- Reconstruction models may produce geometry up to an arbitrary or unstable scale.

Possible solution:

- Treat scale as an explicit provider contract: `unknown`, user reference, camera height, stereo/RGB-D, IMU, or an evaluated metric-depth prior.
- Preserve source, evidence, uncertainty, and units in every derived asset.
- Begin with an optional user-provided reference measurement after the static geometry/semantic contracts are stable.
- Object and scene priors may propose or sanity-check scale, but category averages must not be presented as metric truth.

Output:

- scale factor and coordinate transform where applicable;
- source/evidence and uncertainty;
- before/after estimates clearly labelled as arbitrary or metric.

### 12.2 Semantic 3D Fusion

Problem:

- 2D segmentation and VLM labels exist per image, but final scene is 3D.

Possible solution:

- Project 2D masks into reconstructed point cloud or mesh.
- Fuse labels across frames by visibility and agreement.
- Produce object-level 3D instances.

Output:

- 3D object list;
- per-object point indices or mesh segment IDs;
- label confidence.

### 12.3 Physical Consistency Verification

Problem:

- Reconstructed objects may float, penetrate, tilt incorrectly, or violate obvious scene structure.

Rules to implement incrementally:

- tables, floors, walls, doors, and monitors are approximately planar;
- walls and doors are vertical;
- tables and floors are horizontal;
- cups, bottles, and humans are upright relative to support plane;
- supported objects should touch or be near their support surface;
- solid objects should not strongly interpenetrate.

Output:

- violation count per scene;
- list of violated rules;
- corrected object pose when correction is reliable.

### 12.4 Scene Graph Reasoning

Problem:

- A 3D model alone does not answer semantic spatial questions.

Relations:

- `on`, `under`, `left_of`, `right_of`, `in_front_of`, `behind`, `near`, `inside`, `supports`, `occludes`.

Output:

```json
{
  "objects": [
    {"id": 1, "label": "cup", "center": [0.3, 0.8, 0.7]},
    {"id": 2, "label": "table", "center": [0.0, 0.7, 0.4]}
  ],
  "relations": [
    {"subject": 1, "relation": "on", "object": 2, "confidence": 0.91}
  ]
}
```

---

## 13. Evaluation Plan

Geometry reconstruction is evaluated first on a frozen public benchmark, independently from the later semantic benchmark below.

### 13.1 ETH3D geometry benchmark

The first pilot is the ETH3D high-resolution MVS `pipes` training scene: 14 undistorted 6220×4141 images, reference COLMAP cameras, and the official `scan_eval` laser-scan input. The scene contract is versioned in `benchmarks/eth3d-v1/pipes.json`; large data and generated results remain under ignored `data/` and `outputs/` directories.

Protocol:

- Reconstruction receives RGB images only. ETH3D reference cameras and scans are evaluation-only.
- Match estimated and reference cameras by image name and estimate exactly one global RANSAC + Umeyama Sim(3) from camera centers.
- Apply that transform once to the raw reconstruction PLY, then use the official ETH3D evaluator for Accuracy, Completeness, and F1 at 1/2/5/10/20/50 cm.
- Do not use scan ICP, scan-derived scale refinement, per-window GT alignment, GT-guided cropping, or the generic Z-up point cloud.
- Label results as GT-camera-Sim(3)-aligned geometry quality, not native metric-scale recovery.
- Stabilize the single-scene runner and result schema before adding multi-scene orchestration or tuning reconstruction algorithms.

### 13.2 Static 3DGS evaluation

Stage 2A freezes the executable protocol in `docs/stage2a-contract.md` and `image3d_scenegraph.gaussian.dataset`. The matrix is: a generated 12-camera contract smoke scene; 32 spatially selected views from retained private job `20260723_070028_024e9f25`; 32 views from public ETH3D `terrains`, using only RGB plus project-estimated COLMAP cameras; and 32 views from Mip-NeRF 360 `room` at 779×519 as the primary public indoor product proxy. Mip-NeRF 360 `bonsai` is also retained locally as the public fine-detail/literature-reference scene. The official project page provides no dataset-specific license or redistribution statement, so both scenes remain ignored local inputs and do not establish redistribution rights.

The deterministic split seed is `20260729`. Camera-spatial farthest-point traversal selects held-out views; each scene uses approximately 80/10/10 with at least two validation and two test views. The protocol separates:

- train views used for optimization;
- validation views used for monitoring, checkpoint selection, and bounded hyperparameter decisions;
- test views isolated until candidate/configuration freeze and used only for final evaluation.

Subject to dependency/license approval, report PSNR, SSIM, LPIPS, per-view failures and distributions, render time/FPS, Gaussian count, peak memory, training time, and artifact size. Keep rendering quality separate from ETH3D geometry quality. Record split, seed, effective-config hash, checkpoint/attempt, code/dependency environment, and resource profile. Never tune on test views or present a visually attractive training-view render as held-out evidence.

### 13.3 Semantic office/tabletop benchmark

Build a small office/tabletop benchmark. It does not need to be large, but it must be consistent.

Suggested dataset:

- 20 single-image scenes.
- 10 multi-image scenes.
- 10 short videos.
- Mix of desk, monitor, cup, laptop, chair, person, door, wall, floor.

Annotations:

- object list;
- coarse 2D boxes or masks;
- object relations;
- relative depth order;
- support relation;
- optional approximate real measurements for selected objects.

Metrics:

- object precision / recall / F1;
- depth-order accuracy;
- relation F1;
- support-relation accuracy;
- physical violation count per scene;
- query success rate;
- runtime and GPU memory;
- export success rate.

Ablation table:

```text
Geometry baseline only
+ semantic fusion
+ scale recovery
+ physical verifier
+ physical correction
+ scene graph reasoning
```

This table is essential for showing algorithmic value.

---

## 14. Roadmap Milestones

`plan.md` is the task-level checklist and gate definition. This section records the product ordering; it does not mark future implementation complete.

### Completed foundation: bootstrap, mock product loop, and Stage 1 geometry research

Current repository capabilities include the FastAPI/React manifest-driven demo, geometry adapters, COLMAP/VGGT point-cloud and mesh paths, browser viewers, diagnostics, and the Stage 1 factorial review.

Stage 1 ended with no candidate satisfying Gate G1. `G1.26` remains blocked; `sequential + points + global + any_support + random` is retained only as the reproducible fallback.

### Stage 2: Project-integrated static 3DGS

Deliverables:

- blocking `R2.0` decision/license/build matrix;
- deterministic dataset/camera/coordinate/split/initialization contract;
- project-owned Gaussian model/training lifecycle and versioned hyperparameter/effective-config contract;
- atomic checkpoint/resume/attempts;
- training-time validation and isolated final test evaluation;
- minimal filesystem/manifest async job state and one serial RTX 4060 GPU worker;
- canonical export, integrity checks, browser viewing, and resource profiles;
- clean upload → train → evaluate → export → view/download smoke.

Success criteria:

- Gate G2 passes without Nerfstudio or another full external training platform in runtime.

### Stage 3: Company portal productization and deployment

Deliverables:

- company-approved metadata DB, queue, artifact storage, deployment, migration, backup, and rollback;
- streaming upload validation, quotas, retention, and deletion;
- auth, tenant/RBAC isolation, audit, bounded GPU scheduling, observability, and cost/capacity controls;
- public profile/internal experiment separation;
- rewritten public portal flow with safe progress, cancel/retry, viewing, evaluation, and export.

Success criteria:

- Gate G3P passes in a company-like multi-tenant staging environment.

### Stage 4: Static semantics and evidence-aware scene graph

Deliverables:

- frozen static semantic benchmark and model adapter contracts;
- 2D evidence, mask-to-3D lifting, multi-view object association, and persistent object map;
- explicit scale-provider/uncertainty state;
- geometry-grounded spatial/support/physical relations and conservative correction candidates;
- versioned scene graph, query/export, frontend evidence display, and evaluator.

Success criteria:

- Gate G4S passes without confusing VLM proposals, uncertain scale, or weak geometry with facts.

### Stage 5: Optional video and long-horizon extensions

Deliverables:

- short-video ingestion, tracking/relocalization, keyframes, bounded windows/submaps, queue/backpressure, and online evaluation;
- separately gated loop closure, pose graph, scale drift, dynamic masking/objects, and long-sequence evaluation;
- panorama-aware adapter candidate behind its own benchmark gate.

Success criteria:

- Gate G5A freezes only an evidence-backed short-video near-real-time baseline; Gate G5B independently controls any long-horizon mapping claim. Failure of Stage 5 does not invalidate the static portal.

### Evaluation and packaging throughout

Every stage retains mixed/failed outcomes, fixed-input reproducibility, runtime/GPU measurements, stable schemas, clean build instructions, and evidence-backed wording. Generated datasets, checkpoints, renders, evaluators, point clouds, screenshots, and job diagnostics remain ignored artifacts.

---

## 15. Development Rules

1. Keep changes surgical.
2. Do not add abstractions before at least two call sites need them.
3. Every new model integration must produce a stable internal format.
4. Every long-running algorithm step must write logs and intermediate artifacts.
5. Frontend should read manifests and assets; it should not know model internals.
6. Prefer measurable improvements over visual-only claims.
7. Manage Python dependencies with `uv`; add heavy dependencies only when the related code lands.
8. Update this file when the project direction changes.

---

## 16. Current Implementation Order

Completed foundation:

1. Repository/bootstrap, mock API, frontend upload/viewer, geometry adapter, VGGT/COLMAP paths, and manifest-driven assets.
2. Stage 1 diagnostics and factorial review.
3. `G1.26` review with no candidate promotion; the existing stable configuration remains a fallback and the gate remains blocked.

Next work is documentation-gated:

1. `R2.0` was confirmed on 2026-07-29 for the algorithm-development scope recorded in `plan.md`: project-integrated 3DGS, approved narrow dependencies, frozen evaluation isolation, and one local RTX 4060 8GB with serial GPU-heavy execution.
2. Execute `R2.1` through `R2.16` in dependency order.
3. Productize the static portal in Stage 3 only after the deployment owner supplies database/storage/retention/auth/company-infrastructure requirements.
4. Add static semantic object maps and scene graphs in Stage 4.
5. Treat short video, long-horizon mapping, and panorama as optional Stage 5 extensions.

Stage 2 must not introduce speculative production database, object storage, retention, auth, tenant, multi-GPU, or deployment architecture. It may add only the minimum local async state and single-GPU memory/queue controls required to develop, resume, evaluate, and export 3DGS reliably.

---

## 17. Current Decision Log

- New workspace: `/home/owen/Image3D-SceneGraph`.
- Old workspace `/home/owen/3d_demo` remains an exploration repo.
- Project direction: a static image/multi-image company-portal reconstruction product first; semantic static scenes next; video, long-horizon mapping, and panorama remain later gated extensions.
- Frontend is part of MVP, not a later add-on.
- User does not provide camera parameters; the system estimates them internally.
- Algorithm value should come from scale recovery, semantic 3D fusion, physical consistency, and scene graph reasoning.
- Python environment management should use `uv`.
- Mock backend API now creates local jobs, writes `manifest.json`, serves mock assets, and exposes scene graph JSON.
- `panorama` is a supported input mode for one equirectangular 360 image; real panorama reconstruction will come later.
- Frontend MVP now supports mode selection, file upload, mock job creation, manifest/scene display, asset links, and `.ply` point cloud viewing.
- Reconstruction adapter contract now exposes `geometry_backend` and `output_type`; `mock + point_cloud`, `vggt + point_cloud`, `colmap + point_cloud`, and `colmap_vggt + point_cloud` are implemented.
- Optional heavy model integrations are explicit local backends. They are not installed by the base package; `GET /api/backends` reports availability and `scripts/setup_model.py` is the setup entry point. VGGT setup defaults to dry-run and requires `--install` because its checkpoint is about 5GB plus environment dependencies.
- VGGT first baseline uses `scripts/run_vggt_pointcloud.py`, exports `geometry/points.ply` and `geometry/cameras.json`, and is invoked by `VggtPointCloudAdapter`. On this machine, unset `LD_LIBRARY_PATH` for VGGT/PyTorch CUDA runs to avoid linking against the system cuDNN ahead of the uv environment libraries.
- RTX 4060 8GB cannot run 4 uploaded images through VGGT-1B in one fp32 pass; it OOMs in the depth head. The runner now supports overlapping groups with `--batch-size` and `--overlap-size`, and the backend defaults to `IMAGE3D_VGGT_MAX_IMAGES=8`, `IMAGE3D_VGGT_BATCH_SIZE=8`, and `IMAGE3D_VGGT_OVERLAP_SIZE=4`. The runner converts the VGGT backbone to auto half precision on CUDA while keeping camera/depth heads fp32 to avoid LayerNorm dtype errors.
- The frontend exposes per-job VGGT `Max images`, `Batch size`, and `Overlap` controls. A 225-image upload can be attempted by setting `Max images=225`; with `Batch size=8` and `Overlap=4`, this is still slow and global drift remains likely until a stronger global alignment/bundle-adjustment stage is added.
- Windowed VGGT group stitching now estimates a Sim3 transform from shared camera centers instead of a rigid-only SE3 transform. A 45-image office upload with `Batch size=8` and `Overlap=4` produced 11 groups.
- Point-cloud viewer has display-only X/Y/Z axis flip and Raw/Aligned controls; users can toggle them without rerunning reconstruction. Its Cameras toggle reads the existing COLMAP or VGGT `geometry/cameras.json`: Raw centers/frusta/trajectory stay in the source world frame, while Aligned applies the same retained `diagnostics/alignment.json` 4x4 transform as `points_aligned.ply` before shared cloud-centering and axis-sign display transforms. Missing camera/alignment diagnostics disable only the overlay; no reconstruction asset is modified.
- Residual 45-image drift is still expected because current stitching is local window Sim3, not global pose graph optimization or bundle adjustment. Next geometry improvement should add a global pose graph over VGGT window cameras and optimize Sim3/SE3 constraints before merging point clouds.
- VGGT point colors now come from original RGB images resized/padded to the VGGT input shape, not from model tensors. This fixed the observed green color cast on a 45-image office upload; the fixed point cloud mean RGB was close to the original image mean RGB.
- COLMAP sparse SfM baseline is wired through `scripts/run_colmap_sparse.py` and `ColmapPointCloudAdapter`. It exports `geometry/points.ply` and `geometry/cameras.json` for global SfM comparison, and requires the system `colmap` executable on PATH.
- COLMAP + VGGT dense baseline is wired through `scripts/run_colmap_vggt_dense.py` and `ColmapVggtPointCloudAdapter`. It runs COLMAP global SfM, runs VGGT depth in batches, estimates a per-image depth scale using COLMAP sparse observations versus VGGT depth samples, and fuses dense points in COLMAP's global frame. On the 45-image office set, `--matcher exhaustive` registered/scaled 45/45 images and exported 300000 points.
- Dense fusion must use COLMAP's calibrated camera model and its global pose together. The runner converts COLMAP intrinsics into VGGT's resized/padded depth canvas and inverse-corrects `SIMPLE_RADIAL` or `RADIAL` distortion before back-projection; VGGT predicted intrinsics are retained only for diagnostics. Each COLMAP+VGGT job exports `diagnostics/fusion.json` with per-image intrinsics, depth-scale observation counts, scale dispersion, and VGGT-to-COLMAP focal ratios. This is the Phase A baseline; cross-view depth consistency filtering and voxel/TSDF fusion are still required to remove residual duplicate surfaces.
- Phase B starts with covisibility-constrained depth filtering. The runner excludes the white padded part of VGGT's image canvas, builds up to six COLMAP neighbors per image from at least 20 shared sparse tracks, reprojects dense points into those views, treats nearer observations as occlusions, and rejects only points with visible contradictory depth. It writes `diagnostics/visibility_graph.json` and `diagnostics/consistency.json`; the effective relative-depth threshold is derived from median per-frame scale dispersion and clamped to 2-8%. A 24-image GPU smoke run completed with all images connected. Voxel/TSDF fusion and the corresponding viewer comparison remain a later Phase B step, after validating the filtered full 225-image result.
- User has a Nerfstudio splatfacto checkpoint at `/home/owen/nerfstudio/outputs/drjohnson_hq/splatfacto/2026-06-22_161605/nerfstudio_models/step-000029999.ckpt`, but no browser-ready `.splat/.ply/.ksplat` export was found there.
- Nerfstudio `ns-export gaussian-splat` successfully exported `/home/owen/Image3D-SceneGraph/outputs/exports/drjohnson_hq/splat.ply` from that checkpoint; this file is intentionally under ignored `outputs/`.
- `scripts/register_gaussian_splat.py` can register an exported `.ply/.splat/.ksplat` as a local `nerfstudio_3dgs + gaussian_splat` job.
- ETH3D high-resolution MVS `pipes` is the first frozen geometry benchmark. Its 14 undistorted 6220×4141 images are the only reconstruction inputs; reference COLMAP poses and `scan_eval` are evaluation-only. Offline evaluation estimates one global camera-center RANSAC Sim(3), then invokes the official ETH3D evaluator. Scan ICP, GT-guided reconstruction, and metric-scale claims are prohibited. Full multi-scene benchmark orchestration is deferred until this single-scene result schema is stable.
- Phase B voxel/TSDF fusion is available as an experimental opt-in (`--fusion-mode tsdf` or `IMAGE3D_COLMAP_VGGT_FUSION_MODE=tsdf`); the stable default remains cross-view-filtered point fusion (`points`). Per-frame VGGT depth is rescaled to COLMAP, thresholded by a per-frame confidence percentile, pad-masked, undistorted for `SIMPLE_RADIAL`/`RADIAL`, and integrated into an Open3D `ScalableTSDFVolume`. Auto voxel sizing uses the 0.5%-99.5% percentile-clipped COLMAP sparse extent, not the full min/max: job `20260711_082305_16f2872e` showed that sparse outliers inflated the full diagonal from 19.45 to 142.99 and made voxel size 0.1396 instead of 0.0190, collapsing output from millions of points to 48,601. TSDF now rejects runs that integrate under 90% of frames or produce implausibly sparse output rather than recording them as successful. `--fusion-mode points` and `IMAGE3D_COLMAP_VGGT_FUSION_MODE=points` are the production defaults.
- For the `colmap_vggt` path, poses come entirely from COLMAP's global SfM, so VGGT batching affects depth quality but not global camera drift. Covisibility-ordered overlapping VGGT windows remain experimental (`--vggt-grouping covisibility`, `--vggt-overlap-size 2`); stable API jobs explicitly default to disjoint sequential groups via `IMAGE3D_COLMAP_VGGT_GROUPING=sequential`. Strict G1.7 paired tests found mixed ETH3D changes, while a private 225-image chunked run reduced bookshelf repeated layers from 7 to 4 and increased ROI coverage from 0.6600 to 0.7175. The DPT head's existing `frames_chunk_size=2` path completed that 4-view run on the RTX 4060 8GB without changing model math. G1.8 retained all already-computed window predictions while preserving a byte-identical `first_wins` private point cloud. G1.9 then measured deterministic held-out overlap residuals on private-225, pipes, terrains, and delivery_area: an extra scale-only fit reduced median anchored log-depth p50/p90 by 30.71%/16.42%, -1.62%/2.32%, 7.91%/4.29%, and 26.12%/19.15%, so no scene passed the predeclared 50%/30% scale-dominant gate. G1.10 overlap scale reconciliation is therefore blocked rather than retuned. G1.18 audited 358,085 sparse points across the same four frozen scenes and every production projection/backprojection chain passed the fixed 0.001-pixel maximum-error gate; however, the production `scale*x + pad` convention differs from PIL resize's pixel-center mapping by a deterministic 0.295-0.458 canvas pixels per axis. G1.19 compared identical frozen sparse points, poses, distortion, and canvases: shifting only the COLMAP principal point to the PIL pixel-center convention improved edge reprojection p90 in 4/4 scenes with 79.35% median reduction and no fallback, while raw VGGT intrinsics and VGGT focal lengths improved 0/4 and regressed sharply. G1.20 then replayed the retained first-wins depth/confidence with all other dense-fusion factors frozen and verified production replay by byte-identical PLYs in all four scenes. The pixel-center candidate reduced private monitor/bookshelf thickness by 13.57%/10.91% and layer counts from 5/4 to 2/3, but reduced coverage, raised plane RMS, regressed cross-view residual p90 in 4/4 scenes, and produced mixed ETH3D transfer: 2/5 cm F1 deltas were -0.000859/-0.002633 on pipes, -0.008438/-0.005251 on terrains, and +0.027182/+0.005364 on delivery_area. It is therefore rejected as a production default rather than retuned; current COLMAP intrinsics remain unchanged. Covisibility remains experimental and sequential remains the stable default.
- Points-mode cross-view fusion has an opt-in G1.14 per-final-point provenance sidecar (`--support-diagnostics-output`): a compressed NPZ plus a JSON schema/index retaining source image and first-wins window, canvas pixel, confidence, visible/support/occluded counts, mean residual, sparse scale quality, and same-image retained-overlap disagreement. G1.16 makes every neighbor check an explicit four-state partition by additionally retaining contradiction and not-observed counts: `visible = supported + contradicted`, while supported + contradicted + occluded + not-observed equals checked neighbors. It reuses the existing validation pass and applies the exact final point-budget indices, so each row matches the final PLY vertex order; the default path allocates/writes none of these arrays, TSDF rejects the option, and frontend/manifest contracts remain unchanged. A frozen private-225 replay produced 17,746,411 aligned rows, exactly reproduced aggregate support counts and the baseline PLY hash; old unverified points split into 1,064,699 occluded-only and 3,079,804 not-observed-only. G1.17 tested a threshold-free `contradiction_free` diagnostic policy that preserves zero-visible points but rejects every reliable visible conflict. The private ROI gate passed, and terrains F1 improved, but 2/5 cm F1 regressed on pipes and delivery_area; the candidate therefore failed its predeclared three-scene transfer gate and is not exposed through the production CLI/API/frontend. G1.17a-lite then compared the four frozen scene-pressure profiles: private-225 is uniquely large in absolute images/groups/overlap pairs, but delivery_area has higher normalized multi-window, overlap, and mixed-conflict pressure while regressing most at 5 cm. No single pressure statistic supports automatic activation. Retain G1.7/G1.17 as limited-evidence target-domain conditional research candidates, preserve the historical global-default failure and `any_support` default, and do not fit a selector from these four scenes.
- Points-mode VGGT confidence is globally percentile-thresholded by default. A strict paired ETH3D ablation added `--confidence-threshold-scope per_frame`, deriving one threshold from each valid, non-padding confidence canvas and using the target frame's own threshold during cross-view validation. With identical COLMAP poses, VGGT depths, scales, and camera-centre Sim(3), 2/5 cm F1 changed by +0.030726/+0.125765 on `pipes`, -0.020878/-0.034423 on `terrains`, and +0.022218/+0.032896 on `delivery_area`. The mode is retained as an auditable experimental strategy but does not replace the global production default because it regressed one transfer scene; no percentile was retuned from scan metrics.
- Cross-view points support `--consistency-support-policy adaptive_two` as an experimental accuracy-priority mode. The fixed policy requires two supporting views only when at least two high-confidence, non-occluded neighbors are visible; zero/one-visible cases preserve the baseline requirement. Strict paired 2/5 cm F1 deltas were +0.001591/+0.001877 on `pipes`, +0.002731/+0.003490 on `terrains`, and +0.000145/+0.001771 on `delivery_area`. Accuracy rose across all thresholds and scenes, with small completeness losses and a -0.000171 delivery_area 1 cm F1 delta, so `any_support` remains the default.
- A strict paired 2x2 factorial test evaluated `global/per_frame` confidence scope against `any_support/adaptive_two` on the same in-memory reconstruction. The combined `per_frame + adaptive_two` 2/5 cm F1 deltas from `global + any_support` were +0.025115/+0.114945 on `pipes`, -0.018696/-0.030540 on `terrains`, and +0.013222/+0.028763 on `delivery_area`. Phase 2 only partly compensates for Phase 1 and the interaction is usually slightly negative; the combination therefore remains experimental and the stable defaults remain `global + any_support`.
- A strict paired 2x2x2 factorial added Phase 3 `--point-budget-policy spatial_balanced`, which stable-sorts accepted XYZ points by a 21-bit-per-axis Morton code and deterministically chooses equal-mass stratum midpoints at the same point budget. Against the seeded random cap, baseline-upstream 1/2/5 cm F1 changed by +0.006265/+0.005373/+0.002104 on capped `terrains` and +0.002567/+0.005658/+0.005174 on capped `delivery_area`; `pipes` was unchanged at the frozen 2M budget because all arms were below the cap, while a separately labelled 1M activation check improved all six F1 thresholds. Phase 3 improved every active F1 pair across all four Phase 1/2 upstream combinations on both capped scenes, but it does not reverse Phase 1's terrains regression. Retain `random` as the stable default until one further non-ETH3D large-scene validation; keep the Morton policy as the accepted experimental candidate and never select it from GT.
- The frontend and `POST /api/jobs` expose Phase 1 confidence scope, Phase 2 consistency support, and Phase 3 point budgeting as three independent COLMAP+VGGT controls. All eight combinations are selectable, completed jobs persist the effective policies in manifest metrics, and the stable user-facing defaults remain `global + any_support + random`. Private 225-image comparisons should use separate job IDs and change one factor at a time while holding depth batch, confidence percentile, output point budget, inputs, and environment fixed.
- Generic point-cloud alignment now analyzes three RANSAC plane candidates and selects the candidate with the strongest global inlier ratio unless `--plane-index` is explicitly set. This fixed the TSDF regression job above, whose first candidate had 7.10% support but whose second candidate had 11.96%, without lowering the 8% plane-quality threshold. G1.22 adds a separate diagnostic-only Manhattan-frame evaluator: it analyzes up to eight planes, filters them with the same 8% gate, clusters unoriented near-parallel normals, and reports supported orthogonal triplets plus partial evidence and explicit ambiguity. G1.23 then adds a separate gravity-axis evidence evaluator over an unambiguous G1.22 frame: it audits optional IMU records, COLMAP camera image-up, camera-centre/point robust spans, and reliable boundary-plane ordering with frozen per-source scores, selection/margin gates, and explicit missing/invalid fallback. EXIF Orientation is not treated as gravity, camera image-up is only a capture prior, no plane is labeled ground, and weak/conflicting evidence remains ambiguous. Retained private-225 has no IMU sidecar; all four available geometric sources selected Manhattan axis 0 with combined scores 0.52815/0.23707/0.23477 and margin 0.29108, while camera image-up coherence selected the negative axis direction as up. G1.24 applied that frozen triad/sign in an offline rigid-transform ablation without rerunning reconstruction or plane detection. Against the retained single-plane transform, Manhattan changed reliable-plane support-weighted cardinal residual only from 0.83416° to 0.81222° but reduced the maximum from 2.95460° to 1.25981° and, critically, mapped the selected up to +Z rather than the single-plane result's near-exact -Z. Six exact-relative-view Raw/single-plane/Manhattan screenshots preserved cloud-camera registration and confirmed that local duplicate layers/thickness are rigid-transform invariants, not alignment gains. Ambiguous or unsigned evidence falls back to single-plane. Manhattan is retained only as an experimental G1.25 candidate from one private scene; `align_pointcloud.py`, retained assets, API/frontend contracts, and the production default remain unchanged.
- G1.25 freezes the Stage 1 combination evidence with a read-only analyzer over the existing three-scene eight-arm Phase 1/2/3 factorial; no reconstruction, alignment, official evaluator, or GT-guided operation was rerun. Balanced 2/5 cm F1 effects confirm that `per_frame` remains mixed (positive on pipes/delivery_area, negative on terrains), `adaptive_two` changes sign after averaging its scene-dependent interactions, and `spatial_balanced` remains the strongest bounded factor with positive effects on both capped scenes. Pipes and retained private-225 are explicitly below their point caps, so Phase 3 effects there are inactive rather than gains. Covisibility, `contradiction_free`, pixel-center intrinsics, and Manhattan were not jointly generated with the factorial; their interactions remain unrun/not estimable instead of being inferred. The frozen report is deterministic and preserves all raw arms and mixed outcomes. Production defaults and contracts remain unchanged; G1.26 owns the final baseline selection.
- G1.26 completes the Stage 1 gate review without promoting a candidate. The mandatory Gate G1 combination—quantified private-ROI improvement without unexplained systematic regression across the three ETH3D scenes—is not satisfied by any retained candidate: grouping and Phase 1/2 transfer are mixed, Phase 3 lacks an active non-ETH3D result, contradiction-free and pixel-center intrinsics failed their transfer gates, and Manhattan is private-only rigid orientation evidence. The existing `sequential + points + global + any_support + random` configuration (confidence percentile 50, 2M point cap) is frozen only as the reproducible fallback; runner, adapter, optional API fields, frontend defaults, retained jobs, and assets remain unchanged. `docs/manifest-schema.md` records effective-policy metrics and backward compatibility for historical manifests that omit them. No reconstruction/evaluator/GT-guided operation or selector was run, arbitrary units remain non-metric, and G1.26 stays blocked rather than claiming a new improved geometry baseline.
- Root `plan.md` is the task-level execution checklist beneath this plan of record. Its revised product ordering is: integrated static 3DGS, company-portal productization/deployment, static evidence-aware semantics, then optional video/long-horizon/panorama extensions. All Stage 2 implementation is blocked by one consolidated `R2.0` confirmation round; failed or mixed experiments remain recorded rather than being silently removed.
- Stage 2 must implement 3DGS reconstruction, training-time validation, isolated test evaluation, hyperparameter/effective-config management, checkpoint/resume/attempts, export, and worker/manifest integration inside this repository. Nerfstudio/Splatfacto or another complete external training platform is not an allowed runtime. The old `register_gaussian_splat.py` path and Nerfstudio exports are retained only as legacy/reference fixtures.
- “Integrated in this project” does not silently mean every CUDA/WebGL kernel must be handwritten. PyTorch/CUDA, a pinned `gsplat`-style narrow rasterizer, native extension/container policy, optimizer primitive, browser renderer/format, and all license/build constraints are explicit unresolved `R2.0` decisions. No dependency or trainer work starts before confirmation.
- The target company product is a static image upload → queued GPU geometry/3DGS training → training validation/final test evaluation → integrity-checked export → browser view/download loop. Public users will receive versioned quality profiles; internal CLI/admin keeps full research controls. Stage 2 evolves the current synchronous local `JobStore` only into a backward-compatible filesystem/manifest lifecycle for one serial RTX 4060 worker.
- On 2026-07-29, `R2.0` was confirmed with an explicit ownership boundary: the current work is algorithm implementation on one RTX 4060 8GB. Stage 2 must manage a single GPU-heavy job at a time, Gaussian/resolution memory budgets, OOM behavior, checkpoint/resume, cancel/retry, and local restart recovery. It must not design a production database, Redis/distributed queue, object storage, retention/delete/backup, authentication/tenant/RBAC, signed downloads, company-server concurrency, deployment topology, SLA, or operations stack; deployment owners freeze those separately in Stage 3.
- On 2026-07-29, Stage 2A froze dataset contract v1, deterministic camera-spatial split seed `20260729`, and an immediate 12-view synthetic / 32-view private / 32-view ETH3D `terrains` development matrix. The contract hashes images/cameras, stores OpenCV world↔camera poses, preserves explicit distortion and arbitrary scale, and makes normalization reversible. The selected narrow dependencies are official binary `gsplat==1.5.3+pt23cu121` (Apache-2.0) and lockfile-resolved `@mkkellogg/gaussian-splats-3d==0.4.7` (MIT). The rasterizer passes RTX 4060 sm_89 forward/backward and a color finite-difference check with relative error `5.5668204e-05`; local CUDA 11.7 source compilation is intentionally bypassed by the pinned official wheel. The large Mip-NeRF 360 `bonsai` download remains deferred until local storage is cleared, and source-license approval remains required before a full public 3DGS benchmark claim.

- On 2026-07-30, the locally copied Mip-NeRF 360 `room` scene was added as the primary public indoor product proxy while `bonsai` was retained as the public fine-detail/literature reference. The official `room` COLMAP model registers all 311 views; its PINHOLE intrinsics were scaled independently to the 779×519 `images_4` resolution, then the frozen camera-spatial protocol selected 32 views and produced a 26/3/3 train/validation/test contract with dataset hash `f4beb11902d3f96d3c61aee48d237c7ad9f0856ecad45ff79f684e4cc155b39e`. `room` is one bounded arbitrary-scale indoor scene, not evidence for whole-home coverage, room connectivity, metric accuracy, or a production VR-tour guarantee.
- On 2026-07-30, R2.4 added project-owned Gaussian config schema v1. `standard_v1` is the only public quality profile; its complete effective config is resolved without hidden environment overrides, strictly validated, canonically hashed, and optionally persisted in manifest/log through an internal-only `JobStore` seam. Trusted research overrides may change only known leaves, and an ablation record must compare two complete valid configs that differ in exactly one leaf. Existing geometry/API/frontend behavior and legacy manifests remain unchanged. These are contract defaults rather than a promoted RTX 4060 performance profile: R2.5–R2.7 own checkpoint/worker/trainer consumption, and R2.14 still owns final resource-profile evidence.
- On 2026-07-30, R2.5 added project-owned attempt/checkpoint schema v1. Fresh, retry, and resume always create immutable non-overwriting attempt directories with explicit lineage; resume accepts only a fully committed parent checkpoint whose dataset, effective-config, code, and environment hashes all match, while retry deliberately loads no state. Checkpoints atomically publish mandatory opaque model, optimizer, scheduler, densification, RNG, and metric-history components through a same-filesystem temporary directory plus fsync and one rename, and loaders reject partial, temporary, malformed, tampered, or provenance-mismatched state. Retention selection freezes latest three periodic, best two validation, and final but deletes nothing. A deterministic CPU reference resumes exactly; real trainer/CUDA tolerance evidence remains for R2.7/R2.15. No worker lifecycle, PyTorch serialization, public API/frontend field, manifest state, or new dependency was introduced.
- On 2026-07-30, R2.6 added local lifecycle schema v1 and the minimum serial worker seam. `POST /api/jobs` now durably writes inputs, request, and a queued manifest before returning HTTP 202; one filesystem-leased FIFO worker executes at most one GPU-heavy job per output root. Lifecycle state, timestamps, structured errors, and fresh/retry attempt lineage are atomic and restart-visible; stale running work becomes explicit `worker_interrupted` failure rather than false success. Cancel is idempotent and terminates subprocess groups, retry is clean and capped at three total attempts, and failed/cancelled partial work remains diagnostic-only rather than entering manifest assets. The synchronous `JobStore.create_job` path and legacy terminal manifests remain readable. The research frontend polls and exposes stage/progress/error/cancel/retry. R2.6 added no 3DGS trainer, speculative evaluation/export schema, production database, Redis, object storage, authentication, or multi-worker scheduler; R2.7 remains the next core algorithm task.
- On 2026-07-30, Stage 2C added the first real project-owned 3DGS implementation: owned Gaussian tensors/optimizer groups/L1+SSIM/LR+SH schedules, gsplat-only rasterizer boundary, finite/OOM/cancel handling, densification/pruning/opacity reset, deterministic sparse and filtered-budgeted dense initialization, R2.5 checkpoint/resume serialization, validation-only metrics/progress, and the `project_3dgs` serial-worker adapter. Camera distortion is explicitly resampled before differentiable pinhole rendering because gsplat 1.5.3 distortion UT is non-differentiable. A 12-view CUDA smoke cancelled/resumed at iteration 20/40 and reduced loss 0.14070→0.09866. A same-config 100-iteration `terrains` validation ablation recorded mixed dense-minus-sparse evidence (+0.175 dB PSNR, -0.00191 SSIM, +9,484 Gaussians, +20 MiB reserved VRAM), so neither initializer is promoted. No test views or external full trainer were used; LPIPS remains explicitly not run pending R2.11 audit. R2.12 canonical/browser export and the final `room`/RTX/end-to-end gates remain open.

- Later-stage order is portal-first: Stage 3 owns auth/tenant isolation, durable metadata/queue/artifact storage, bounded production GPU scheduling, secure upload/export/retention/delete, observability, migration and rollback; Stage 4 adds static semantic object maps and evidence-aware scene graphs; Stage 5 optionally adds short-video incremental reconstruction, then independently gated long-horizon mapping and panorama support. Online SLAM is not a prerequisite for the static portal.
