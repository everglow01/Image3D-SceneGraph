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

### Stage 5: Optional bounded video and long-horizon extensions

Deliverables:

- Stage 5A bounded offline ingestion for one 10-second–10-minute video: streamed upload, orientation normalization, quality-aware keyframes, exhaustive COLMAP registration gates, temporally grouped dataset splits, and reuse of the Project/Graphdeco Gaussian contract;
- Stage 5B research-only `vggt_ba` Gaussian geometry for video: fixed bounded VGGT windows, strong/weak camera classification, one deterministic bounded recovery attempt per adjacent disconnect, ALIKED/VGGSfM local tracks and BA, reliable-component robust global Sim(3), then a partial VGGT seed through COLMAP 4 triangulation, omitted-image registration, and global BA. Ordinary COLMAP remains the default and is also an explicit late fallback only for the three classified post-recovery geometry-quality states; requested/effective source and fallback reason remain auditable;
- Stage 5C research-only `vggt_visibility_v1` postprocessing for either Gaussian geometry source: Train-only scale-aligned VGGT depth evidence, conservative free-space/capture-envelope filtering, immutable Original plus filtered A/B derivatives, and no hole filling, wall creation, retraining, Test use, or navigation change;
- separately gated tracking/relocalization, bounded windows/submaps, loop closure, pose graph, scale drift, dynamic masking/objects, and long-sequence evaluation;
- panorama-aware adapter candidate behind its own benchmark gate.

Success criteria:

- Gate G5A first proves the bounded offline video-to-Gaussian contract without using Test for extraction, registration gates, training, selection, or navigation. Stage 5B/5C remain explicit experimental options until ordinary end-to-end jobs demonstrate connected camera geometry, honest open/closed trajectory status, bounded resources, shared Project/Graphdeco contracts, and conservative filtering without systematic structure removal. Later near-real-time/long-horizon claims require independent evidence. Failure of Stage 5 does not invalidate the static portal.

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
2. Stage 2 static 3DGS Gate G2 is frozen: project trainer, isolated evaluation, deterministic canonical/browser export, serial worker lifecycle, measured RTX 4060 profile, and upload-to-view delivery are implemented. LPIPS remains an explicit audited not-run item because no approved local pretrained trunk exists; this does not silently change the metric definition.
3. Productize the static portal in Stage 3 only after the deployment owner supplies database/storage/retention/auth/company-infrastructure requirements.
4. Add static semantic object maps and scene graphs in Stage 4.
5. Treat short video, long-horizon mapping, and panorama as optional Stage 5 extensions.

Stage 2 must not introduce speculative production database, object storage, retention, auth, tenant, or deployment architecture. It may add only the minimum local async state and serial single-job GPU memory/queue controls required to develop, resume, evaluate, and export 3DGS reliably; one project-owned trainer job may use all visible GPUs as recorded in §17.

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
- Stage 2 must implement 3DGS reconstruction, training-time validation, isolated test evaluation, hyperparameter/effective-config management, checkpoint/resume/attempts, export, and worker/manifest integration inside this repository. A full external platform other than the explicitly selected Graphdeco research trainer is not an allowed runtime; the former exported-splat registration path was removed with the Nerfstudio integration.
- “Integrated in this project” does not silently mean every CUDA/WebGL kernel must be handwritten. PyTorch/CUDA, a pinned `gsplat`-style narrow rasterizer, native extension/container policy, optimizer primitive, browser renderer/format, and all license/build constraints are explicit unresolved `R2.0` decisions. No dependency or trainer work starts before confirmation.
- The target company product is a static image upload → queued GPU geometry/3DGS training → training validation/final test evaluation → integrity-checked export → browser view/download loop. Public users will receive versioned quality profiles; internal CLI/admin keeps full research controls. Stage 2 evolves the current synchronous local `JobStore` only into a backward-compatible filesystem/manifest lifecycle for one serial RTX 4060 worker.
- On 2026-07-29, `R2.0` was confirmed with an explicit ownership boundary: the current work is algorithm implementation on one RTX 4060 8GB. Stage 2 must manage a single GPU-heavy job at a time, Gaussian/resolution memory budgets, OOM behavior, checkpoint/resume, cancel/retry, and local restart recovery. It must not design a production database, Redis/distributed queue, object storage, retention/delete/backup, authentication/tenant/RBAC, signed downloads, company-server concurrency, deployment topology, SLA, or operations stack; deployment owners freeze those separately in Stage 3.
- On 2026-07-29, Stage 2A froze dataset contract v1, deterministic camera-spatial split seed `20260729`, and an immediate 12-view synthetic / 32-view private / 32-view ETH3D `terrains` development matrix. The contract hashes images/cameras, stores OpenCV world↔camera poses, preserves explicit distortion and arbitrary scale, and makes normalization reversible. The selected narrow dependencies are official binary `gsplat==1.5.3+pt23cu121` (Apache-2.0) and lockfile-resolved `@mkkellogg/gaussian-splats-3d==0.4.7` (MIT). The rasterizer passes RTX 4060 sm_89 forward/backward and a color finite-difference check with relative error `5.5668204e-05`; the accepted binary-wheel path bypasses local source compilation. The large Mip-NeRF 360 `bonsai` download remains deferred until local storage is cleared, and source-license approval remains required before a full public 3DGS benchmark claim.

- On 2026-07-30, the locally copied Mip-NeRF 360 `room` scene was added as the primary public indoor product proxy while `bonsai` was retained as the public fine-detail/literature reference. The official `room` COLMAP model registers all 311 views; its PINHOLE intrinsics were scaled independently to the 779×519 `images_4` resolution, then the frozen camera-spatial protocol selected 32 views and produced a 26/3/3 train/validation/test contract with dataset hash `f4beb11902d3f96d3c61aee48d237c7ad9f0856ecad45ff79f684e4cc155b39e`. `room` is one bounded arbitrary-scale indoor scene, not evidence for whole-home coverage, room connectivity, metric accuracy, or a production VR-tour guarantee.
- On 2026-07-30, R2.4 added project-owned Gaussian config schema v1. `standard_v1` is the only public quality profile; its complete effective config is resolved without hidden environment overrides, strictly validated, canonically hashed, and optionally persisted in manifest/log through an internal-only `JobStore` seam. Trusted research overrides may change only known leaves, and an ablation record must compare two complete valid configs that differ in exactly one leaf. Existing geometry/API/frontend behavior and legacy manifests remain unchanged. These are contract defaults rather than a promoted RTX 4060 performance profile: R2.5–R2.7 own checkpoint/worker/trainer consumption, and R2.14 still owns final resource-profile evidence.
- On 2026-07-30, R2.5 added project-owned attempt/checkpoint schema v1. Fresh, retry, and resume always create immutable non-overwriting attempt directories with explicit lineage; resume accepts only a fully committed parent checkpoint whose dataset, effective-config, code, and environment hashes all match, while retry deliberately loads no state. Checkpoints atomically publish mandatory opaque model, optimizer, scheduler, densification, RNG, and metric-history components through a same-filesystem temporary directory plus fsync and one rename, and loaders reject partial, temporary, malformed, tampered, or provenance-mismatched state. Retention selection freezes latest three periodic, best two validation, and final but deletes nothing. A deterministic CPU reference resumes exactly; real trainer/CUDA tolerance evidence remains for R2.7/R2.15. No worker lifecycle, PyTorch serialization, public API/frontend field, manifest state, or new dependency was introduced.
- On 2026-07-30, R2.6 added local lifecycle schema v1 and the minimum serial worker seam. `POST /api/jobs` now durably writes inputs, request, and a queued manifest before returning HTTP 202; one filesystem-leased FIFO worker executes at most one GPU-heavy job per output root. Lifecycle state, timestamps, structured errors, and fresh/retry attempt lineage are atomic and restart-visible; stale running work becomes explicit `worker_interrupted` failure rather than false success. Cancel is idempotent and terminates subprocess groups, retry is clean and capped at three total attempts, and failed/cancelled partial work remains diagnostic-only rather than entering manifest assets. The synchronous `JobStore.create_job` path and legacy terminal manifests remain readable. The research frontend polls and exposes stage/progress/error/cancel/retry. R2.6 added no 3DGS trainer, speculative evaluation/export schema, production database, Redis, object storage, authentication, or multi-worker scheduler; R2.7 remains the next core algorithm task.
- On 2026-07-30, Stage 2C added the first real project-owned 3DGS implementation: owned Gaussian tensors/optimizer groups/L1+SSIM/LR+SH schedules, gsplat-only rasterizer boundary, finite/OOM/cancel handling, densification/pruning/opacity reset, deterministic sparse and filtered-budgeted dense initialization, R2.5 checkpoint/resume serialization, validation-only metrics/progress, and the `project_3dgs` serial-worker adapter. Camera distortion is explicitly resampled before differentiable pinhole rendering because gsplat 1.5.3 distortion UT is non-differentiable. A 12-view CUDA smoke cancelled/resumed at iteration 20/40 and reduced loss 0.14070→0.09866. A same-config 100-iteration `terrains` validation ablation recorded mixed dense-minus-sparse evidence (+0.175 dB PSNR, -0.00191 SSIM, +9,484 Gaussians, +20 MiB reserved VRAM), so neither initializer is promoted. No test views or external full trainer were used; LPIPS remains explicitly not run pending R2.11 audit. R2.12 canonical/browser export and the final `room`/RTX/end-to-end gates remain open.

- On 2026-07-30, Stage 2D completed project-owned evaluation and delivery. Validation and standalone test evaluation share one renderer/metric core, while the trainer still cannot load test; test requires a dataset/config/model-hash-bound frozen candidate and creates an exclusive consumed record before loading views. Evaluation records PSNR/SSIM distributions, failures, timing/FPS, Gaussian/opacity/scale/topology/memory, previews, and a separate geometry-not-run state. LPIPS 0.1.4 was not installed because its default AlexNet trunk would download an ImageNet checkpoint without an approved local hash/weight-license record; output says `not_run` rather than substituting a metric. Canonical deterministic binary PLY, browser derivative, explicit normalized/arbitrary coordinate metadata, bounded camera path, and deterministic result bundle are project-owned; the existing audited 0.4.7 viewer parsed all exported rows. The internal-only measured RTX profile is 100 iterations/320px/20k sparse initial/50k cap (hash `a0b91841…`), completing room with 18,765 Gaussians and 94,371,840 peak reserved bytes; validation/test PSNR were 2.2271/2.3272 and SSIM 0.1894/0.2116, which proves lifecycle execution rather than high quality. Integrated ignored job `20260730_092826_4c1497f3` uploaded 32 room images and completed COLMAP→train→validation→one consumed held-out test→export→API download→viewer parse on its first attempt, producing 3,608 Gaussians. An earlier diagnostic job retained two failures that exposed a relative/absolute workspace symlink bug; attempt inputs are now copied inside the disposable workspace to satisfy dataset-root integrity, while original inputs remain preserved. Gate G2 is frozen for the single-machine algorithm-development scope; arbitrary units, Stage 1 Gate G1, production deployment, and high-quality profile claims remain separate.
- On 2026-07-31, the project-owned trainer contract advanced to Gaussian config schema v2 after a real 225-image job exposed a blank/collapsed export. Densification statistics now use gsplat screen units, growth distinguishes small-Gaussian duplication from local quaternion-frame splitting, Adam survivor state is remapped while new topology moments start clean, pruning reports causes and disables unverified raw-pixel radius pruning, opacity resets stop before densification ends and never run on the final iteration, images stream CPU→GPU one view at a time, and the frozen model is selected by Validation rather than blindly using the last iteration. Export rejects models without effective-opacity rows; the browser reads SH/opacity metadata and no longer displays missing `Content-Length` as zero. A serial 32-view room run at 3,000 iterations/640px/250k cap completed in 83.17 s with 1.095 GB peak reserved VRAM, Validation PSNR/SSIM 16.7151/0.5803, and healthy duplicate/split activity. The frozen model then consumed Test exactly once (8.8663/0.4438), without feeding Test back into selection. Two exports matched canonical/browser/bundle hashes and the audited browser loader decoded all 250,000 rows. This measured configuration became the initial public `standard_v1` and internal RTX 4060 development profile. After a stopped million-Gaussian frontend job accumulated 7.23 GB across eleven full cadence checkpoints, config schema v3 removed the obsolete checkpoint cadence, fresh training stopped writing periodic full checkpoints, and successful training publishes one final checkpoint; Validation candidates remain one overwrite-in-place model snapshot without Adam state. A subsequent 225-image job showed that this fast profile finished in 59.24 s with only 1.076 GB reserved, reached the 250k cap by iteration 1,000, and registered only 176 inputs under sequential COLMAP matching. The user-authorized quality-priority revision keeps schema v3 but raises the profile to 8,000 iterations/960px/350k, accepts up to 50k sparse seeds, extends densification through 4,000, and uses exhaustive COLMAP matching for Project 3DGS. A fresh frozen-room Validation-only run completed in 298.57 s with 1.508 GB peak reserved and PSNR/SSIM 22.8994/0.7884 versus the prior same-split 16.7151/0.5803; it retained only the final checkpoint. A subsequent user-requested capacity profile raised training to 15,000 iterations/1280px/600k, accepted up to 75k sparse seeds, and extended densification through 7,500. Its frozen-room Validation-only run completed in 774.79 s with 3.221 GB peak reserved, approximately 97–98% GPU utilization, and selected iteration 13,500 at PSNR/SSIM 25.5475/0.8244; only the final iteration-15,000 checkpoint was retained. Export now records robust scene framing for stable browser orbit/zoom. Held-out Test isolation, arbitrary-unit semantics, and the prohibition on metric claims remain unchanged.
- On 2026-08-01, a retained 225-image result exposed large circular ceiling/exterior floaters as a capped topology defect rather than a browser-decoding defect. The old 600k allocator allowed duplication to starve splitting after iteration 1,200, and screen-size pruning was disabled. Gaussian config schema v4 keeps the project-owned trainer and hard 600k cap while adding bounded split/duplicate allocation, early normalized screen-radius splitting at 0.05, cleanup above 0.15 screen radius or 0.1 normalized maximum 3D scale, 3,000-step opacity resets with refinement recovery pauses, Validation-before-reset ordering, cleanup at 7,500 and 13,500, and export-visible health diagnostics. The behavior was informed by Graphdeco 3DGS `54c035f…`, Nerfstudio `50e0e3c…`, and gsplat `2b902ff…`; no upstream trainer code or new runtime dependency was copied. On the retained scene, the accepted 13,500-cleanup/1,500-recovery model changed final Validation mean/P10 PSNR from 25.2788/20.2775 to 24.8390/20.2025 dB and mean SSIM from 0.8697 to 0.8791, while screen-radius violations above 0.15 fell 827→81 and high-opacity violations 586→76. It performed 58,950 split replacements, pruned 2,979 screen-oversized rows, retained 591,511 Gaussians, and used 3.135 GB peak reserved VRAM. The remaining 81 violations are an explicitly accepted post-cleanup recovery tradeoff, not complete artifact removal. The effective profile hash is `9287f537144bfef7c24503452db826ceb9e4b01201eda361070d6f7ebbde8efb`; only Train/Validation were loaded, the already-consumed held-out Test was not rerun, and normalized sizes remain arbitrary units.
- On 2026-08-03, the Gaussian browser preview stopped assuming normalized `+Z` is room vertical. Calibration-free COLMAP normalization provides translation and scale but does not make its arbitrary world frame upright, and the Gaussian median can sit away from the room's natural viewing center. Following Nerfstudio `50e0e3c…` COLMAP defaults without importing its Viser stack, the existing browser viewer now derives display up from averaged OpenCV image-up camera axes and derives its initial orbit pivot from averaged normalized camera centers in the exported `camera_path.json`. Reset/Top/Front/Side presets share that camera-derived basis, while missing or degenerate camera metadata falls back to the robust export frame and `+Z`. Canonical PLY coordinates and project-owned 3DGS training/evaluation/export remain unchanged; the upside-down `Flip` control was removed.
- On 2026-08-03, schema v5 replaced the accumulated custom 15k/600k/floater-cleanup profile with an official, reproducible 3DGS baseline. Project 3DGS now runs COLMAP with shared `OPENCV` intrinsics, exhaustive matching, global-BA tolerance `1e-6`, registered-image/point model selection, and COLMAP undistortion before constructing its pinhole dataset and sparse initialization. Training uses the pinned Apache-2.0 `gsplat==1.5.3` `DefaultStrategy` for signed-gradient duplicate/split, stochastic local splitting, opacity prune/reset, and optimizer-state remapping; it runs 30,000 iterations, promotes SH every 1,000, decays only position LR, keeps official constant non-position LRs, and shuffles cameras without replacement. The old hard cap, split quota, screen cleanup, late cleanup, and recovery pause are not layered underneath. Graphdeco `54c035f…` remains a non-commercial-license algorithm/CLI reference only; no trainer code was copied, and Nerfstudio `50e0e3c…` is not a runtime dependency. Validation selection, one final checkpoint, deterministic export, arbitrary units, browser contracts, and held-out Test isolation remain unchanged; existing consumed Test artifacts were not rerun. A completed public `room` Validation-only run used all 112,627 sparse points, took 2,255.04 seconds with 2.116 GB peak reserved CUDA memory, and selected iteration 7,000 at PSNR/SSIM 25.9971/0.8396 over iteration 30,000 at 25.2327/0.8056. Its selected model contains 1,008,582 Gaussians and exported a 250,129,924-byte browser PLY. Fixed renders are detailed and mostly free of circular blur but retain peripheral stretched artifacts, and the 32-view split does not directly test the ceiling; this is a usable baseline rather than a claim that the private ceiling defect is resolved. No Test view was loaded.
- On 2026-08-04, the experimental trainer comparison boundary was explicitly reopened without replacing the stable product default. `project_3dgs + gaussian_splat` now accepts one trainer ID—`project`, `graphdeco`, or `nerfstudio`—through CLI, API, manifest provenance, and frontend. The external arms are subprocess runtimes in separate pinned ignored environments: Graphdeco commit `54c035f7834b564019656c3e3fcc3646292f727d` is research/evaluation-only and requires explicit license acceptance; Nerfstudio commit `50e0e3c70c775e89333256213363badbf074f29d` is Apache-2.0. Native datasets are deterministically generated from one frozen project contract with identical image hashes, explicit Train/Validation/Test IDs, pinhole cameras, and sparse seeds. Graphdeco uses a minimal runtime split wrapper; Nerfstudio receives explicit filename splits, frozen normalization, and camera optimization off. External INRIA PLYs are converted into the project normalized snapshot, then use the same Validation evaluator, canonical/browser exporter, manifest roles, and Viewer. Integration and smoke cannot load Test. No comparison is authorized until all three short CUDA smokes show finite falling loss and the user reviews the exact comparison matrix. The machine had 29 GB free after approved cleanup, but external environment installation/smoke was still blocked because the NVIDIA driver was unavailable; installation commands are handed to the user after the driver is restored.
- On 2026-08-05, all three CUDA smokes passed and the user authorized a serial formal comparison on the Mip-NeRF 360 `room` data. The frozen single-scene protocol is recorded under ignored `outputs/experiments/gaussian-trainer-room-comparison-20260805/protocol.json`: 32 contracted 779×519 views, the existing 26/3/3 Train/Validation/Test split, all 112,627 accepted sparse points, seed `20260729`, 30,000 optimization steps, maximum SH degree 3, camera optimization off, and one common project Validation renderer at the native contracted resolution. The 30,000-step final model is the primary candidate for every arm; any trainer-native/best-Validation checkpoint is supplementary and cannot replace it in the primary table. Rasterizer, optimizer, camera order, background/pixel schedule, and topology policy remain native method behavior and are reported rather than rewritten. PSNR and SSIM distributions, wall time, sampled process RSS, native/measured GPU memory, Gaussian count, artifact sizes, render throughput, health, and failures are reported separately. Common LPIPS remains explicitly not run because its package and weight provenance are not frozen; this one-scene project split is not presented as directly comparable to the original Mip-NeRF 360 paper split or as a full benchmark-suite claim. Arms run one at a time with `MAX_JOBS=1`; OOM is recorded rather than silently granting one method a lower resolution. Test is not loaded during comparison or trainer selection and requires a later separately frozen candidate decision.
- On 2026-08-06, COLMAP execution was isolated from Ubuntu's non-CUDA 3.7 package through a pinned project-local COLMAP 3.9.1 build under ignored `external/colmap-cuda/`. Resolution order is explicit `IMAGE3D_COLMAP_BIN`, project-local CUDA build, then PATH fallback, shared by availability probes and both COLMAP runners. Project 3DGS requests CUDA SIFT extraction and matching on GPU index 0 while Mapper/bundle adjustment retain bounded CPU threads; this preprocessing occurs before Project/Graphdeco/Nerfstudio trainer dispatch. Logs identify the resolved executable/build and compute controls. CUDA SIFT acceleration does not change arbitrary-unit semantics, matcher coverage, held-out Test isolation, or justify metric-accuracy claims.
- On 2026-08-12, production COLMAP moved from 3.9.1/CUDA 11.7 to the isolated COLMAP 4.0.0 build under ignored `external/colmap-4-cuda/`, compiled for RTX 4060 SM 89 with CUDA 12.2, ONNX Runtime 1.24.1, and cuDNN 9.25. The runners were updated for COLMAP 4's `FeatureExtraction.*` and `FeatureMatching.*` GPU/thread options while retaining SIFT extraction, brute-force matching, incremental Mapper, exhaustive matching for the Gaussian baseline, global-BA tolerance `1e-6`, and arbitrary-unit semantics. A Train-only 40-image end-to-end Gaussian preprocessing smoke used zero Validation/Test images, registered 40/40 images, produced 4,838 sparse points, undistorted to PINHOLE, and passed the existing camera/point-cloud input contract. ALIKED, LightGlue, and Global Mapper remain experimental and require controlled Validation ablations; Test remains unavailable for feature, matcher, mapper, threshold, or default selection. Graphdeco's reproducible setup moved from PyTorch `2.0.1+cu117` to `2.3.1+cu121` and explicitly compiles `simple-knn`, `diff-gaussian-rasterization`, and `fused-ssim` with `/usr/local/cuda-12.2`; isolated builds of all three linked `libcudart.so.12`, and real simple-KNN plus rasterizer forward/backward CUDA checks passed. The prior Graphdeco cu117 environment also passed with the system CUDA 11 runtime hidden because its PyTorch wheel carries CUDA 11 runtime, but it is no longer the setup contract. This evidence authorizes retirement of COLMAP 3.9.1 and `/usr/local/cuda-11.7` after explicit destructive-action confirmation; it does not authorize deleting retained jobs/evidence or changing learned-feature defaults.
- On 2026-08-12, Project `standard_v1` advanced to Gaussian config schema v7 by applying the bounded conclusion of the frozen `room` 2×2 initialization × screen-pruning factorial. With screen pruning disabled, 3NN RMS reached Validation PSNR/SSIM `26.7929/0.8456` at 10k versus legacy `26.4699/0.8423`; enabling the 20-pixel rule caused an immediate roughly 83k-Gaussian deficit at 3.1k under either initializer and remained the dominant schema-v6 regression cause. V7 therefore retains 3NN RMS, identity quaternions, opacity LR `0.025`, Train-camera extent scaling, render clamp, opacity/world-scale pruning, 30k/1280px/SH3 budget, eight Validation observations, and gsplat `DefaultStrategy`, while explicitly disabling only destructive training-time screen-radius pruning. Screen health remains diagnostic. New frontend Project jobs export the Validation-selected model without loading Test or creating `*.test-consumed.json`; the standalone frozen-candidate Test evaluator remains available only for a separately authorized final evaluation. Graphdeco stays the frontend default and retains its restricted research/evaluation license until cross-scene Validation and user visual review justify promotion. This one-scene result does not authorize early stopping, a hard Gaussian cap, general trainer-superiority claims, or metric units.
- On 2026-08-14, the server execution boundary expanded from one RTX 4060 to one serial project-owned trainer job spanning all visible NVIDIA L2 GPUs through gsplat distributed rasterization. Initial Gaussians, optimizer state, and `DefaultStrategy` topology state are rank-local; each rank contributes one Train view per optimizer step, Validation renders collaboratively on every rank, rank 0 alone publishes progress/artifacts, and best-model shards merge back into the existing single-model/export contract. Distributed checkpoints retain rank-local model/optimizer/strategy/RNG/camera-order state and require the same world size on resume. COLMAP no longer pins CUDA device 0 and may use all visible GPUs, while sparse extraction/mapping keep source resolution. The frontend exposes a config-hashed 1280/1920/2560/3072 px longest-edge choice for COLMAP-undistorted training images and Gaussian Train/Validation views; 1280 remains the frozen `standard_v1` default. Two 24 GB cards are not represented as one generic 48 GB CUDA device, distributed runs change the effective per-step camera batch and are not directly comparable to retained single-GPU v7 evidence, OOM is reported rather than silently lowering resolution, GPU-heavy jobs remain serial, Test remains excluded from training/selection/navigation, and geometry remains normalized arbitrary units.
- On 2026-08-14, the first full 273-image/3072px/two-L2 Project Job completed COLMAP but Linux killed one rank before the first optimizer event: the server had 31 GiB RAM plus 4 GiB swap, the killed rank had reached roughly 16.2 GiB RSS, `oom_kill` incremented, and no `progress.jsonl` existed. The failure was duplicated host residency, not pooled-GPU capacity or a CUDA OOM: every rank retained all Train and Validation RGB pixels as CPU float32. Undistorted CPU views now remain uint8 and only the selected view converts to normalized float32 during device transfer, reducing resident image bytes by 4× without changing pixels, splits, camera order, losses, Validation selection, checkpoints, effective-config hashes, or exports. This does not pool the two 24 GB GPUs, silently lower 3072px, guarantee every future scene fits CUDA memory, change the frozen 1280px default, authorize Test use, or change normalized arbitrary-unit semantics.

- On 2026-08-10, Project `standard_v1` advanced to Gaussian config schema v6 and directly adopted the first Graphdeco-alignment stage for the existing frontend-selectable `project` trainer: 3-nearest-neighbor RMS initial scale without global clipping, identity initial quaternions, opacity LR `0.025`, Train-camera extent scaling for position LR and world-space topology thresholds, a `[0,1]` render clamp before L1/SSIM, and 20-pixel screen-radius pruning from iteration 3,100 through 14,900 without enabling screen-size splitting. The 30,000-iteration/1280px budget, SH schedule, Validation-only model selection, final checkpoint/export contracts, arbitrary units, and Test isolation remain unchanged. Schema-v5 results remain immutable historical evidence; no training or Test evaluation was run while landing schema v6.
- On 2026-08-10, Stage 2.1 expanded schema-v6 Validation observations from 7,000/30,000 to 3,000, 5,000, 7,000, 10,000, 15,000, 20,000, 25,000, and 30,000. Every observation records nominal iterations, actual optimizer updates, current-attempt elapsed time, render metrics, Gaussian count, and topology/health diagnostics in the existing progress stream. Highest Validation PSNR still selects one overwrite-in-place model snapshot, while iteration 30,000 remains the sole retained full lifecycle checkpoint. This is observation and model-selection instrumentation only: it adds no early stopping, does not lower the 30,000-step budget, and cannot load or consume Test. No training was run while landing Stage 2.1.
- On 2026-08-10, the Stage 2.2 schema-v6 Project Validation-only trajectory completed on the frozen 32-view `room` contract (26 Train / 3 Validation / 3 untouched Test), all 112,627 sparse points, seed `20260729`, and a 30,000-step ceiling. The eight Validation PSNR/SSIM observations were: 3k `10.6753/0.2481`, 5k `12.2432/0.4848`, 7k `11.5762/0.4318`, 10k `11.1324/0.3601`, 15k `12.5068/0.4507`, 20k `20.7823/0.6837`, 25k `20.7715/0.6896`, and 30k `20.7430/0.6913`. Validation PSNR selected 20k with 662,453 Gaussians; the complete run took 1,233.26 seconds and peaked at 1.405 GB reserved CUDA memory. This materially regresses from retained schema-v5 best-Validation 7k (`25.9971/0.8396`) and Graphdeco final 30k (`26.2792/0.8294`), so it does not authorize a 7k or 20k early-stop default. Topology evidence shows a sharp 676,675→509,144→469,714 count collapse at 3.0k→3.1k→3.3k and recovery only after refinement stops. Test IDs `117/207/40` were never loaded, no Test artifact/consumption record was created, and no additional training was run during audit. Frozen protocol, complete trajectory, hashes, and audit remain ignored under `outputs/experiments/schema-v6-stage2-2-room-validation-trajectory-20260810/`.
- On 2026-08-10, a frozen Validation-only 2×2 factorial isolated the Stage 2.2 regression across initial scale (`3NN RMS` versus legacy clipped `1NN`) and screen-radius pruning (`20 px` versus disabled). It reused the completed 3NN+20px cell and ran only the three missing 30k cells serially with identical `room` data, points/colors, seed, resolution, budget, and Validation cadence. Best PSNR/SSIM was 3NN+20px `20.7823/0.6837`, 3NN+off `26.7929/0.8456`, legacy+20px `21.9523/0.7233`, and legacy+off `26.4699/0.8423`. Disabling screen pruning retained roughly 83k additional Gaussians immediately at 3.1k under either initializer and improved final PSNR by 5.5888 dB under 3NN and 4.0938 dB under legacy. With pruning disabled, 3NN was slightly better than legacy at the common 10k observation (`+0.3231` dB PSNR, `+0.0033` SSIM). Common-view previews independently showed detailed screen-off renders and blur/stretch/opaque smearing in both screen-on arms. The bounded causal decision is that the 20-pixel rule—not 3NN initialization—was the dominant schema-v6 regression cause on this scene. No public profile or early-stopping policy was changed from this one-scene result; Test remained completely untouched. Complete ignored evidence is under `outputs/experiments/schema-v6-screen-init-ablation-20260810/`.

- On 2026-08-10, near-term product direction was reprioritized and Project↔Graphdeco alignment/optimization was paused. `project_3dgs + gaussian_splat` retains only `graphdeco` and `project`, with pinned Graphdeco now the CLI/API/frontend default until Project quality work resumes; Graphdeco remains restricted to its research/evaluation license. The isolated project-local `external/nerfstudio/` checkout, Splatfacto command/dataset/export paths, frontend/API options, setup path, legacy imported-splat backend, and registration script were removed. Historical retained comparison and Test-consumption evidence remains immutable under ignored `outputs/`; no experiment asset was deleted or rewritten.
- The frozen next-work priority is: (1) first-person scene traversal with keyboard movement, pointer-lock mouse look, real-time novel-view rendering, collision volume, explicit walkable boundaries, and no escape from the reconstructed scene—the exact collision/boundary contract is specified before implementation; (2) bounded offline portrait-video reconstruction for up to ten minutes and 800 quality-aware keyframes, while robust cross-room loop closure, global pose optimization/bundle adjustment, and bounded-drift claims remain separately gated; (3) 3DGS robustness and quality under difficult lighting, specular/transparent glass, and blurred glass; (4) batched VGGT-BA integration and controlled experiments; (5) resume Project trainer changes using the retained screen-pruning factorial evidence. Product work must not claim metric dimensions while the scene remains normalized arbitrary units.
- On 2026-08-11, Phase 2 productized the reviewed Train-only Gaussian navigation builder without changing customer-visible geometry. Completed `project_3dgs + gaussian_splat` jobs now attempt navigation after Gaussian export and publish the stable roles `collision_mesh`, `navigation`, and `navigation_diagnostics` together only after normalized/arbitrary coordinate, source hash, Train split, selected-render subset, empty Validation/Test usage, GLB integrity, ≤50k triangle, ≤10 MiB, ≤5-minute, non-self-intersection, manifold-with-boundary, and orientability gates pass. Navigation failure is fail-soft: the Gaussian job remains `done` and retains `scene_splat`, while `navigation_status: unavailable` records a stable reason. Existing successful Gaussian jobs use idempotent `POST /api/jobs/{job_id}/navigation-assets`; it never retrains, runs behind the same one-worker filesystem lease, stages each attempt under `lifecycle/navigation/`, atomically renames one complete directory, quarantines interrupted/partial publication, and supports cancellation/retry independently of the immutable Gaussian result. Navigation assets remain outside the existing job ZIP, whose copied manifest is sanitized so it does not reference omitted navigation files. Geometry remains normalized arbitrary units and no metric claim is introduced.
- On 2026-08-11, Phase 3 connected those assets to the desktop Gaussian viewer. Old completed Gaussian jobs expose an explicit frontend generation action and poll navigation lifecycle state; an `available` set is strictly parsed as normalized arbitrary-unit, Train-only data, and the browser verifies collision byte count, SHA-256, and triangle count before building the installed Three.js `Octree`. The collision GLB stays hidden in customer mode while a `Capsule` uses fixed-step WASD/arrow movement, Pointer Lock mouse look, collision penetration resolution, and the hard concave boundary; Esc preserves the safe pose and Reset returns to the validated spawn. Speed (`0.4–1.2H/s`), FOV (`50°–90°`), and sensitivity (`0.0005–0.005`) persist only after validation. Debug collision/boundary/capsule/contact rendering is opt-in. Missing or invalid navigation leaves Orbit viewing intact and disables Walk. The retained Graphdeco job `20260806_060729_a5d1d377` publishes the verified 11,337-triangle, 156,584-byte Train-only collision set without retraining or new Test consumption. Single-level standing movement remains normalized arbitrary units; jump, crouch, sprint, mobile, gravity gameplay, and multi-floor traversal remain deferred.
- On 2026-08-17, Stage 5A added a bounded offline video front end to the existing Project/Graphdeco Gaussian pipeline. One MP4/MOV/M4V/WebM from 10 seconds through a public 10-minute limit (606-second technical tolerance), up to 2 GiB, is staged in 8 MiB chunks and probed with external FFmpeg/ffprobe. `video_keyframes_standard_v1` physically applies quarter-turn orientation, scores at 12 fps without retaining full candidate frames, and selects `clamp(round(duration_seconds * 6), 24, 3,636)` keyframes from at most 7,272 candidates. Generated JPEGs use Orientation 1 and truthful Software metadata; source PTS, quality/rejection reasons, hashes, and probe metadata remain in stable sidecars. Exhaustive COLMAP must register at least 12 frames, 70% of selected frames, and 80% temporal coverage before Gaussian training. Registered frames split by indivisible two-second temporal groups; Train/Validation behavior and Validation-selected export are unchanged, Test remains unavailable to extraction/registration/training/selection/navigation, and no Test-consumption record is created. Missing FFmpeg disables only video ingestion. This is bounded offline arbitrary-unit reconstruction, not realtime SLAM, metric scale, guaranteed loop closure, or drift-free ten-minute mapping.
- On 2026-08-17, Stage 5B/5C entered the integrated system as research-only experimental controls rather than new trainers or standalone smoke pipelines. `gaussian_geometry_source` keeps `colmap` as the default and adds video-only `vggt_ba`: DINOv2 proposes bounded nonlocal bridge windows; fixed 8/4 VGGT windows use ALIKED queries, VGGSfM tracks, local PyCOLMAP BA, robust shared-camera Sim(3) edges, and one anchored global window graph; merged VGGT cameras initialize the existing COLMAP 4 SIFT/exhaustive `point_triangulator` and global `bundle_adjuster`. The initial integration had no geometry fallback; the 2026-08-18 recovery decision below supersedes that behavior. Absence of verified nonlocal evidence is published as `open_trajectory_unverified`, not a loop/drift claim. After the ordinary temporal split, only sparse points with at least two Train observations initialize Gaussians and their RGB is resampled only from Train observations, while all-image BA remains permitted for camera/point positions. Independently, `gaussian_postprocess=vggt_visibility_v1` can follow COLMAP or VGGT-BA geometry: after immutable Original Validation/export, it recomputes depth from at most 64 Train views and conservatively derives a row-aligned mask for unsupported multi-view free-space floaters and capture-envelope-exterior oversized Gaussians. It creates no geometry, fills no holes, changes no navigation, and uses no Validation/Test to decide removal; filtered Validation is reporting-only. Original remains the stable `scene_splat`, the filtered derivative receives separate hashes/evaluation/export and Viewer A/B roles, and any derivative failure is `unavailable` without failing Original. The implementation pins VGGT/DINOv2/LightGlue revisions and weight hashes with no runtime downloads, but real CUDA/PyCOLMAP-COLMAP4 compatibility, resource bounds, cross-scene camera quality, threshold safety, and license review remain open before any promotion. Coordinates remain normalized arbitrary units, Test remains untouched, and no new Test-consumption record is authorized.
- On 2026-08-18, real six-minute video evidence (`base-0003` support `[2097, 2033, 2196, 2017, 0, 0, 9, 2027]`) replaced the all-frame local VGGT-BA gate with bounded weak-frame tolerance. Frames with at least 32 final reliable observations are strong; weaker frames are excluded from local BA and Sim(3) evidence rather than terminating the Job. Adjacent disconnects receive one deterministic recovery attempt of at most eight reliable frames with at least three from each side, while existing nonlocal bridges remain bounded. Only a finite, non-worsening connected component passing 12 reliable cameras, 70% support, and 80% temporal coverage seeds a partial COLMAP model; `point_triangulator` is followed by `image_registrator` for omitted frames and one global `bundle_adjuster`. Shared SIFT/exhaustive features and matches are reused by ordinary Mapper only for `vggt_graph_unusable_after_recovery`, `vggt_seed_geometry_insufficient`, or `vggt_registration_gate_failed`. OOM/CUDA, dependencies/checkpoints, non-finite values, corrupted input, I/O, cancellation, unexpected exceptions, and unclassified subprocess failures never fallback. Requested/effective geometry, fallback boolean/reason, recovery/component diagnostics, and final registration metrics are persisted; a fallback Job is viewable but excluded from successful VGGT-BA A/B evidence. `open_trajectory_unverified` alone never triggers fallback. Effective VGGT-BA alone receives Train-supported sparse filtering; fallback follows ordinary-COLMAP initialization. The policy changes no video bounds, serial GPU execution, normalized arbitrary units, research/license status, Validation/Test isolation, navigation geometry, or Test-consumption authorization.
- On 2026-08-18, `video_keyframes_standard_v1` candidate sampling density tripled from 4 fps to 12 fps, with the candidate cap scaling from 2,424 to 7,272 (12 × the 606-second technical tolerance) so the full duration range stays admissible. The selected keyframe budget `clamp(round(duration_seconds * 2), 24, 800)`, scoring/rejection logic, registration gates, and all other video bounds are unchanged: denser candidates only widen the pool the same selector picks from, trading decode/scoring time for sharper per-bucket frame choices.
- On 2026-08-18, the selected keyframe budget tripled from `clamp(round(duration_seconds * 2), 24, 800)` to `clamp(round(duration_seconds * 6), 24, 3,636)`, with the cap derived as 6 × the 606-second technical tolerance the same way the candidate cap is 12 ×. Candidate sampling, scoring/rejection logic, registration gates, temporal-group validation splitting, and all other video bounds are unchanged; the cost is denser COLMAP exhaustive matching and longer Gaussian training, accepted deliberately for higher view density on long videos.
- On 2026-08-19, Project Gaussian video jobs gained an experimental `colmap_matcher=sequential` opt-in (form field, `IMAGE3D_GAUSSIAN_COLMAP_MATCHER` fallback) that runs COLMAP `sequential_matcher` with loop detection against the Flickr100K vocab tree, replacing the O(N²) exhaustive pair count for the now-denser keyframe sets. The default remains `exhaustive`; sequential is video-only, requires the git-ignored `external/colmap-vocab/` tree installed by `scripts/setup_colmap_vocab_tree.py` (dry-run by default, SHA-256 pinned, no runtime download), and fails fast with the setup command when the tree is missing rather than silently falling back, because sequential matching without loop closure splits revisiting walks into models `find_largest_sparse_model` would otherwise discard quietly. Registration gates, feature extraction, mapping, undistortion, initialization, point-cloud/`colmap_vggt`/VGGT-BA paths, and all video bounds are unchanged; promotion to the video default requires controlled A/B evidence (matching time, registration rate/coverage, sparse point counts, training quality) against the exhaustive baseline.
- On 2026-08-19, the Gaussian splat viewer gained a display-only opacity-threshold slider (orbit mode, top-right panel) that injects a runtime `alphaThreshold` discard uniform into the installed splat material fragment shader so low-opacity floater haze can be hidden without reloading the asset. The slider floor/default is the export's `viewer_minimum_opacity`; job assets, manifests, training, evaluation, and the Walk pipeline are untouched, and the control hides itself if the shader layout is unrecognized.
- On 2026-08-19, a floater investigation on the frozen num4_room baseline (job `20260818_095134_33882c23`, final 2,403,516 gaussians, 88% born during densification: ~464k faint haze at opacity 0.005–0.05 hugging SfM surfaces, ~32k oversized blobs, and high-opacity ≥0.25 glass/reflection clouds) falsified all three cheap mitigation routes, and the temporary loss hooks added for the trial (`loss.opacity_reg` / `loss.scale_reg`) were reverted without landing. Post-hoc `vggt_visibility_v1` filtering removed only 44,566 gaussians (1.85%), all front-free-space contradictions, with train and perturbed novel views visually identical. Training-side mean-opacity regularization (0.01) raised haze from 19.3% to 37.0% because a constant downward gradient spreads surfaces into more faint gaussians; the binarizing o·(1−o) variant (0.01) left novel views worse (haze 23.6%, high-opacity 35.7%, veils/blur/spikes). Raising the pruning gate `opacity_threshold` 0.005→0.02 OOMed at iteration 14,599 with 3,814,854 gaussians: densification compensates removed coverage and the opacity reset floor (2×threshold) revives targets above the prune line, so the threshold is not a model-size or floater-count lever. Conclusion: the visible clouds are training-consistent content (reflections/haze) for this scene; the remaining candidates — distortion loss (requires a rasterizer swap) and input-side COLMAP gate changes (strict reprojection, relaxed track count) — are undecided and need explicit approval.
- On 2026-08-20, the first remote `colmap_matcher=sequential` run (`20260820_021004_31413ac1`) died in `sequential_matcher` with SIGABRT; a local repro on the same COLMAP 4.0.0 build traced it to a format break, not a corrupt download. COLMAP switched its visual index from flann to faiss in May 2025, and the legacy demuc.de Flickr100K tree fails `VisualIndex::Read` with "Failed to read faiss index". `scripts/setup_colmap_vocab_tree.py` and `resolve_colmap_vocab_tree` now pin the official faiss tree from COLMAP release 3.11.1 (`vocab_tree_faiss_flickr100K_words256K.bin`, 72,412,636 bytes, SHA-256 `96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc`), verified end-to-end locally by running loop-detection sequential matching against a cached ETH3D database.
- On 2026-08-20, `scripts/analyze_gaussian_floaters.py` landed as the frozen floater census that gates every mitigation experiment (SOR postprocess, opacity-reset fix, absgrad, MCMC). It is axis-free: each gaussian's nearest distance to the SfM cloud and to camera centers, both in colmap_world (positions via the sibling export.json `world_from_normalized`; scales stay in the normalized training frame), with hug/free radii calibrated from the seeded 50k-subsample SfM neighbor spacing (median / p90, `RandomState(0)`); free space = dSfM > nn-p90 AND dCam > 1.0 world unit. Two frozen reference censuses. `20260819_011254_d62fbce0` (12fps video baseline, 2,662,630 gaussians, nn 0.0253/0.0673): haze 676,307 (25.4%, hug 35.3%), core 901,292 (hug 57.4%, free 14.0%), thick 1,085,031 (hug 70.3%, free 8.7%), free-space haze 202,879 (30.0%, scale median 0.0053, veils >0.01: 25,905 / >0.03: 3,186). `20260818_095134_33882c23` (num4_room baseline, 2,403,516 gaussians, nn 0.0219/0.0579): haze 569,465 (23.7%, hug 23.1%), free-space haze 231,850 (40.7%, scale median 0.0052, veils 27,797 / 3,284). Both jobs confirm opacity anti-correlating with distance to the SfM surface; the earlier ad-hoc 231,977 free-haze figure for d62fbce0 was a no-camera-constraint variant (34.3%) and is superseded by the canonical 202,879.
- On 2026-08-20, Stage 1 (post-hoc SOR floater cleanup, zero retraining) ran to completion. Tools: `scripts/filter_gaussian_sor.py` (profile `sor_v1`, Open3D `remove_statistical_outlier`; `full` removes every flagged gaussian, `band` additionally protects flagged gaussians with activated opacity ≥ `--band-opacity`; 50% removal safety gate), `scripts/import_gaussian_ply.py` (canonical PLY → evaluable `.pt`), `scripts/evaluate_gaussian_simple.py` (validation-split render eval without a resolved-config record), and `write_binary_ply` promoted to public in `gaussian/export.py`. Two plan deviations, both forced by what was on disk: the planned `--model-snapshot` filter mode was dropped because both frozen jobs already export canonical PLYs, and room evals bypass `run_evaluation` because the frozen room config is schema 5 while current code requires 7. Census sweep (3 presets × 2 variants × both frozen jobs, deterministic — in-memory mask re-run reproduces the sweep count exactly): the `band` variant strictly dominates `full` — identical free-space haze reduction with zero core/thick touch (core hug fraction Δ = +0.00, thick free-space count unchanged), while `full`'s extra removals are exactly the isolated core/thick gaussians. Absolute SOR effect is modest: aggressive-full (k=10, σ=1.0) removes only 13.1% (d62fbce0: 202,879 → 176,372) and 7.8% (33882c23: 231,850 → 213,777) of free-space haze, because clustered free-space haze is not position-isolated; large veils respond better (conservative preset removes 31% / 22% of veils >0.01). Room render evidence (official-baseline-room-v7; baseline reproduced bit-exact at PSNR 25.9971 / SSIM 0.8396): `full` collapses quality — conservative −2.81 dB, balanced −3.66 dB, aggressive −4.97 dB (21.03) — because SOR-flagged isolated gaussians include legitimate thin-structure content; `band` is render-lossless — conservative/balanced/aggressive give PSNR 25.9549 / 25.9514 / 25.9432 (Δ ≤ 0.054 dB, SSIM +0.0004..+0.0005, noise-level) while removing 3,174–5,883 gaussians. Verdict: `full` rejected; `conservative band` (k=30, σ=2.0, band 0.05) is the only lossless configuration and the only candidate for promotion (on the frozen jobs it removes ~18k / ~11k low-opacity haze including ~6k–8k veils with zero high-opacity damage). SOR is a light auxiliary cleanup, not the main mitigation — position-space isolation tests cannot see clustered free-space haze, so the primary attack remains training-side (Stage 2 opacity-reset / absgrad). Promotion decision left to the user. Artifacts: `outputs/experiments/sor-floater-cleanup-v1/` (filter records + masks + census for all 12 sweep cells; the sweep `filtered.ply` files were deleted as regenerable with source hashes recorded; room keeps `filter/`, `model.pt`, `eval/` per variant). Also fixed: `import_inria_ply` saved the full 62-column base matrix as storage for each of the five state tensors (views → ~5× `.pt` bloat, 1.15 GB for a 1M-gaussian model); now `np.ascontiguousarray` per tensor (~230 MB).
- On 2026-08-20, Stage 2a landed config schema 8 (7→8, leaf-only addition following the b7f2a7f checklist precedent): `opacity_reset.floor_multiplier` (default 2.0 = behavior-preserving, validated in (0, 10] with cross-check `opacity_threshold × floor_multiplier < 1.0`), and `trainer.py` now reads `threshold × floor_multiplier` instead of the hardcoded 2.0 reset value — the contract already required reset to be configured, so the hardcode was the debt. Schema 8 applies to all new jobs; defaults reproduce prior behavior exactly. The room A/B (`outputs/experiments/opacity-reset-floor-v1/`, protocol frozen before execution, single-leaf certified by `assert_single_field_ablation`, same seed/contract/init, pre-reset trajectories matched within noise at iter 3000) FAILED its quality gates: baseline floor 0.01 → PSNR 26.8018/SSIM 0.8463 (952,309 gaussians); fix floor 0.0045 → 26.4937/0.8422 (848,581). Deltas: PSNR −0.308 dB and SSIM −0.0041 versus gates ≥ −0.1 dB / ≥ −0.002; free-space haze −12,155 (−10.0%, gate passed) but core count fell −61,741 (−13.1%) and veils rose (+6.5% / +29%). The mechanism is real — post-reset the fix arm pruned ~91k more by iter 3100 — but only ~12k of the ~104k removed were free-space floaters; the rest are supported-but-slow-recovering content pruned inside the unusable recovery band. Two permanent discoveries: (1) the gsplat rasterizer has an empirical opacity cutoff in (0.0039, 0.0045) — uniform-opacity render test on the baseline model gave 0 visible gaussians at 0.0025/0.0039 and 831,328 at 0.0045, and the original fix arm at floor 0.0025 died at iter 3000 with the trainer guard "no training view sees any initialized Gaussian" (retained as evidence under `fix-floor-0.5/`); the reset floor's usable band is therefore only (≈0.004, 0.005), leaving no recovery margin below the 0.005 prune line; (2) the schema-8 baseline arm itself scores 26.80/0.8463 versus frozen v5's 25.9971/0.8396 (schema-gap effect: opacity LR 0.05→0.025, clamp_render, validation schedule), so arms compare against each other only. Verdict: 2a in minimal single-leaf form is falsified on room; the planned remote frozen-job retrain is not warranted. Remaining training-side candidates each need separate approval: absgrad (2b), reset+prune timing redesign (multi-field), MCMC relocation (stage 3). Code/schema-8 changes remain uncommitted alongside Stage 1.
- On 2026-08-21, the Stage 1 SOR band filter (conservative preset only: k=30, σ=2.0, band 0.05, 50% removal gate) landed as a DEFAULT pipeline step for `project_3dgs + gaussian_splat`, per user directive — the first opt-out style option in the repo. Semantics are in-place by user decision: the filter runs on the selected snapshot BEFORE validation eval (`ProjectGaussianAdapter.run`, progress 0.70), so `scene_splat`, metrics, export, and any VGGT derivative all see the filtered model; export threads the filter record/mask through the existing `--postprocess-record/--postprocess-mask` hash chain. Fail-soft: on any failure the original model proceeds with status `unavailable` + reason. Kill switches: form field `gaussian_sor_filter` (frontend select 「SOR 浮点清理」), jobs.py option validation, and env `IMAGE3D_GAUSSIAN_SOR_FILTER=off` resolved at enqueue (form > env > default on); the API form field defaults to None so the env switch applies to jobs omitting it. `filter_gaussian_sor.py` gained a `--model-snapshot` input mode reusing the VGGT filter's row-indexed `.pt` payload; manifest gains the `gaussian_sor_filter` triplet (documented in `docs/manifest-schema.md`). Composes deterministically with `vggt_visibility_v1` (SOR first, then VGGT derivative of the SOR model; statuses independent). Uncommitted.
- On 2026-08-21, config schema 9 (8→9, leaf-only) added `opacity_reset.recovery_prune` (`enabled` default false, `window_iterations` 500, `opacity_threshold` 0.05; validation: positive window ≤ iterations, threshold in (0,1), and threshold > `opacity_threshold × floor_multiplier`), and the trainer fires a one-shot `gsplat.strategy.ops.remove` of `sigmoid(opacity) < threshold` at `reset_iteration + window` for each opacity reset (default: 3500/6500/9500/12500), emitting `recovery_prune` progress events with before/after counts. The room A/B (`outputs/experiments/recovery-gated-prune-v1/`, protocol frozen before execution, single-leaf certified: only `recovery_prune.enabled` differs) PASSED all gates: baseline 26.7433/0.8459 (953,200 gaussians) vs fix 26.8944/0.8440 (641,026): PSNR +0.151 dB (gate ≥ −0.1), SSIM −0.00196 (gate ≥ −0.002, passes by 0.0000384 — borderline), free-space haze −104,825 (−87.8%), total haze −88.9%, veils −89.8%/−80.4% (the healthy direction; 2a saw veils rise), core hugging stable (0.2813→0.2758). The four prunes removed 510,412 gaussians (141,838/172,494/120,825/75,255, shrinking as expected) while densification regrew supported content. Why this works where 2a failed: the floor stays rasterizer-safe at 0.01 and separation is by post-reset recovery SPEED with a wide 500-iteration window and a 5×-floor threshold, instead of squeezing the floor into the unusable (≈0.004, 0.005) band. Two caveats on record: (1) the determinism cross-check did NOT bit-reproduce the 2a baseline (26.7433 vs 26.8018, −0.059 dB; in-trainer events also diverged) — gsplat rasterization has ≥ ~0.06 dB run-to-run GPU nondeterminism, so the +0.151 dB gain (~2.5× the noise band) is the meaningful signal and arms compare within-experiment; (2) the fix arm was checkpointed at iteration 11,596 by a planned shutdown and resumed via the proven `--attempt-kind resume` path (RNG/optimizer/camera cursor restored; all four prune events present). Promotion decision belongs to the user (candidates: floater-rich-scene confirmation then default-on; video-job test; combine with the SOR pipeline step — mechanisms are independent). Uncommitted alongside the SOR pipeline changes. (Both were committed later the same day as 775ad5d.)
- On 2026-08-21, the user directed a pre-validation rebalance of the video front end and the recovery-prune promotion path. (1) `video_keyframes_standard_v1` bounds changed: candidate sampling 12→6 fps (`CANDIDATE_FPS`, cap 7,272→3,636 keeping the fps×606s derivation) and the selected-keyframe cap 3,636→1,000 (`MAX_KEYFRAMES`); the 6/s selection rate, MIN 24, scoring/rejection logic, and registration gates are unchanged. For the 6-minute validation video this moves selection from ~2,160 to exactly 1,000 keyframes (cap now binding above ~167 s), and historical video jobs (`d62fbce0`, 12fps/2,160-class) stop being directly comparable references. (2) `gaussian_recovery_prune` landed as an experimental job option (form field, `IMAGE3D_GAUSSIAN_RECOVERY_PRUNE` env fallback, default `off`): when on, enqueue resolves `standard_v1` with the single override `opacity_reset.recovery_prune.enabled=true` (record hash computed after override), the manifest echoes `gaussian_recovery_prune`, and the trainer/adapter are untouched. No frontend switch until promotion. (3) The planned next run is a SINGLE-ARM remote video job on the 6-minute video (`gaussian_trainer=project`, `gaussian_geometry_source=colmap`, `colmap_matcher=sequential` after the successful B-group vocab-tree run, `gaussian_recovery_prune=on`, default SOR on, no VGGT postprocess). Recorded limits: single arm gives census/visual confirmation only, not a promotion-grade delta (a same-video paired baseline arm remains required for any default promotion); the haze reduction is not attributable between SOR and recovery_prune in one job; the fps/cap change breaks comparability with the frozen 30.0%/40.7% reference censuses.
- On 2026-08-21, a registration-hole audit of frozen job `d62fbce0` (400.8 s, 2,405 selected frames) root-caused the user-reported "some viewpoints render badly" in video scenes: the selection layer was clean (max selected-frame gap 1.10 s), but COLMAP dropouts clustered into three registration holes at 296.9–299.0 s (2.06 s), 305.3–310.1 s (4.82 s), and 330.8–338.2 s (7.42 s with ZERO registered frames; 66/90 candidates there passed quality checks, so the holes are scene-caused weak texture + motion blur — a registration-side scene limit, explicitly not fixed here). The split compounded the holes by design: `deterministic_temporal_group_split` picks holdout groups spatially farthest-first and its holdouts clustered in the same 270–358 s region, dropping training density there to ~1/3 of the median. Two fixes landed per user decision (SOFT semantics, explicitly not a hard gate): (1) soft gap warning in `_write_video_registration_diagnostics` — module constant `MAX_REGISTERED_GAP_SECONDS = 2.0` (the split's temporal-group granularity: a larger gap empties a whole 2 s group); the diagnostics payload gains `maximum_registered_gap_threshold_seconds` + `gap_violations`, metrics gain `video_registration_gap_violation_count` (only when non-empty), and run.log gains one `video_registration_gap_warning=N max=X.XXs intervals=...` line; the job continues and the 12/70%/80% gates are untouched. (2) Split hole avoidance (function signature unchanged): the groups adjacent on both sides of any >`group_seconds` input-timestamp gap are excluded from holdout eligibility, `heldout_count` is re-clamped against the eligible count, behavior is bit-identical when there are no holes, and dataset-contract construction fails loudly if fewer than 2 holdout groups per side remain eligible (the contract requires ≥2 validation and ≥2 test views). A replay of d62fbce0's real diagnostics confirmed exactly 3 violations and all 6 hole-adjacent groups in Train; in this specific job the old spatial order happened not to hold any of them out, so the split output is unchanged — the warning is the part that fires for d62fbce0, the avoidance is the safety net. Open question: whether `colmap_matcher=sequential` mitigates the 281–338 s holes is to be judged from the next remote job's `video_registration.json`. Tests: 338 green, including the first test of the gate-failure raise path (previously zero repo-wide coverage).
- On 2026-08-24, the user approved extending the existing experimental `colmap_matcher=sequential` option to the video-only VGGT-BA geometry path before the next combined VGGT-BA + VGGT visibility + default SOR + recovery-prune run. The default remains `exhaustive`; only an explicit sequential request changes matching. `ProjectGaussianAdapter` now resolves the matcher and pinned vocab tree once for either `colmap` or `vggt_ba`, fails fast when the tree is missing, and passes the same `--matcher/--vocab-tree-path` policy to the selected runner. `run_vggt_ba_sparse.py` uses sequential matching with vocab-tree loop detection when selected, records `colmap_matcher` in its profile/diagnostics/log and the manifest metrics, and reuses that same database for seeded triangulation/registration and any classified ordinary-Mapper fallback. This removes VGGT-BA's hard-coded O(N²) exhaustive matching for the planned 1,000-keyframe run without changing VGGT windows, BA, registration gates, fallback classes, SOR, recovery-prune, or VGGT postprocessing. This combined run is still a multi-variable integration check, not promotion-grade evidence for any one component.
- On 2026-08-24, the two video P0 implementation foundations landed without changing the default job path. `video_keyframes_standard_v2` keeps a 6 fps candidate stream, selects an immutable 4 fps quality-aware base, and uses deterministic sparse Lucas-Kanade motion (descriptor novelty fallback for weak texture) for at most one additional frame per second up to 5 fps; filenames are stable candidate-index + source-PTS identities and selection schema 2 records motion/source/reason evidence. The historical v1 selector and 1,000-frame cap remain replayable and remain the API/frontend default pending geometry gates. A shared registration timeline now owns the strict `gap > 2.0s` definition for diagnostics and temporal split protection. Both ordinary COLMAP and VGGT-BA runners can explicitly consume a v2 source/selection pair, set sequential overlap to `clamp(ceil(effective_fps * 4), 16, 24)`, and, after their final initial sparse model but before the only undistortion/export, run at most two local incremental recovery rounds. Each round materializes only viable candidates within a gap plus two-second bridge margins, is capped at 25% of the initial selection and 50% cumulatively, reuses the database/camera through local pair matching, then runs image registration, non-clearing triangulation, and BA. A recovered model is published only if it loses no registered camera, retains at least 90% of sparse points, keeps the 12/70%/80% hard gates, and strictly improves max gap, total gap excess, or registration count; command/model failure preserves the prior model and prior accepted selection. VGGT-BA applies the same recovery after either the seeded model or one of the three classified ordinary-Mapper fallbacks without changing `effective_geometry_source`. Diagnostics remain soft when gaps survive. No remote experiment, default promotion, public raw tuning controls, or full 3DGS run is authorized by this code landing; geometry-only A/B evidence remains required.
- On 2026-08-24, the explicit v2 path was connected end to end without promotion. `POST /api/jobs` and `JobStore` now accept `video_keyframe_profile=standard_v1|standard_v2` while retaining v1 as the API default and frontend hidden submission. `ProjectGaussianAdapter` passes the requested profile to extraction and passes the original video plus selection sidecar to either ordinary COLMAP or VGGT-BA only for v2; after geometry it reloads the accepted final sidecar before registration diagnostics, temporal dataset construction, and training. Completed v2 manifests publish stable `video_registration_recovery` diagnostics plus initial/base/adaptive/recovery selection counts, attempted/accepted rounds, local pair/time totals, registered gain, and pre/post registration-gap/sparse-point summaries; each round records materialization and five COLMAP substage timings and one run-log summary. Recovery remains fail-soft, surviving gaps remain soft warnings, and final `video_selected_count`/registration metrics describe the actual training input/model. Recovery stages occupy the mapping-to-undistortion interval and `JobStore` clamps progress against regression. The frontend adds only a diagnostics download role, not a profile or raw-policy control. Local verification passed 362 tests, the production frontend build, and the backend smoke; the smoke now waits for asynchronous job completion before requesting terminal assets, removing its pre-existing queued-job race. No remote job, geometry A/B, default switch, or full 3DGS run was started; those remain separately gated.
- On 2026-08-25, the first complete explicit-v2 job (`20260824_082806_a31d80a0`) blocked promotion but isolated two correctable P0 defects. The 4→5 fps initial selection was beneficial (1,824/1,707 registered, 93.586%, three gaps and 26.321 s total versus frozen ordinary-v1 1,000/926, four gaps and 30.647 s), and 27 byte-identical shared Validation views improved +1.172 dB/+0.01865 SSIM after full training. Recovery then added all 35 locally viable frames in round one, registered 12, improved only one gap boundary by 0.166 s, and spent 318 s triangulating plus 618 s in BA; round two exited in 0.011 s because no unused candidate remained, even though round-one triangulation had created 26,635 new points that had never received a second registration pass. Initial v2 Mapper also took about 3.5 h, putting extraction+geometry near 6.9× the frozen v1 baseline. The fix changes the unit of the two-round budget from “candidate-addition rounds” to registration propagation rounds: round two may run with zero new candidates, each `image_registrator` is inspected before triangulation, only strict gap count/max/excess improvement proceeds, and accepted triangulations share one final CUDA BA with one CPU fallback. Registering only out-of-gap cameras no longer justifies expensive recovery. Explicit v2 ordinary Mapper and classified VGGT-BA ordinary fallback now share frozen 1.5 growth ratios, 1,000-frame/1,000,000-point intermediate global-BA frequencies, and one refinement per global cycle; v1 command behavior is unchanged. New keyframe/COLMAP timing diagnostics and `scripts/evaluate_video_v2_promotion.py` make the existing gate executable: two v2 selections must match, registration ≥95%, zero gaps above 2 s, no camera loss, ≥90% point retention, ≤2 rounds/50% recovery, and extraction+geometry ≤2× a same-code v1 baseline. Per user direction, no local real-video geometry/training or automatic remote task is run; v1 remains API/frontend default until the user returns an all-PASS remote geometry-only report.
- On 2026-08-28, the first remote run of the propagation/global-BA fix still failed four promotion checks and identified full-sequence Mapper as the remaining bottleneck: final registration was 1,735/1,848 (93.885%), two gaps survived with a 13.178 s maximum, and extraction+geometry was 3.143× same-code v1. The run did verify deterministic selection, zero initial-camera loss, point retention, the two-round/50% limits, and one successful CUDA final BA. Timing attributed 5,596.110 s of 6,909.314 s COLMAP time to Mapper; recovery used only 24 of the 912-frame allowance and gained seven cameras in augmentation plus one in propagation, so adding rounds or candidates was rejected as low-yield and contract-breaking. Ordinary-COLMAP v2 now extracts and matches the complete selected set but gives Mapper an immutable image list of at most 1,000 uniformly spaced `base` frames. It then runs at most two selected-frame `image_registrator` + non-clearing `point_triangulator` expansion passes against the same database before the existing two-round gap recovery. Each expansion pass must add cameras, preserve every prior camera, and retain at least 90% of points; no BA runs inside expansion, and any accepted expansion forces the existing single final CUDA/CPU-fallback BA even when no recovery gap remains. New schema-1 `video_initial_registration_expansion` diagnostics, manifest metrics/assets, progress stages, and the corresponding COLMAP timing stage make this strategy measurable; v1 and VGGT-BA behavior are unchanged. This is a remote-evidence-driven iteration inside the existing P0 gate, not a promotion: real-video rerun evidence is still required and v1 remains the API/JobStore/frontend default.
- On 2026-08-31, the user approved a three-part SfM inspection surface: query the three input cameras most similar to the current Gaussian view, inspect original frames plus detector keypoints and candidate/geometrically verified pair correspondences, and switch to the final sparse PLY with camera trajectory. The stable backend foundation exports from the final accepted COLMAP database/model only—never `lifecycle/partial`—after dataset normalization/splits are frozen. `sfm_diagnostics` schema 1 uses deterministic sharded gzip JSON, stable frame IDs and multi-run detector/matcher provenance; it excludes descriptors and the mutable database, keeps image-space `(x,y)` at 0.01-pixel precision, and distinguishes untested pairs from tested zero-match pairs. `sfm_sparse_point_cloud` publishes the raw `colmap_world` PLY as an independently addressable role, while generic postprocessing may also publish `point_cloud_aligned`; nearest-camera records use the Gaussian normalized arbitrary-unit frame. Export is default but fail-soft; retained 1,824-image evidence estimates roughly 180 MB compressed versus 1.6 GB for the source database. Frontend rendering remains manifest-driven and lazy. Pixel homography stitching, final 3D-track interaction, and multi-algorithm side-by-side comparison are deliberately deferred; schema 1 does not claim metric scale or capture coverage from nearest-camera rank alone.
- On 2026-08-31, the SfM inspection surface gained upright reuse and the sparse view gained alignment. Gaussian jobs now run the generic RANSAC dominant-plane alignment on the final sparse cloud (`_try_align_point_cloud` falls back from `point_cloud` to `sfm_sparse_point_cloud`), and the SfM sparse viewer defaults to the aligned PLY with the Raw/Aligned toggle restored; old jobs were backfilled out-of-band with the same parameters. The Gaussian splat viewer additionally applies a display-only upright rotation R = rot(alignment transform) ∘ unitRot(export `world_from_normalized`) via the splat scene rotation option, selects signed global ±Z from the rotated mean camera-up (avoiding an upside-down room when the fitted plane normal sign is ambiguous), rotates the viewer frame, walk contract vectors/collision root, and inverse-rotates “检查输入视图” queries so all subsystems stay registered; missing or invalid alignment degrades to the trained orientation. Gaussian view deliberately gets no X/Y/Z mirror toggles (negative scale would break splat sorting), and exported assets remain untouched.
- On 2026-09-01, the three-part SfM surface was promoted from isolated viewer controls to a reconstruction-evidence workbench. A manifest-derived evidence rail links current-view inputs, tested feature pairs, sparse geometry and the Gaussian result; unavailable historical diagnostics remain explicit. The inspector now searches every feature-extracted frame (including unregistered frames), lazily loads per-frame keypoints, and navigates actual tested pair adjacency sorted by verified support, with run provenance and deterministic 50/150/300-line Canvas rendering. Sparse geometry uses robust framing, a full camera trajectory plus at most 120 sampled frusta, independent visibility controls and camera-up-based ±Z display orientation. The demo adds focus mode and collapses verbose metrics/assets behind a core quality summary. This remains a frontend/diagnostic promotion only: schema 1, arbitrary-unit semantics, Train/Validation/Test boundaries, descriptor/database exclusion and fail-soft export are unchanged; a new real job is still required before declaring the accepted-attempt exporter path visually validated.
- On 2026-09-01, the user approved a third Gaussian trainer identity, experimental native `mcmc`, alongside isolated Graphdeco and Project v7 without changing the Graphdeco default. Gaussian config schema 10 freezes `mcmc_v1`: installed Apache-2.0 gsplat 1.5.3 `MCMCStrategy`, 30k iterations, initial opacity 0.5, frozen 3NN scale ×0.1, opacity/scale regularization 0.01, opacity LR 0.05, position delay 0.01, relocation/growth from 500 through 25k every 100 steps, and a GLOBAL 3,000,000-Gaussian cap split exactly across ranks. MCMC runs inside the existing native model/optimizer/checkpoint/Validation-selected lifecycle, disables Default prune/opacity reset/recovery-prune, synchronously rejects any rank with an all-dead relocation set before gsplat's empty multinomial, and records strategy/cap/topology telemetry. Project and MCMC share distributed dispatch and common SOR/Validation/export/VGGT/manifest/Viewer stages; Graphdeco remains isolated. Complete native preparation now publishes a hash-validated `gaussian/replay/` containing the unchanged dataset/camera paths, registered undistorted images, and frozen initialization via hardlink/copy, but no COLMAP database/matches; `--initialization frozen` reruns Project or MCMC without geometry, point selection, or 3NN recomputation. Stable assets distinguish immutable `gaussian_raw_model` from the post-SOR `gaussian_model` and expose replay dataset/record. API/frontend/CLI can select MCMC, but raw method leaves remain private, Test remains unloaded, units remain normalized/arbitrary, and no default promotion or real CUDA evidence is claimed until the user runs the remote smoke and replay A/B.
- On 2026-09-01, the user directed the next geometry work to proceed in RGB-pipeline order, starting with local feature extraction before local matching, pairing, calibration, Mapper, triangulation/BA, and downstream 3DGS evidence. Phase 1 productizes COLMAP 4's native ALIKED capability without adding HLoc: `sfm_feature_profile=sift_v1|aliked_n16rot_v1` applies consistently to ordinary COLMAP, COLMAP+VGGT, and the final COLMAP database stage of Project/VGGT-BA. `sift_v1` remains the default and explicitly preserves SIFT/8192/`SIFT_BRUTEFORCE`; experimental `aliked_n16rot_v1` freezes ALIKED N16Rot/8192/min-score 0.2/`ALIKED_BRUTEFORCE`. Official ONNX assets are installed only by a dry-run-by-default setup script and verified by size/SHA at request and runner boundaries; Jobs never download or silently fall back. Existing `colmap_matcher=exhaustive|sequential` remains the pairing policy, not the local matcher. SfM frontend diagnostics schema 2 separates feature, local matcher, pairing, geometric verification and incremental Mapper provenance while the frontend maps historical schema 1 to its known SIFT baseline. LightGlue, ALIKED vocab pairing, feature replay, Global Mapper, camera profiles and BA changes remain later single-factor phases; Test, normalized arbitrary units, video v1 default and all trainer defaults remain unchanged pending real geometry evidence.
- On 2026-09-02, Phase 2 independently productized COLMAP local matching as `sfm_local_matcher=bruteforce|lightglue` across ordinary COLMAP, COLMAP+VGGT, Project ordinary geometry, and the final COLMAP database stage of VGGT-BA. The selected extractor/matcher options also reach standard-v2 recovery `feature_extractor`/`matches_importer` commands for newly materialized frames, preventing those rounds from reverting to COLMAP's SIFT/brute-force defaults. The feature descriptor determines the concrete enum: SIFT/ALIKED × brute-force/LightGlue map to the four compatible COLMAP matchers; LightGlue fixes the descriptor-specific minimum score at 0.1 and uses pinned COLMAP-release ONNX assets whose size/SHA is checked by the existing dry-run setup and at enqueue/runner boundaries. `/api/backends` reports matcher availability nested under each feature, API/frontend expose a separate local-matcher control, and manifest/diagnostics distinguish stable requested/effective matcher profile from the concrete COLMAP enum/model SHA. `bruteforce` remains default. The frozen 2026-08-13 2048-keypoint/1280px ETH3D matrix remains risk evidence—SIFT-LightGlue nearly disconnected, ALIKED-LightGlue completed at much higher matching cost—but cannot select the current 8192-keypoint configuration. No runtime download, silent fallback, pairing/Mapper/BA/trainer/default change, real geometry job, 3DGS training, Test use, or metric-scale claim is authorized by this code landing; Phase 1.5 feature-database replay remains required for the strictest matcher-only A/B, and Global Mapper remains Phase 6.
- On 2026-09-02, Phase 3 productized image-pair selection as `sfm_pairing=exhaustive|sequential_loop|vocab_tree` across ordinary COLMAP, COLMAP+VGGT, Project ordinary geometry, and the final COLMAP database stage of VGGT-BA. New product Jobs retain `exhaustive`; video-only `sequential_loop` runs temporal sequential matching with loop detection, while multi-image-only `vocab_tree` runs retrieval matching. Both resolve a descriptor-compatible, hash-recorded official FAISS tree: the existing COLMAP 3.11.1 SIFT Flickr100K 256K tree or COLMAP 3.13.0 ALIKED N16Rot Flickr100K 64K tree; SIFT trees are never reused for ALIKED, and missing/corrupt assets fail without exhaustive fallback. The dry-run setup pins URL/size/SHA and is the only download path. Legacy `colmap_matcher=exhaustive|sequential` and runner `--matcher` remain compatible; conflicting old/new fields fail. `/api/backends` reports pairing availability under the selected feature/local matcher with mode restrictions, the frontend exposes a separate image-pair control, and requested/effective pairing plus command/tree provenance enter manifests, timing, logs, and SfM diagnostics schema 2. Standard-v2 recovery remains the existing bounded temporal pair-list operation and records that identity separately instead of pretending to rerun the initial pairing. No HLoc, guided/transitive matching, RANSAC, camera, Mapper, BA, trainer, Test, or default-promotion change is included; real geometry-only exhaustive-vs-retrieval evidence remains required, and Global Mapper remains Phase 6.
