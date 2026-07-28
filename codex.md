# codex.md

> This file is the project working plan and maintenance guide for Codex and future development.
> It should evolve with the project. When scope changes, update this file first or in the same commit.

---

## 1. Project Name

**Image3D-SceneGraph**

Working description:

> A calibration-free image/video/panorama to semantic 3D scene reconstruction system.
> Users upload one image, multiple images, a video, or a 360 panorama; the system estimates geometry automatically, reconstructs a 3D scene, attaches semantic objects, infers spatial relations, and provides a web interface for viewing and export.

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

A user opens the web app and can:

1. Upload a single image, multiple images, a video, or a 360 panorama.
2. Start a reconstruction job.
3. Watch job progress and logs.
4. View the reconstructed 3D result in the browser.
5. Toggle geometry, RGB, semantic objects, camera trajectory, and scene graph.
6. Click an object and see its label, approximate 3D position, and relations.
7. Export results as `.ply`, `.glb`, scene graph `.json`, and optionally a zip bundle.

The system should feel like a usable demo, not a loose collection of scripts.

---

## 4. Core Problem Definition

Inputs:

- One RGB image.
- Multiple unordered RGB images.
- A video, internally converted to selected frames.
- One equirectangular 360 panorama image.

Unavailable from the user:

- camera intrinsics;
- camera extrinsics;
- camera trajectory;
- depth map;
- RGBD sensor data;
- manual object annotations.

Outputs:

- reconstructed geometry: point cloud first, mesh / 3DGS later;
- estimated camera parameters or trajectory when available;
- semantic object instances;
- object-level 3D positions;
- spatial relations;
- physical consistency diagnostics;
- exportable files and frontend visualization.

---

## 5. Anti-Goals

Do not turn this into a universal 3D reconstruction platform.

Explicitly avoid:

- supporting every reconstruction model;
- supporting every viewer format early;
- training a large image-to-3D foundation model from scratch;
- overbuilding distributed infrastructure before the local pipeline works;
- claiming centimeter-level accuracy without a benchmark;
- hiding model limitations behind a polished frontend.

Prefer one reliable baseline and one clear algorithmic improvement path.

---

## 6. High-Level Architecture

```text
Frontend
  upload image/images/video/panorama
  view job status
  inspect 3D scene
  inspect semantic objects and relations
  export results

Backend API
  create job
  store input assets
  run reconstruction worker
  serve generated assets
  serve scene graph JSON

Reconstruction Worker
  frame extraction and selection
  geometry model inference
  point cloud / camera export
  mesh or 3DGS generation
  semantic segmentation / VLM parsing
  2D-to-3D semantic fusion
  scale recovery and physical consistency optimization
  scene graph generation
```

---

## 7. Recommended Tech Stack

Frontend:

- React + Vite + TypeScript.
- Three.js or React Three Fiber for interactive 3D viewing.
- Keep UI practical: upload area, task timeline, viewer, object panel, export panel.

Backend:

- Python environment management: `uv`.
- FastAPI.
- Local filesystem job storage for MVP.
- Background worker process for GPU jobs.
- SQLite can be added when job metadata becomes useful.

Algorithm stack:

- Geometry baseline: VGGT first, or DUSt3R / MASt3R if VGGT is inconvenient.
- Single-image object-level baseline: optional TripoSR-style model later.
- Multi-image/video/panorama scene output: point cloud first, mesh or 3D Gaussian Splatting later.
- Semantics: segmentation + VLM parsing, projected/fused into 3D.
- Algorithm contribution: scale recovery, physical consistency, semantic scene graph.

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

Each job should produce a directory like:

```text
outputs/jobs/{job_id}/
  input/
    images/
    video.mp4
  frames/
  geometry/
    cameras.json
    points.ply
    depth/
    mesh.glb
    scene.splat
  semantic/
    masks/
    objects.json
  scene_graph/
    scene.json
  logs/
    run.log
  manifest.json
```

`manifest.json` should be the frontend's stable entry point. It should list what assets exist and their relative paths.

Example manifest:

```json
{
  "job_id": "demo_001",
  "status": "done",
  "mode": "video",
  "assets": {
    "point_cloud": "geometry/points.ply",
    "mesh": "geometry/mesh.glb",
    "scene_graph": "scene_graph/scene.json",
    "log": "logs/run.log"
  },
  "metrics": {
    "num_frames": 48,
    "num_points": 1200000,
    "num_objects": 18
  }
}
```

---

## 10. API Draft

```text
POST /api/jobs
  Create a reconstruction job from uploaded image/images/video/panorama.
  Required reconstruction contract fields:
    mode: image | multi_image | video | panorama
    geometry_backend: mock | vggt | colmap | colmap_vggt | dust3r | mast3r | nerfstudio_3dgs
    output_type: point_cloud | mesh | gaussian_splat

GET /api/jobs/{job_id}
  Return job status, stage, progress, and errors.

GET /api/jobs/{job_id}/manifest
  Return output manifest.

GET /api/jobs/{job_id}/scene
  Return semantic scene graph JSON.

GET /api/jobs/{job_id}/assets/{path}
  Serve generated asset files.

GET /api/jobs/{job_id}/download
  Download complete result bundle.

GET /api/backends
  Return optional model availability, supported outputs, missing paths,
  and setup commands for frontend gating.
```

Do not design a large API before the MVP works. Add endpoints only when the frontend needs them.

---

## 11. Frontend MVP

The first frontend should include:

1. Upload panel.
2. Reconstruction mode selector: single image / multi image / video / panorama.
3. Job progress timeline.
4. 3D viewer.
5. Object list panel.
6. Scene graph / relations panel.
7. Export buttons.

Do not build a marketing landing page as the first screen. The first screen should be the actual tool.

Viewer features, in order:

1. Load a static demo `.ply` or `.glb`.
2. Load generated asset from a completed job.
3. Toggle point cloud / mesh.
4. Highlight selected object.
5. Display camera trajectory.
6. Display relation edges.

---

## 12. Algorithmic Contributions

This project must not be only model integration. The resume value should come from measurable algorithmic modules.

Primary algorithm modules:

### 12.1 Scale Recovery

Problem:

- Reconstruction models may produce geometry up to an arbitrary or unstable scale.

Possible solution:

- Infer scale from object priors: human height, door height, monitor size, desk height.
- Use scene priors: floor/table plane height, camera height range.
- Use optional user-provided single reference measurement later, but not in MVP.

Output:

- scale factor;
- confidence;
- before/after metric estimates.

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

### 13.2 Semantic office/tabletop benchmark

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

## 14. Milestones

### Milestone 0: Clean Project Bootstrap

Deliverables:

- repository skeleton;
- `.gitignore` for outputs, checkpoints, external models, caches;
- `README.md` with project statement;
- this `codex.md` committed.

Success criteria:

- a new contributor can understand the project in 5 minutes.

### Milestone 1: Frontend + Mock Backend

Deliverables:

- upload UI;
- job creation API;
- mock job status;
- viewer loads a sample `.ply` or `.glb`;
- export buttons can download sample assets.

Success criteria:

- the app demonstrates the intended workflow even before real reconstruction works.

### Milestone 2: Geometry Baseline

Deliverables:

- image/multi-image/video/panorama input handling;
- frame extraction for video;
- geometry model adapter;
- point cloud export;
- frontend displays generated point cloud.

Success criteria:

- upload real office video/images and view reconstructed geometry in browser.

### Milestone 3: Mesh or 3DGS Output

Deliverables:

- mesh or 3DGS export path;
- frontend viewer support;
- comparison with raw point cloud.

Success criteria:

- output is visually strong enough for a project demo video.

### Milestone 4: Semantic Fusion

Deliverables:

- segmentation/VLM baseline;
- object label output;
- 2D-to-3D fusion;
- object click/highlight in frontend.

Success criteria:

- common office objects appear as selectable 3D instances.

### Milestone 5: Physical Consistency + Scene Graph

Deliverables:

- plane detection;
- support and upright reasoning;
- violation reporting;
- scene graph JSON;
- query UI.

Success criteria:

- the project has measurable algorithmic contribution beyond calling a reconstruction model.

### Milestone 6: Benchmark and Resume Packaging

Deliverables:

- small annotated benchmark;
- metrics script;
- ablation table;
- demo video;
- README with architecture diagram and results.

Success criteria:

- the project can be explained in an interview as a complete vision algorithm system.

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

## 16. Initial Implementation Order

Recommended first steps:

1. Create repo skeleton and `.gitignore`. Done.
2. Build FastAPI mock job API. Done.
3. Build React/Vite frontend with upload and 3D viewer. Done.
4. Add one sample output asset to exercise the viewer. Covered by mock job output.
5. Add a geometry adapter interface. Done.
6. Integrate VGGT or DUSt3R as the first baseline. Done for VGGT point-cloud output.
7. Connect generated output to frontend via `manifest.json`. Done for mock, Nerfstudio import, and VGGT point-cloud jobs.
8. Add semantic fusion only after geometry output is stable.

Do not begin with training or fine-tuning.

---

## 17. Current Decision Log

- New workspace: `/home/owen/Image3D-SceneGraph`.
- Old workspace `/home/owen/3d_demo` remains an exploration repo.
- Project direction: image/video/panorama to semantic 3D scene reconstruction.
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
- Generic point-cloud alignment now analyzes three RANSAC plane candidates and selects the candidate with the strongest global inlier ratio unless `--plane-index` is explicitly set. This fixed the TSDF regression job above, whose first candidate had 7.10% support but whose second candidate had 11.96%, without lowering the 8% plane-quality threshold. G1.22 adds a separate diagnostic-only Manhattan-frame evaluator: it analyzes up to eight planes, filters them with the same 8% gate, clusters unoriented near-parallel normals, and reports supported orthogonal triplets plus partial evidence and explicit ambiguity. G1.23 then adds a separate gravity-axis evidence evaluator over an unambiguous G1.22 frame: it audits optional IMU records, COLMAP camera image-up, camera-centre/point robust spans, and reliable boundary-plane ordering with frozen per-source scores, selection/margin gates, and explicit missing/invalid fallback. EXIF Orientation is not treated as gravity, camera image-up is only a capture prior, no plane is labeled ground, and weak/conflicting evidence remains ambiguous. Retained private-225 has no IMU sidecar; all four available geometric sources selected Manhattan axis 0 with combined scores 0.52815/0.23707/0.23477 and margin 0.29108, while camera image-up coherence selected the negative axis direction as up. Both tools are diagnostic-only and do not alter `align_pointcloud.py`, retained assets, or production alignment; applying and evaluating that candidate remains G1.24.
- Root `plan.md` is the task-level execution checklist beneath this plan of record. It decomposes the roadmap into five gated stages: trustworthy offline geometry, 3DGS rendering, near-real-time incremental video reconstruction, long-horizon consistency/scale/dynamics, and evidence-aware semantic scene graphs. Tasks are completed one at a time and marked done only after their stated acceptance criteria pass; failed or mixed experiments remain recorded rather than being silently removed.
