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
    geometry_backend: mock | vggt | dust3r | mast3r | nerfstudio_3dgs
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
5. Add a geometry adapter interface. Done for the mock point-cloud path.
6. Integrate VGGT or DUSt3R as the first baseline.
7. Connect generated output to frontend via `manifest.json`.
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
- Reconstruction adapter contract now exposes `geometry_backend` and `output_type`; only `mock + point_cloud` is implemented.
- User has a Nerfstudio splatfacto checkpoint at `/home/owen/nerfstudio/outputs/drjohnson_hq/splatfacto/2026-06-22_161605/nerfstudio_models/step-000029999.ckpt`, but no browser-ready `.splat/.ply/.ksplat` export was found there.
- Nerfstudio `ns-export gaussian-splat` successfully exported `/home/owen/Image3D-SceneGraph/outputs/exports/drjohnson_hq/splat.ply` from that checkpoint; this file is intentionally under ignored `outputs/`.
- `scripts/register_gaussian_splat.py` can register an exported `.ply/.splat/.ksplat` as a local `nerfstudio_3dgs + gaussian_splat` job.
