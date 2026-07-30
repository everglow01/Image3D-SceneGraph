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
| `mesh_variants` | array | Optional generated mesh alternatives. |

Historical manifests can omit fields added after that job was created. Readers must tolerate absent optional fields and must not manufacture an effective policy that the old job did not record.

## Assets

Common asset roles include:

- `point_cloud`, `point_cloud_aligned`, and `cameras`
- `alignment_diagnostics`, `fusion_diagnostics`, `visibility_graph`, `scale_disagreement_diagnostics`, and `consistency_diagnostics`
- `mesh` and `mesh_diagnostics`
- `scene_splat`, `scene_graph`, and `log`

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
