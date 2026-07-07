from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an exported Gaussian splat asset as a local job.")
    parser.add_argument("--splat", required=True, type=Path, help="Path to a .ply, .splat, or .ksplat file.")
    parser.add_argument("--name", default="imported_splat", help="Human-readable source name.")
    parser.add_argument(
        "--output-root",
        default=os.environ.get("IMAGE3D_OUTPUT_ROOT", "outputs/jobs"),
        type=Path,
        help="Job output root.",
    )
    args = parser.parse_args()

    if args.splat.suffix.lower() not in {".ply", ".splat", ".ksplat"}:
        raise SystemExit("splat asset must end with .ply, .splat, or .ksplat")
    if not args.splat.is_file():
        raise SystemExit(f"splat asset not found: {args.splat}")

    job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = args.output_root / job_id
    geometry_dir = job_dir / "geometry"
    scene_graph_dir = job_dir / "scene_graph"
    logs_dir = job_dir / "logs"
    geometry_dir.mkdir(parents=True, exist_ok=False)
    scene_graph_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)

    target_name = f"scene{args.splat.suffix.lower()}"
    target_path = geometry_dir / target_name
    shutil.copy2(args.splat, target_path)

    scene = {
        "job_id": job_id,
        "mode": "imported_asset",
        "coordinate_system": "nerfstudio_export",
        "objects": [],
        "relations": [],
        "diagnostics": {
            "scale_recovered": False,
            "physical_checks": [],
        },
    }
    (scene_graph_dir / "scene.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")

    log_lines = [
        f"job_id={job_id}",
        "mode=imported_asset",
        "geometry_backend=nerfstudio_3dgs",
        "output_type=gaussian_splat",
        f"source_name={args.name}",
        f"source_splat={args.splat}",
        f"scene_splat=geometry/{target_name}",
        "",
    ]
    (logs_dir / "run.log").write_text("\n".join(log_lines), encoding="utf-8")

    manifest = {
        "job_id": job_id,
        "status": "done",
        "stage": "imported_gaussian_splat",
        "progress": 1.0,
        "mode": "imported_asset",
        "input_type": "gaussian_splat_asset",
        "geometry_backend": "nerfstudio_3dgs",
        "output_type": "gaussian_splat",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": [
            {
                "filename": args.splat.name,
                "path": str(args.splat),
                "content_type": None,
                "size_bytes": args.splat.stat().st_size,
            }
        ],
        "assets": {
            "scene_splat": f"geometry/{target_name}",
            "scene_graph": "scene_graph/scene.json",
            "log": "logs/run.log",
        },
        "metrics": {
            "num_inputs": 1,
            "num_objects": 0,
            "splat_asset_bytes": args.splat.stat().st_size,
        },
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"registered job_id={job_id}")
    print(f"manifest={job_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
