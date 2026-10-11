"""Offline, CPU-only repair of missing diagnostics on completed multi-image COLMAP Jobs."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from image3d_scenegraph.geometry.adapters import ReconstructionContext, _try_export_sfm_diagnostics
from image3d_scenegraph.gpu_lease import FileLease
from image3d_scenegraph.jobs import JobError, JobStore


def repair_diagnostics(output_root: Path, job_id: str) -> dict:
    store = JobStore(output_root.resolve())
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise JobError("job ID must be a directory name")
    job = store.job_dir(job_id)
    if job.is_symlink() or not job.is_dir():
        raise JobError("job directory must exist and must not be a symbolic link")
    # The backend must be stopped: its worker owns this lease even while idle.
    with FileLease(store.output_root / ".worker.lock"):
        job = store.job_dir(job_id)
        manifest_path = store._contained_file(job, "manifest.json")
        before = manifest_path.read_bytes()
        manifest = json.loads(before)
        if (
            manifest.get("job_id") != job_id
            or manifest.get("status") != "done"
            or manifest.get("mode") != "multi_image"
            or manifest.get("geometry_backend") != "project_3dgs"
            or manifest.get("output_type") != "gaussian_splat"
            or manifest.get("gaussian_geometry_effective_source") != "colmap"
            or manifest.get("result_kind") is not None
            or manifest.get("navigation_status") in {"queued", "generating"}
        ):
            raise JobError("repair requires an idle completed ordinary multi-image COLMAP Gaussian Job")
        if manifest.get("assets", {}).get("sfm_diagnostics"):
            raise JobError("SfM diagnostics already published; refusing replacement")
        dataset = store._contained_file(job, manifest["assets"]["gaussian_dataset"])
        database = store._contained_file(job, "colmap/database.db")
        output = job / "diagnostics" / f"sfm-repair-{uuid.uuid4().hex}"
        context = ReconstructionContext(job_id, job, "multi_image", manifest["inputs"], {})
        assets, metrics, _ = _try_export_sfm_diagnostics(
            context=context, database_path=database, source_image_root=job / "input/images",
            dataset_path=dataset, output_dir=output,
            feature_profile=manifest["sfm_feature_effective_profile"],
            local_matcher=manifest["sfm_local_matcher_effective"],
            pairing=manifest["sfm_pairing_effective"],
            geometric_verification=manifest["sfm_geometric_verification_effective"],
            camera_calibration=manifest["sfm_camera_calibration_effective"],
            geometry_source="colmap", mapper=manifest["sfm_mapper_effective"],
            video_selection_path=None,
        )
        if metrics.get("sfm_diagnostics_status") != "available":
            raise JobError(str(metrics.get("sfm_diagnostics_reason")))
        if manifest_path.read_bytes() != before:
            raise JobError("manifest changed during repair; diagnostics were not attached")
        with (output / "manifest-before-repair.json").open("xb") as stream:
            stream.write(before)
        manifest["assets"].update(assets)
        manifest.setdefault("metrics", {}).update(metrics)
        manifest["metrics"].pop("sfm_diagnostics_reason", None)
        store._write_json(manifest_path, manifest)
        return {"job_id": job_id, "sfm_diagnostics": assets["sfm_diagnostics"], "status": "available"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    print(json.dumps(repair_diagnostics(args.output_root, args.job_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
