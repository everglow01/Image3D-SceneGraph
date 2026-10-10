"""Submit the approved apartment demo pair to the existing standard Job worker."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import ProxyHandler, build_opener

from image3d_scenegraph.jobs import JobStore, UploadedInput
from prepare_eyeful_capture_metadata import prepare


def write_new(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, sort_keys=True)
        stream.write("\n")


def check_backend(output_root: Path) -> None:
    with build_opener(ProxyHandler({})).open(
        "http://127.0.0.1:8000/api/health", timeout=15
    ) as response:
        if json.load(response) != {"status": "ok"}:
            raise RuntimeError("standard backend is not healthy")
    with (output_root / ".worker.lock").open("r") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise RuntimeError("standard Job worker does not own its lease")


def submit(store, root: Path, metadata: dict, control: Path, trainer: str, source=None):
    options = {
        "gaussian_trainer": trainer,
        "gaussian_geometry_source": "colmap",
        "gaussian_longest_edge": 2048,
        "gaussian_final_fit": "train_validation_v1",
        "gaussian_sor_filter": "on",
        "gaussian_recovery_prune": "on" if trainer == "project" else "off",
    }
    if source is None:
        options.update(
            sfm_feature_profile="sift_v1",
            sfm_local_matcher="bruteforce",
            sfm_pairing="rig_neighbors_vocab_v1",
            sfm_camera_calibration="folder_grouped_opencv_v1",
            sfm_geometric_verification="default_v1",
            sfm_mapper="incremental",
            sfm_capture_metadata=json.dumps(metadata, sort_keys=True),
        )
    else:
        options["gaussian_geometry_source_job_id"] = source
    # JobStore moves staged uploads; only disposable copies may be passed here.
    with TemporaryDirectory(prefix=trainer + "-inputs-", dir=control) as staging:
        files = []
        for name in sorted(metadata["images"]):
            original = root / "images-jpeg-2k" / name
            target = Path(staging) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            files.append(
                UploadedInput(
                    filename=name, staged_path=target, content_type="image/jpeg"
                )
            )
        return store.enqueue_job(
            "multi_image",
            files,
            geometry_backend="project_3dgs",
            output_type="gaussian_splat",
            options=options,
        )


def wait_for_job(store: JobStore, job_id: str) -> None:
    previous = None
    while True:
        manifest = store.get_manifest(job_id)
        snapshot = {
            key: manifest.get(key)
            for key in ("job_id", "status", "stage", "progress", "error")
        }
        if snapshot != previous:
            print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            previous = snapshot
        if manifest["status"] == "done":
            return
        if manifest["status"] in {"failed", "cancelled"}:
            raise RuntimeError(
                f"{job_id}: {manifest['status']}: {manifest.get('error')}"
            )
        check_backend(store.output_root)
        time.sleep(30)


def monitor(control: Path, store: JobStore) -> None:
    while True:
        for trainer in ("project", "mcmc"):
            record = control / f"{trainer}.json"
            if record.is_file():
                job_id = json.loads(record.read_text())["job_id"]
                manifest = store.get_manifest(job_id)
                print(
                    json.dumps(
                        {
                            key: manifest.get(key)
                            for key in (
                                "job_id",
                                "status",
                                "stage",
                                "progress",
                                "error",
                            )
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        for terminal in ("failed.json", "complete.json"):
            if (control / terminal).is_file():
                print((control / terminal).read_text(), flush=True)
                if terminal == "failed.json":
                    raise SystemExit(1)
                return
        check_backend(store.output_root)
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()
    store = JobStore(args.output_root)
    if args.monitor:
        monitor(args.control_dir, store)
        return
    control = args.control_dir
    control.mkdir(parents=True, exist_ok=True)
    with (control / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new(control / ".once", {"expected_revision": args.expected_revision})
        try:
            metadata = prepare(args.dataset_root)
            if len(metadata["images"]) != 3744:
                raise ValueError("approved apartment Train image count changed")
            write_new(control / "capture_metadata.json", metadata)
            source = None
            jobs = {}
            for trainer in ("project", "mcmc"):
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
                if revision != args.expected_revision:
                    raise RuntimeError("deployed revision changed; refusing submission")
                if subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], text=True
                ).strip():
                    raise RuntimeError("tracked deployment files changed")
                check_backend(store.output_root)
                subprocess.run(["nvidia-smi", "-L"], check=True)
                manifest = submit(
                    store, args.dataset_root, metadata, control, trainer, source
                )
                source = manifest["job_id"]
                jobs[trainer] = source
                write_new(control / f"{trainer}.json", {"job_id": source})
                print(json.dumps({"submitted": trainer, "job_id": source}), flush=True)
                wait_for_job(store, source)
            write_new(control / "complete.json", {"status": "done", "jobs": jobs})
        except Exception as exc:
            write_new(control / "failed.json", {"status": "failed", "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
