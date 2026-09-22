"""Bounded GPU acceptance for cloud trimming; only new test documents are mutable."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import time
import traceback

import numpy as np

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.editor_session import EditorSessions
from image3d_scenegraph.gaussian.editing import GaussianEditStore
from image3d_scenegraph.jobs import JobStore


def camera_from_path(job):
    key = json.loads((job / "camera_path.json").read_text())["keyframes"][0]
    rotation = np.asarray(key["world_from_camera"], dtype=np.float64)[:3, :3]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ np.asarray(key["center_normalized"])
    return {
        "camera_from_normalized": pose.tolist(),
        "width": 640,
        "height": 360,
        "intrinsic": [[360, 0, 320], [0, 360, 180], [0, 0, 1]],
    }


def store_png(frame, path):
    assert frame["image"].startswith("data:image/png;base64,")
    path.write_bytes(base64.b64decode(frame["image"].split(",", 1)[1]))


async def check_model(
    service, jobs, job_id, variant, camera, output, item, *, diagnose=False
):
    started = time.monotonic()
    doc = service.edits.create(job_id, variant_id=variant)
    source = doc["source"]
    path = jobs.get_asset_path(job_id, source["ply_asset"])
    original_hash = sha256_file(path)
    item.update(
        variant=variant,
        document=doc["edit_id"],
        source_sha256=original_hash,
        gaussian_count=source["gaussian_count"],
        phase="loading",
    )
    await service.create(doc["edit_id"])
    s = service.active
    try:
        await s["loader"]
        assert s["state"] == "viewing", s["error"]
        item["load_seconds"] = time.monotonic() - started
        item["phase"] = "prepare"
        prepared = await service.prepare(s, 1, camera)
        assert prepared["camera_seq"] == 1 and prepared["revision"] == 0
        store_png(prepared, output / f"{variant}-fixed.png")
        frame = await service.freeze_prepared(s, prepared["ticket"], 0)
        assert frame == {"ticket": prepared["ticket"], "revision": 0, "camera_seq": 1}
        bind = {"ticket": prepared["ticket"], "expected_revision": 0}
        item["frame"] = {
            key: prepared[key]
            for key in ("width", "height", "render_ms", "camera_digest")
        }
        # Small candidate rectangles: do not silently switch to through-selection on empty pixels.
        result = None
        item["probes"] = []
        for cx, cy in ((320, 180), (240, 160), (400, 180)):
            polygon = [
                [cx - 16, cy - 16],
                [cx + 16, cy - 16],
                [cx + 16, cy + 16],
                [cx - 16, cy + 16],
            ]
            request = dict(
                bind,
                shape="polygon",
                polygon=polygon,
                depth_range=[0.01, 100.0],
                coverage=False,
                combine="replace",
                mode="visible",
                layer_tolerance=0.02,
            )
            item["phase"] = f"visible_{cx}_{cy}"
            if diagnose:
                for tolerance in (0.02, 0.1, 0.3):
                    picked = await service.visible_pick(
                        s, polygon, [0.01, 100.0], tolerance
                    )
                    item["probes"].append(
                        {
                            "center": [cx, cy],
                            "tolerance": tolerance,
                            "selected_ids": len(picked["ids"]),
                            **{
                                key: value
                                for key, value in picked.items()
                                if key != "ids"
                            },
                        }
                    )
                continue
            result = await service.selection(s, request)
            item["probes"].append(
                {"center": [cx, cy], "selected_ids": result["selected_count"]}
            )
            if 0 < result["selected_count"] < source["gaussian_count"] / 2:
                break
        if diagnose:
            item["phase"] = "diagnostic_only"
            return
        assert result and 0 < result["selected_count"] < source["gaussian_count"] / 2, (
            "no safe front-layer selection in sampled rectangles"
        )
        assert result["selected_count"] == result["deletable_count"]
        item["visible"] = {"rectangle": polygon, "count": result["selected_count"]}
        item["phase"] = "through_comparison"
        through = await service.selection(s, {**request, "mode": "through"})
        item["through_count"] = through["selected_count"]
        result = await service.selection(s, request)
        assert result["selected_count"] == item["visible"]["count"]
        item["phase"] = "protection"
        protected = await service.protection(
            s, dict(bind, kind="add", selection_token=result["selection_token"])
        )
        assert protected["protected_count"] == result["selected_count"]
        result = await service.selection(s, request)
        assert result["selected_count"] > 0 and result["deletable_count"] == 0
        protected = await service.protection(
            s, dict(bind, kind="remove", selection_token=result["selection_token"])
        )
        assert protected["protected_count"] == 0
        result = await service.selection(s, request)
        assert result["deletable_count"] == result["selected_count"]
        item["phase"] = "preview"
        for mode in ("isolated", "after_delete"):
            image = await service.preview(
                s, dict(bind, selection_token=result["selection_token"], mode=mode)
            )
            store_png(image, output / f"{variant}-{mode}.png")
        item["phase"] = "delete_undo"
        deleted = await service.operation(
            s,
            dict(
                bind,
                selection_token=result["selection_token"],
                kind="delete",
                operation_id=f"test-delete-{variant}",
                confirm_large=False,
            ),
        )
        assert (
            deleted["visible_count"]
            == source["gaussian_count"] - result["selected_count"]
        )
        refreshed = await service.freeze(s, s["seq"], camera)
        restored = await service.operation(
            s,
            dict(
                ticket=refreshed["ticket"],
                expected_revision=1,
                kind="undo",
                operation_id=f"test-undo-{variant}",
                confirm_large=False,
            ),
        )
        assert restored["visible_count"] == source["gaussian_count"]
        item["delete_undo"] = {
            "deleted": result["selected_count"],
            "restored": restored["visible_count"],
        }
        item["phase"] = "closed"
    finally:
        await service.close(s)
        item["source_unchanged"] = sha256_file(path) == original_hash
        item["elapsed_seconds"] = time.monotonic() - started
        assert item["source_unchanged"]


async def run(args, report):
    jobs = JobStore(args.jobs)
    service = EditorSessions(jobs)
    service.edits = GaussianEditStore(jobs, args.output / "edits")
    camera = camera_from_path(jobs.output_root / args.job_id)
    try:
        for variant in (
            ("project-train-only",)
            if args.diagnose
            else ("project-train-only", "mcmc-train-only")
        ):
            item = {"variant": variant, "phase": "starting"}
            report["models"].append(item)
            await asyncio.wait_for(
                check_model(
                    service,
                    jobs,
                    args.job_id,
                    variant,
                    camera,
                    args.output,
                    item,
                    diagnose=args.diagnose,
                ),
                timeout=600,
            )
    finally:
        await service.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--job-id", default="num4-retake-1080-train-only-comparison")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="One model; read-only selection metrics, no deletion",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "scope": "one-model read-only front-layer diagnostics"
        if args.diagnose
        else "two existing full models, one GPU, independent test documents, no save/export/training/Test",
        "models": [],
    }
    try:
        asyncio.run(run(args, report))
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"status": report["status"], "output": str(args.output)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
