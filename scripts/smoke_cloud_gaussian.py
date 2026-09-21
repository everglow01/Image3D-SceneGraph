"""Bounded remote smoke test; reads source PLYs, writes only the new run root."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import traceback

import numpy as np

from image3d_scenegraph.file_integrity import sha256_file
from image3d_scenegraph.gaussian.cloud_media import CloudMedia, browser_ice
from image3d_scenegraph.gaussian.editor_session import EditorSessions
from image3d_scenegraph.gaussian.editing import GaussianEditStore
from image3d_scenegraph.jobs import JobStore


async def media_check(service, session, camera, transport):
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aioice.ice import TransportPolicy

    ice = browser_ice(session["id"])["iceServers"][0]
    client = RTCPeerConnection(
        RTCConfiguration(
            iceServers=[
                RTCIceServer(
                    urls=[url for url in ice["urls"] if url.endswith("=" + transport)],
                    username=ice["username"],
                    credential=ice["credential"],
                )
            ]
        )
    )
    client.addTransceiver("video", direction="recvonly")
    channel = client.createDataChannel("camera", ordered=False)
    received = asyncio.get_running_loop().create_future()
    consumers = []

    @channel.on("open")
    def opened():
        channel.send(json.dumps({"sequence": session["seq"] + 1, "camera": camera}))

    @client.on("track")
    def track_received(track):
        async def consume():
            try:
                shapes = []
                start = time.monotonic()
                for _ in range(5):
                    frame = await track.recv()
                    shapes.append([frame.width, frame.height])
                if not received.done():
                    received.set_result(
                        {
                            "decoded_frames": len(shapes),
                            "sizes": shapes,
                            "elapsed_seconds": time.monotonic() - start,
                        }
                    )
            except Exception as exc:
                if not received.done():
                    received.set_exception(exc)

        consumers.append(asyncio.create_task(consume()))

    media = CloudMedia(service, session)
    session["media"] = media
    try:
        # aiortc has no public relay-only setting; this diagnostic pins the installed aioice API.
        gatherers = {
            t.receiver.transport.transport.iceGatherer for t in client.getTransceivers()
        }
        gatherers.add(client.sctp.transport.transport.iceGatherer)
        for gatherer in gatherers:
            gatherer._connection._transport_policy = TransportPolicy.RELAY
        await client.setLocalDescription(await client.createOffer())
        candidates = [
            line
            for line in client.localDescription.sdp.splitlines()
            if line.startswith("a=candidate:")
        ]
        assert candidates and all(" typ relay " in line + " " for line in candidates)
        answer = await media.answer(client.localDescription.sdp)
        await client.setRemoteDescription(RTCSessionDescription(**answer))
        result = await asyncio.wait_for(received, 60)
        nominated = client.sctp.transport.transport.iceGatherer._connection._nominated
        assert nominated and all(
            pair.local_candidate.type == "relay" for pair in nominated.values()
        )
        result.update(
            transport=transport, relay_only=True, camera_sequence=session["seq"]
        )
        return result
    finally:
        for task in consumers:
            task.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
        await client.close()
        await media.close()
        session["media"] = None


async def run(args, report):
    jobs = JobStore(args.jobs)
    service = EditorSessions(jobs)
    service.edits = GaussianEditStore(jobs, args.output / "edits")
    job = jobs.output_root / args.job_id
    keyframe = json.loads((job / "camera_path.json").read_text())["keyframes"][0]
    rotation = np.asarray(keyframe["world_from_camera"])[:3, :3]
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ np.asarray(keyframe["center_normalized"])
    camera = {
        "camera_from_normalized": pose.tolist(),
        "width": 640,
        "height": 360,
        "intrinsic": [[360, 0, 320], [0, 360, 180], [0, 0, 1]],
    }
    try:
        for variant in ("project-train-only", "mcmc-train-only"):
            started = time.monotonic()
            doc = service.edits.create(args.job_id, variant_id=variant)
            source = doc["source"]
            path = jobs.get_asset_path(args.job_id, source["ply_asset"])
            before = sha256_file(path)
            await service.create(doc["edit_id"])
            session = service.active
            await session["loader"]
            assert session["state"] == "viewing", session["error"]
            item = {
                "variant": variant,
                "source_sha256": before,
                "gaussian_count": source["gaussian_count"],
                "load_seconds": time.monotonic() - started,
            }
            report["models"].append(item)
            frame = await service.freeze(session, 1, camera)
            assert frame["image"].startswith("data:image/png;base64,")
            import base64

            (args.output / (variant + ".png")).write_bytes(
                base64.b64decode(frame["image"].split(",", 1)[1])
            )
            center = session["means"][0].astype(float)
            selected = await service.selection(
                session,
                {
                    "ticket": frame["ticket"],
                    "expected_revision": 0,
                    "shape": "box",
                    "minimum": (center - 1e-7).tolist(),
                    "maximum": (center + 1e-7).tolist(),
                    "combine": "replace",
                },
            )
            assert 0 < selected["selected_count"] < source["gaussian_count"] / 2
            await service.preview(
                session,
                {
                    "ticket": frame["ticket"],
                    "expected_revision": 0,
                    "selection_token": selected["selection_token"],
                    "mode": "isolated",
                },
            )
            deleted = await service.operation(
                session,
                {
                    "ticket": frame["ticket"],
                    "expected_revision": 0,
                    "selection_token": selected["selection_token"],
                    "kind": "delete",
                    "operation_id": "smoke-delete",
                    "confirm_large": False,
                },
            )
            assert (
                deleted["visible_count"]
                == source["gaussian_count"] - selected["selected_count"]
            )
            frame = await service.freeze(session, 1, camera)
            undone = await service.operation(
                session,
                {
                    "ticket": frame["ticket"],
                    "expected_revision": 1,
                    "kind": "undo",
                    "operation_id": "smoke-undo",
                    "confirm_large": False,
                },
            )
            assert undone["visible_count"] == source["gaussian_count"]
            await service.save(session, 2)
            item["delete_undo_save"] = "passed"
            item["media"] = []
            for transport in ("udp", "tcp"):
                await service.resume(session)
                item["media"].append(
                    await media_check(service, session, camera, transport)
                )
            await service.close(session)
            item["source_unchanged"] = sha256_file(path) == before
            assert item["source_unchanged"]
            item["elapsed_seconds"] = time.monotonic() - started
    finally:
        await service.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--job-id", default="num4-retake-1080-train-only-comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "models": [],
        "scope": "server-side relay and GPU smoke, not browser/performance acceptance",
    }
    try:
        asyncio.run(asyncio.wait_for(run(args, report), 540))
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
