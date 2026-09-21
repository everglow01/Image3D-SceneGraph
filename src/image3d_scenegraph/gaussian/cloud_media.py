"""Optional aiortc transport. Importing the base API never opens ICE sockets."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import importlib.util
import json
import os
import re
import time

from image3d_scenegraph.gpu_lease import cloud_enabled
from .cloud_render import CloudRenderError


def turn_settings():
    host = os.environ.get("IMAGE3D_TURN_HOST", "")
    secret = os.environ.get("IMAGE3D_TURN_SECRET", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host) or len(secret) < 32:
        raise CloudRenderError(
            "尚未配置同机认证 TURN（8082 TCP/UDP）；不会切换为图像流"
        )
    return host, secret


def capabilities():
    reason = None
    if not cloud_enabled():
        reason = "云渲染尚未由部署管理员启用；本地查看仍可用"
    elif any(
        importlib.util.find_spec(name) is None
        for name in ("aiortc", "av", "torch", "gsplat")
    ):
        reason = "缺少 cloud / GPU 可选依赖；需在获准服务器安装"
    else:
        try:
            turn_settings()
        except CloudRenderError as exc:
            reason = str(exc)
    return {
        "editing_core": True,
        "cloud_available": reason is None,
        "reason": reason,
        "media": "webrtc",
        "encoding": "aiortc_default_software",
        "max_gaussians": 3_000_000,
    }


def browser_ice(session_id):
    host, secret = turn_settings()
    username = f"{int(time.time()) + 3600}:{session_id}"
    credential = base64.b64encode(
        hmac.new(secret.encode(), username.encode(), hashlib.sha1).digest()
    ).decode()
    return {
        "iceTransportPolicy": "relay",
        "iceServers": [
            {
                "urls": [
                    f"turn:{host}:8082?transport=udp",
                    f"turn:{host}:8082?transport=tcp",
                ],
                "username": username,
                "credential": credential,
            }
        ],
    }


class CloudMedia:
    def __init__(self, service, session):
        from aiortc import RTCPeerConnection, RTCConfiguration, VideoStreamTrack
        from av import VideoFrame

        self.service, self.session = service, session
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.disconnect_task = None
        self.closed = False
        self.channel = None
        self.last_input = 0.0
        self.pending_camera = None
        self.input_handle = None

        class Track(VideoStreamTrack):
            async def recv(track):
                pts, time_base = await track.next_timestamp()
                rgb, sequence, revision = await service.video_frame(session)
                frame = VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts, frame.time_base = pts, time_base
                if (
                    self.channel
                    and self.channel.readyState == "open"
                    and self.channel.bufferedAmount < 16384
                ):
                    self.channel.send(
                        json.dumps({"camera_seq": sequence, "revision": revision})
                    )
                return frame

        self.track = Track()
        self.pc.addTrack(self.track)

        @self.pc.on("datachannel")
        def channel_open(channel):
            if channel.label != "camera" or self.channel is not None:
                channel.close()
                return
            self.channel = channel

            @channel.on("message")
            def message(data):
                if not isinstance(data, str) or len(data) > 8192:
                    return
                try:
                    value = json.loads(data)
                    if (
                        isinstance(value, dict)
                        and set(value) == {"paused"}
                        and type(value["paused"]) is bool
                    ):
                        session["paused"] = value["paused"]
                        return
                    if not isinstance(value, dict) or set(value) != {
                        "sequence",
                        "camera",
                    }:
                        return
                    sequence = value["sequence"]
                    if type(sequence) is not int or not 0 <= sequence <= 2**53 - 1:
                        return
                    if sequence <= session["seq"] or (
                        self.pending_camera
                        and sequence <= self.pending_camera["sequence"]
                    ):
                        return
                    self.pending_camera = value
                    if self.input_handle is None:
                        delay = max(0, 1 / 60 - (time.monotonic() - self.last_input))
                        self.input_handle = asyncio.get_running_loop().call_later(
                            delay, self._flush_camera
                        )
                except (ValueError, TypeError, KeyError):
                    return

        @self.pc.on("connectionstatechange")
        async def changed():
            state = self.pc.connectionState
            if self.closed:
                return
            if state in {"failed", "disconnected"} and self.disconnect_task is None:
                self.disconnect_task = asyncio.create_task(self._disconnected())

    def _flush_camera(self):
        self.input_handle = None
        value, self.pending_camera = self.pending_camera, None
        self.last_input = time.monotonic()
        if value is not None and not self.closed:
            try:
                self.service.camera_input(
                    self.session, value["sequence"], value["camera"]
                )
            except (ValueError, TypeError, KeyError):
                pass

    async def _disconnected(self):
        try:
            await asyncio.sleep(30)
            if not self.closed and self.pc.connectionState in {
                "failed",
                "disconnected",
            }:
                await self.service.close(self.session)
        finally:
            self.disconnect_task = None

    async def answer(self, sdp):
        from aiortc import RTCSessionDescription

        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        return {"type": "answer", "sdp": self.pc.localDescription.sdp}

    async def close(self):
        self.closed = True
        if self.input_handle:
            self.input_handle.cancel()
            self.input_handle = None
        self.pending_camera = None
        task = self.disconnect_task
        if task and task is not asyncio.current_task():
            task.cancel()
        self.track.stop()
        await self.pc.close()
