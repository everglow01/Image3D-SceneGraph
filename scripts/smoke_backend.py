from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main() -> None:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.setdefault("IMAGE3D_OUTPUT_ROOT", "outputs/jobs")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_server(base_url, process)
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            created = client.post(
                "/api/jobs",
                data={"mode": "image"},
                files={"files": ("smoke.jpg", b"smoke-image", "image/jpeg")},
            )
            created.raise_for_status()
            manifest = created.json()
            job_id = manifest["job_id"]

            panorama = client.post(
                "/api/jobs",
                data={"mode": "panorama"},
                files={"files": ("smoke_360.jpg", b"smoke-panorama", "image/jpeg")},
            )
            panorama.raise_for_status()
            assert panorama.json()["input_type"] == "equirectangular_panorama"

            for path in [
                f"/api/jobs/{job_id}",
                f"/api/jobs/{job_id}/manifest",
                f"/api/jobs/{job_id}/scene",
                f"/api/jobs/{job_id}/assets/geometry/points.ply",
            ]:
                response = client.get(path)
                response.raise_for_status()

        print(f"ok job_id={job_id}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"uvicorn exited early:\n{output}")
        try:
            response = httpx.get(f"{base_url}/api/health", timeout=1.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    output = process.stdout.read() if process.stdout else ""
    raise TimeoutError(f"uvicorn did not become ready:\n{output}")


if __name__ == "__main__":
    main()
