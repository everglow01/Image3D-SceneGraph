from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/open_rtx_browser.sh"


def test_rtx_launcher_uses_isolated_profile_and_quotes_url(tmp_path):
    chrome = tmp_path / "google-chrome-stable"
    chrome.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'gpu': [os.environ[k] for k in "
        "['__NV_PRIME_RENDER_OFFLOAD', '__GLX_VENDOR_LIBRARY_NAME', '__VK_LAYER_NV_optimus']]}))\n"
    )
    chrome.chmod(0o700)
    env = {**os.environ, "HOME": str(tmp_path / "home with spaces"), "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    for args, expected_url in [([], "http://localhost:8081/"), (["http://localhost:8081/?q=a b;$(false)"], "http://localhost:8081/?q=a b;$(false)")]:
        result = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, check=True)
        record = json.loads(result.stdout)
        assert record["gpu"] == ["1", "nvidia", "NVIDIA_only"]
        assert record["args"] == ["--ozone-platform=x11", "--use-angle=vulkan",
                                  f"--user-data-dir={env['HOME']}/.cache/image3d-rtx4060-chrome",
                                  "--new-window", expected_url]
    for args in [["file:///etc/passwd"], ["--no-sandbox"], ["http://localhost", "extra"]]:
        result = subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True)
        assert result.returncode == 2 and not result.stdout
