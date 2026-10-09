import os
from pathlib import Path

import pytest

from scripts import profile_gaussian_checkpoint as profile


def test_process_memory_reads_own_process_and_handles_exit():
    row = profile.process_memory(os.getpid())
    assert row["VmRSS_bytes"] > 0
    assert row["VmHWM_bytes"] >= row["VmRSS_bytes"]
    assert row["Pss_bytes"] > 0
    assert profile.process_memory(999999999) == {"pid": 999999999}


def test_profile_rejects_local_host_before_model_import(monkeypatch, tmp_path):
    monkeypatch.setattr(profile.socket, "gethostname", lambda: "local-host")
    monkeypatch.setattr(profile.sys, "argv", ["profile", "--output-dir", str(tmp_path / "new")])
    with pytest.raises(ValueError, match="remote"):
        profile.main()
    assert not (tmp_path / "new").exists()


def test_profile_records_memory_units(monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda p: "VmRSS: 10 kB\nVmHWM: 12 kB\n" if p.name == "status" else "Pss: 8 kB\nAnonymous: 7 kB\n")
    assert profile.process_memory(123) == {"pid": 123, "VmRSS_bytes": 10240, "VmHWM_bytes": 12288,
                                          "Pss_bytes": 8192, "Anonymous_bytes": 7168}
