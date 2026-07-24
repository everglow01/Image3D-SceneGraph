from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_vggt_group_diagnostics import (  # noqa: E402
    GroupDiagnosticsError,
    parse_run_configuration,
)


def test_parse_run_configuration_uses_authoritative_runner(tmp_path: Path):
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "\n".join(
            [
                "vggt_batch_size=4",
                "vggt_overlap_size=2",
                "vggt_grouping=sequential",
                "runner=python scripts/run_colmap_vggt_dense.py "
                "--vggt-batch-size 4 --vggt-overlap-size 2 --vggt-grouping sequential",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert parse_run_configuration(run_log) == {
        "grouping": "sequential",
        "batch_size": 4,
        "overlap_size": 2,
    }


def test_parse_run_configuration_rejects_unknown_grouping(tmp_path: Path):
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "runner=python scripts/run_colmap_vggt_dense.py "
        "--vggt-batch-size 4 --vggt-overlap-size 2 --vggt-grouping typo\n",
        encoding="utf-8",
    )

    with pytest.raises(GroupDiagnosticsError, match="unknown value"):
        parse_run_configuration(run_log)


def test_parse_run_configuration_rejects_runner_log_disagreement(tmp_path: Path):
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "\n".join(
            [
                "vggt_batch_size=8",
                "runner=python scripts/run_colmap_vggt_dense.py "
                "--vggt-batch-size 4 --vggt-overlap-size 2 --vggt-grouping covisibility",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(GroupDiagnosticsError, match="disagrees"):
        parse_run_configuration(run_log)
