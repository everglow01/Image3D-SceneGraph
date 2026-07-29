from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_stage1_factorial as factorial  # noqa: E402


TOLERANCES = (0.02, 0.05)


def test_balanced_main_pair_and_three_way_contrasts():
    coefficients = {
        "base": 1.0,
        "A": 2.0,
        "B": 3.0,
        "C": 5.0,
        "AB": 7.0,
        "AC": 11.0,
        "BC": 13.0,
        "ABC": 17.0,
    }
    cells = synthetic_cells(coefficients, budget_active=True)

    contrasts = factorial.calculate_contrasts(cells, [0.02], True)
    values = {
        item["contrast"]: item["value"]
        for item in contrasts
        if item["metric"] == "f1"
    }

    assert values == pytest.approx(
        {
            "A": 2.0 + 7.0 / 2 + 11.0 / 2 + 17.0 / 4,
            "B": 3.0 + 7.0 / 2 + 13.0 / 2 + 17.0 / 4,
            "C": 5.0 + 11.0 / 2 + 13.0 / 2 + 17.0 / 4,
            "AB": 7.0 + 17.0 / 2,
            "AC": 11.0 + 17.0 / 2,
            "BC": 13.0 + 17.0 / 2,
            "ABC": 17.0,
        }
    )


def test_inactive_budget_suppresses_only_budget_contrasts():
    contrasts = factorial.calculate_contrasts(synthetic_cells({}, budget_active=False), [0.02], False)
    status = {
        item["contrast"]: item["estimability"]
        for item in contrasts
        if item["metric"] == "f1"
    }

    assert status["A"] == "estimable"
    assert status["B"] == "estimable"
    assert status["AB"] == "estimable"
    assert status["C"] == "inactive_point_budget"
    assert status["AC"] == "inactive_point_budget"
    assert status["BC"] == "inactive_point_budget"
    assert status["ABC"] == "inactive_point_budget"


def test_failed_cell_is_preserved_and_dependent_contrasts_are_not_estimable(tmp_path):
    inputs = write_fixture(tmp_path)
    failed = inputs["benchmark_root"] / "scene" / "phase3_factorial" / "evaluations" / "arms" / "phase1" / "result.json"
    payload = json.loads(failed.read_text(encoding="utf-8"))
    payload.update(status="failed", metrics=None, error="frozen evaluator failure")
    payload["evaluator"]["return_code"] = 1
    failed.write_text(json.dumps(payload), encoding="utf-8")

    report = run_analysis(inputs)
    scene = report["scenes"][0]
    cell = next(item for item in scene["cells"] if item["arm"] == "phase1")
    assert cell["status"] == "failed"
    assert cell["metrics"] is None
    assert cell["failure"]["message"] == "frozen evaluator failure"
    assert all(
        item["estimability"] == "missing_or_failed_cells"
        for item in scene["contrasts"]
        if "phase1" in item["contributing_arms"]
    )


def test_hash_and_tolerance_mismatches_are_rejected(tmp_path):
    inputs = write_fixture(tmp_path)
    baseline = inputs["benchmark_root"] / "scene" / "phase3_factorial" / "evaluations" / "arms" / "baseline" / "result.json"
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["inputs"]["reconstruction_ply_sha256"] = "0" * 64
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(factorial.FactorialAnalysisError, match="SHA-256 mismatch"):
        run_analysis(inputs)

    inputs = write_fixture(tmp_path / "tolerance")
    phase1 = inputs["benchmark_root"] / "scene" / "phase3_factorial" / "evaluations" / "arms" / "phase1" / "result.json"
    payload = json.loads(phase1.read_text(encoding="utf-8"))
    payload["metrics"][0]["tolerance"] = 0.03
    phase1.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(factorial.FactorialAnalysisError, match="tolerance grid mismatch"):
        run_analysis(inputs)


def test_inventory_marks_unrun_interactions_and_cli_check_is_deterministic(tmp_path):
    inputs = write_fixture(tmp_path)
    output = tmp_path / "report.json"
    args = cli_args(inputs, output)

    subprocess.run([sys.executable, str(SCRIPTS_DIR / "analyze_stage1_factorial.py"), *args], check=True)
    first = output.read_bytes()
    subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "analyze_stage1_factorial.py"), *args, "--check"],
        check=True,
    )

    report = json.loads(first)
    assert output.read_bytes() == first
    assert all(item["status"] == "not_estimable" for item in report["unrun_interactions"])
    assert report["stage1_evidence_inventory"]["g1_24_manhattan_alignment"][
        "interaction_status"
    ] == "not_estimable_private_only"
    assert report["stage1_evidence_inventory"]["private_retained_configuration"][
        "point_budget_applied"
    ] is False


def synthetic_cells(coefficients: dict[str, float], budget_active: bool) -> list[dict]:
    cells = []
    for arm, bits in factorial.ARM_CONFIGS.items():
        a, b, c = bits
        value = (
            coefficients.get("base", 0.0)
            + coefficients.get("A", 0.0) * a
            + coefficients.get("B", 0.0) * b
            + coefficients.get("C", 0.0) * c
            + coefficients.get("AB", 0.0) * a * b
            + coefficients.get("AC", 0.0) * a * c
            + coefficients.get("BC", 0.0) * b * c
            + coefficients.get("ABC", 0.0) * a * b * c
        )
        cells.append(
            {
                "arm": arm,
                "factor_bits": list(bits),
                "metrics": [
                    {"tolerance": 0.02, "accuracy": value, "completeness": value, "f1": value}
                ],
                "point_budget": {"applied": budget_active},
            }
        )
    return cells


def write_fixture(root: Path) -> dict[str, Path]:
    benchmark_root = root / "benchmarks"
    factorial_dir = benchmark_root / "scene" / "phase3_factorial"
    runner = factorial_dir / "runner_reconstruction" / "diagnostics"
    runner.mkdir(parents=True)
    summaries = []
    cameras = b'{"schema_version": 1}\n'
    for arm, bits in factorial.ARM_CONFIGS.items():
        arm_dir = factorial_dir / "arms" / arm / "geometry"
        arm_dir.mkdir(parents=True)
        ply = arm_dir / "points.ply"
        ply.write_bytes(f"ply-{arm}".encode())
        (arm_dir / "cameras.json").write_bytes(cameras)
        levels = {
            name: values[bits[index]]
            for index, (name, values) in enumerate(factorial.LEVELS.items())
        }
        summaries.append(
            {
                "arm": arm,
                **levels,
                "path": str(ply),
                "candidate_points": 100,
                "accepted_points": 90,
                "policy": levels["point_budget_policy"],
                "input_points": 90,
                "output_points": 80,
                "applied": True,
            }
        )
        result_dir = factorial_dir / "evaluations" / "arms" / arm
        result_dir.mkdir(parents=True)
        result = {
            "schema_version": 1,
            "scene_id": "scene",
            "status": "succeeded",
            "inputs": {
                "reconstruction_ply": str(ply),
                "reconstruction_ply_sha256": factorial.sha256_file(ply),
            },
            "evaluator": {"return_code": 0},
            "metrics": [
                {"tolerance": tolerance, "accuracy": 0.4, "completeness": 0.5, "f1": 0.44}
                for tolerance in TOLERANCES
            ],
        }
        (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (runner / "fusion.json").write_text(json.dumps({"factorial_outputs": summaries}), encoding="utf-8")
    (runner / "consistency.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    reports = root / "reports"
    reports.mkdir(parents=True)
    g1_17 = reports / "g1_17.json"
    g1_17.write_text(json.dumps({"decision_gate": {"passed": False}}), encoding="utf-8")
    g1_20 = reports / "g1_20.json"
    g1_20.write_text(json.dumps({"evaluation": "g1_20"}), encoding="utf-8")
    g1_24 = reports / "g1_24.json"
    g1_24.write_text(
        json.dumps(
            {
                "selection": {"selected_strategy": "manhattan"},
                "metrics": {"rigid_transform_invariants": ["point_count"]},
            }
        ),
        encoding="utf-8",
    )
    private = reports / "private_fusion.json"
    private.write_text(
        json.dumps(
            {
                "cross_view_filter": {
                    "confidence_threshold_scope": "per_frame",
                    "support_policy": "adaptive_two",
                },
                "point_budget": {"policy": "spatial_balanced", "applied": False},
            }
        ),
        encoding="utf-8",
    )
    g1_7 = root / "g1_7"
    paths = [
        g1_7 / scene / "evaluations" / "covisibility" / "result.json"
        for scene in factorial.SCENES
    ] + [
        g1_7 / "private_225" / "evaluations" / name
        for name in ("sequential_g1_4_raw.json", "covisibility_g1_4_raw.json")
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return {
        "benchmark_root": benchmark_root,
        "g1_7_root": g1_7,
        "g1_17_report": g1_17,
        "g1_20_report": g1_20,
        "g1_24_report": g1_24,
        "private_fusion": private,
    }


def run_analysis(inputs: dict[str, Path]) -> dict:
    return factorial.analyze_stage1_factorial(
        benchmark_root=inputs["benchmark_root"],
        scenes=["scene"],
        g1_7_root=inputs["g1_7_root"],
        g1_17_report=inputs["g1_17_report"],
        g1_20_report=inputs["g1_20_report"],
        g1_24_report=inputs["g1_24_report"],
        private_fusion=inputs["private_fusion"],
    )


def cli_args(inputs: dict[str, Path], output: Path) -> list[str]:
    return [
        "--benchmark-root",
        str(inputs["benchmark_root"]),
        "--scenes",
        "scene",
        "--g1-7-root",
        str(inputs["g1_7_root"]),
        "--g1-17-report",
        str(inputs["g1_17_report"]),
        "--g1-20-report",
        str(inputs["g1_20_report"]),
        "--g1-24-report",
        str(inputs["g1_24_report"]),
        "--private-fusion",
        str(inputs["private_fusion"]),
        "--output",
        str(output),
    ]
