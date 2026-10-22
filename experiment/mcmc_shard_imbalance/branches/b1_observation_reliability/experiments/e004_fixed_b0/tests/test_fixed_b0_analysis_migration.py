"""标准库小样本、拒绝边界和已有文本的迁移一致性；不执行旧脚本。"""

from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ENTRY = Path(__file__).resolve().parents[1] / "analyze.py"
REPO = next(p for p in ENTRY.parents if (p / "pyproject.toml").is_file())
OLD = REPO / "outputs/analysis/fixed-b0-results-20261022-v2"
spec = importlib.util.spec_from_file_location("fixed_b0_analysis_migration", ENTRY)
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def point(slot, role, gradients):
    cfg = {"lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}
    branches = {}
    for name, gradient in zip(("a", "b"), gradients):
        m, v = 0.9 * 0.01 + 0.1 * gradient, 0.999 * 0.001 + 0.001 * gradient**2
        sigma = math.sqrt(v / (1 - 0.999**4))
        update = -0.01 * (m / (1 - 0.9**4)) / (sigma + 1e-8)
        branches[name] = {
            "update_param_group": cfg,
            "P0_step": 3,
            "P3_step": 4,
            "P0_parameter": [0.5],
            "P0_exp_avg": [0.01],
            "P0_exp_avg_sq": [0.001],
            "P2_gradient": [gradient],
            "P3_parameter": [0.5 + update],
            "actual_update": [update],
            "parameter_reference_error": [0.0],
            "float64_scalar_reference": [
                {
                    "parameter": 0.5 + update,
                    "update": update,
                    "sigma_over_eps": sigma / 1e-8,
                }
            ],
        }
    a, b = branches["a"], branches["b"]
    return {
        "owner": 0,
        "slot": slot,
        "role": role,
        "parameter": "quats",
        "matched_to": 1 if role == "control" else None,
        "visibility": [True, True],
        "cameras": [],
        "raw_quaternion_norm": 1.0,
        "scale_axis_ratio": 1.0,
        "opacity": 0.1,
        "historical_AA_by_parameter": {
            "quats": {
                "branches": branches,
                "a_minus_b": {
                    "P2_gradient": [a["P2_gradient"][0] - b["P2_gradient"][0]],
                    "P3_parameter": [a["P3_parameter"][0] - b["P3_parameter"][0]],
                },
            }
        },
    }


def sample(root):
    save(root / "incoming/protocol.json", {"fixture": True})
    protocol_sha = analyzer.file_identity(root / "incoming/protocol.json")["sha256"]
    pair = {
        "elements": 1,
        "num_strict_unequal": 1,
        "num_bitwise_unequal": 1,
        "num_mismatch": 0,
        "num_nonfinite": 0,
        "max_abs_error": 1e-9,
        "max_ulp": 1,
        "ulp_histogram": {"1": 1},
    }
    output = {
        "element_count": 1,
        "pairs": {str(i): copy.deepcopy(pair) for i in range(10)},
        "summary": {"unique_changed_elements": 1, "unique_above_tolerance_elements": 0},
        "observed_max_range": 1e-9,
        "native_a_reference_only": [copy.deepcopy(pair)],
    }
    report = {
        "protocol_sha256": protocol_sha,
        "observed_pairwise_status": "numerically_equivalent",
        "ranks": {
            str(r): {
                name: copy.deepcopy(output)
                for name in ("v_means2d", "v_conics", "v_colors", "v_opacities")
            }
            for r in range(2)
        },
        "historical_point_analysis": [
            point(1, "prior_anomaly", (0.1, 0.2)),
            point(2, "control", (0.1, 0.1)),
        ],
        "repeat_point_outputs": [],
        "initializations": {},
    }
    save(root / "incoming/comparison.json", report)
    save(
        root / "incoming/replay/complete.json",
        {"resources": {"per_rank_peak_reserved_bytes": [1, 1]}},
    )
    files = {
        "outputs/analysis/fixed-b0-20261022-v2/"
        + str(p.relative_to(root / "incoming")): analyzer.file_identity(p)
        for p in (root / "incoming").rglob("*")
        if p.is_file()
    }
    save(
        root / "remote-verification.json",
        {"status": "verified", "protocol_sha256": protocol_sha, "files": files},
    )
    save(
        root / "independent-metrics.json",
        {"status": "verified", "independent_tensor_pairs": 80},
    )
    for role in ("execution", "monitor"):
        save(
            root / f"{role}-status.json",
            {"finished": True, "running": False, "exit_code": 0},
        )
    return report


class FixedB0AnalysisMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fixed-b0-analysis-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "input"
        self.root.mkdir()
        self.here = self.base / "experiment"
        self.here.mkdir()

    def test_small_scalar_and_summary(self):
        report = sample(self.root)
        sensitivity = analyzer.compute_sensitivity(report)
        self.assertEqual(len(sensitivity["pairs"]), 1)
        self.assertEqual(sensitivity["pairs"][0]["flat_component"], 0)
        control = sensitivity["pairs"][0]["points"][1]
        self.assertEqual(control["gradient_a_minus_b"], 0)
        self.assertEqual(control["parameter_a_minus_b"], 0)
        self.assertLess(
            sensitivity["independent_float64_reference_max_disagreement"], 1e-12
        )
        outputs = analyzer.summarize(self.root, sensitivity)
        assessment = json.loads(outputs["assessment.json"])
        self.assertEqual(assessment["tensor_pair_comparisons"], 80)
        self.assertEqual(assessment["strict_unequal_pair_elements"], 80)
        self.assertEqual(assessment["above_tolerance_pair_elements"], 0)
        self.assertEqual(assessment["changed_pair_ULP_le_4_fraction"], 1)
        self.assertEqual(
            len(list(csv.DictReader(io.StringIO(outputs["repetition-metrics.csv"])))), 8
        )
        self.assertFalse((self.root / "assessment.json").exists())

    def test_semantic_rejections(self):
        report = sample(self.root)
        changed = copy.deepcopy(report)
        changed["historical_point_analysis"][0]["historical_AA_by_parameter"]["quats"][
            "branches"
        ]["a"]["P3_step"] = 9
        with self.assertRaisesRegex(ValueError, "step differs"):
            analyzer.compute_sensitivity(changed)
        sensitivity = analyzer.compute_sensitivity(report)
        save(
            self.root / "monitor-status.json",
            {"finished": False, "running": True, "exit_code": None},
        )
        with self.assertRaisesRegex(ValueError, "did not complete"):
            analyzer.summarize(self.root, sensitivity)
        save(
            self.root / "independent-metrics.json",
            {"status": "failed", "independent_tensor_pairs": 80},
        )
        with self.assertRaisesRegex(ValueError, "verification incomplete"):
            analyzer.summarize(self.root, sensitivity)

    def test_input_manifest_and_paths(self):
        save(self.root / "incoming/protocol.json", {})
        name = "incoming/protocol.json"
        identity = analyzer.file_identity(self.root / name)
        source = {
            "inputs": {name: identity},
            "source_protocol_sha256": identity["sha256"],
        }
        self.assertEqual(analyzer.verify_inputs(self.root, source), source["inputs"])
        wrong = {**source, "source_protocol_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            analyzer.verify_inputs(self.root, wrong)
        for bad in ("../escape.json", "/tmp/escape.json", "missing.json"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                analyzer.checked_file(self.root, bad)
        save(self.root / "incoming/extra.json", {})
        with self.assertRaisesRegex(ValueError, "file set differs"):
            analyzer.verify_inputs(self.root, source)
        (self.root / "incoming/extra.json").unlink()
        save(self.root / name, {"changed": True})
        with self.assertRaisesRegex(ValueError, "identity differs"):
            analyzer.verify_inputs(self.root, source)
        (self.root / name).unlink()
        with self.assertRaisesRegex(ValueError, "missing input"):
            analyzer.verify_inputs(self.root, source)
        outside = self.base / "outside.json"
        save(outside, {})
        (self.root / name).symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            analyzer.verify_inputs(self.root, source)

    def test_source_manifest_identity(self):
        source = analyzer.load_source()
        self.assertEqual(len(source["inputs"]), 33)
        self.assertEqual(source["source_run_id"], "fixed-b0-results-20261022-v2")
        save(self.here / "analysis-source.json", {"source_run_id": "v1"})
        with (
            patch.object(analyzer, "HERE", self.here),
            self.assertRaisesRegex(ValueError, "manifest identity"),
        ):
            analyzer.load_source()

    def test_output_safety_and_failure_has_no_completion(self):
        with patch.object(analyzer, "HERE", self.here):
            output = self.here / "runs/new"
            analyzer.check_output(self.root, output)
            with self.assertRaisesRegex(ValueError, "overlap"):
                analyzer.check_output(self.here / "runs", output)
            with self.assertRaisesRegex(ValueError, "under this experiment"):
                analyzer.check_output(self.root, self.base / "escape")
            output.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                analyzer.check_output(self.root, output)
            (self.here / "runs/link").symlink_to(self.root, target_is_directory=True)
            with self.assertRaises(ValueError):
                analyzer.check_output(self.root, self.here / "runs/link/new")
            candidate = self.here / "runs/failure"
            with patch.object(
                analyzer,
                "load_source",
                return_value=analyzer.read(ENTRY.parent / "analysis-source.json"),
            ):
                with self.assertRaises(ValueError):
                    analyzer.analyze(self.root, candidate)
            self.assertFalse(candidate.exists())

    def test_help_isolated_and_optimized_rejection(self):
        (self.base / "json.py").write_text("raise RuntimeError('untrusted import')\n")
        before = set(self.base.rglob("*"))
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(ENTRY), "--help"],
            cwd=self.base,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.base.rglob("*")), before)
        output = ENTRY.parent / "runs/optimized-rejection-must-not-exist"
        self.assertFalse(output.exists())
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                "-O",
                str(ENTRY),
                "--input-root",
                str(self.root),
                "--output-dir",
                str(output),
            ],
            cwd=self.base,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing input", result.stderr)
        self.assertFalse(output.exists())

    def test_static_no_heavy_or_execution_dependencies(self):
        tree = ast.parse(ENTRY.read_text())
        allowed = {
            "__future__",
            "argparse",
            "collections",
            "csv",
            "datetime",
            "hashlib",
            "io",
            "json",
            "math",
            "pathlib",
        }
        for node in ast.walk(tree):
            self.assertNotIsInstance(node, ast.Assert)
            if isinstance(node, ast.Import):
                self.assertTrue({a.name for a in node.names} <= allowed)
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "__import__"})

    @unittest.skipUnless(OLD.is_dir(), "旧文本产物不在本机；不冒充历史迁移验证通过")
    def test_historical_text_golden_outputs(self):
        source = analyzer.load_source()
        analyzer.verify_inputs(OLD, source)
        sensitivity = analyzer.compute_sensitivity(
            analyzer.read(OLD / "incoming/comparison.json")
        )
        outputs = analyzer.summarize(OLD, sensitivity)
        for name, text in outputs.items():
            with self.subTest(output=name):
                reference = OLD / name
                self.assertEqual(
                    analyzer.file_identity(reference), source["reference_outputs"][name]
                )
                if name.endswith(".json"):
                    self.assertEqual(json.loads(text), analyzer.read(reference))
                elif name.endswith(".csv"):
                    self.assertEqual(text.encode(), reference.read_bytes())
                else:
                    self.assertEqual(
                        [s for s in text.splitlines() if s.strip()],
                        [s for s in reference.read_text().splitlines() if s.strip()],
                    )


if __name__ == "__main__":
    unittest.main()
