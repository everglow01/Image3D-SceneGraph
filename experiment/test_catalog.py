"""研究导航检查：只读本地文件，不导入项目、Torch 或执行研究脚本。"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import unittest
import warnings

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiment/mcmc_shard_imbalance"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NODES = {"Q1", "Q2", "Q3", "B1", "B2", "B3"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def repository_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"索引路径必须相对仓库根且不能逃逸：{value}")
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"索引路径越界：{value}")
    return resolved


class ResearchCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = {
            p: read(p) for p in STUDY.rglob("experiment.json")
        }
        cls.records = {
            p: read(p) for p in STUDY.glob("**/records/*.json")
        }

    def test_experiment_identity_and_local_paths(self):
        self.assertTrue(self.experiments)
        ids = [e["id"] for e in self.experiments.values()]
        self.assertEqual(len(ids), len(set(ids)))
        for path, item in self.experiments.items():
            with self.subTest(path=path):
                self.assertEqual(item["schema_version"], 1)
                self.assertIn(item["owner"], NODES)
                self.assertTrue(item["id"].startswith(f'MCMC-{item["owner"]}-'))
                self.assertTrue(set(item["related_questions"]) <= NODES)
                self.assertEqual(repository_path(item["design_document"]), path.with_name("README.md"))
                self.assertIn(item["migration_status"], {
                    "indexed_legacy_not_moved", "analysis_migrated_execution_frozen", "new_experiment_design_only",
                    "gpu_implementation_migrated_remote_not_run",
                    "gpu_implementation_migrated_remote_executed",
                    "new_experiment_implemented_remote_not_run", "new_experiment_remote_executed",
                    "new_experiment_partially_implemented_remote_not_run",
                    "new_experiment_partially_implemented_remote_executed",
                    "new_experiment_implemented_remote_partially_executed",
                })
                if item["migration_status"] in {"indexed_legacy_not_moved", "new_experiment_design_only",
                                                "new_experiment_implemented_remote_not_run", "new_experiment_remote_executed",
                                                "new_experiment_partially_implemented_remote_not_run",
                                                "new_experiment_partially_implemented_remote_executed",
                                                "new_experiment_implemented_remote_partially_executed"}:
                    self.assertNotIn("active_analysis", item)
                else:
                    self.assertIn("active_analysis", item)
                for key in ("legacy_entrypoints", "legacy_tests", "legacy_analysis_scripts",
                            "run_records", "shared_run_records", "reviewed_reports"):
                    for value in item.get(key, []):
                        target = repository_path(value)
                        if value.startswith("outputs/") and not target.exists():
                            continue  # 原始证据缺席由专门的完整性检查报告。
                        self.assertTrue(target.is_file(), value)
                if "gpu_runner" in item:
                    self.assertEqual(item["id"], "MCMC-B1-E004")
                    expected = ("gpu_implementation_migrated_remote_not_run"
                                if item["gpu_runner"]["remote_validation"] == "not_run"
                                else "gpu_implementation_migrated_remote_executed")
                    self.assertEqual(item["migration_status"], expected)
                    self.assertEqual(repository_path(item["gpu_runner"]["entrypoint"]), path.with_name("run.py"))
                    self.assertTrue(path.with_name("run.py").is_file())
                elif "implementation" in item:
                    implementation = item["implementation"]
                    if item["id"] == "MCMC-B1-E007":
                        self.assertIn(item["migration_status"], {"new_experiment_partially_implemented_remote_not_run",
                                                                "new_experiment_partially_implemented_remote_executed",
                                                                "new_experiment_implemented_remote_partially_executed"})
                        self.assertEqual(implementation["phase"], "A_only")
                        if item["migration_status"] == "new_experiment_implemented_remote_partially_executed":
                            stage_b = implementation["stage_B"]
                            self.assertEqual(repository_path(stage_b["entrypoint"]), path.with_name("run_b.py"))
                            if stage_b["remote_validation"] == "not_run":
                                self.assertEqual(implementation["remaining_stage_B"], "implemented_remote_not_run")
                            else:
                                self.assertEqual(stage_b["remote_validation"], "structural_preflight_passed")
                                self.assertIn(stage_b["formal_run"], {"not_run", "calibration_blocked"})
                                expected = ("formal_calibration_blocked" if stage_b["formal_run"] == "calibration_blocked"
                                            else "structural_preflight_passed_formal_not_run")
                                self.assertEqual(implementation["remaining_stage_B"], expected)
                                record_path = repository_path(stage_b["remote_validation_record"])
                                self.assertIn(stage_b["remote_validation_record"], item["run_records"])
                                preflight = self.records[record_path]
                                self.assertEqual(preflight["kind"], "phase_B_synthetic_structural_preflight")
                                self.assertEqual(preflight["execution_status"], "completed")
                                self.assertEqual(preflight["acceptance_status"], "structural_preflight_passed")
                                self.assertIsNone(preflight["protocol_sha256"]["value"])
                                self.assertEqual(set(preflight["panel_statuses"]), {"execution", "monitor"})
                                for panel in preflight["panel_statuses"].values():
                                    self.assertTrue(panel["finished"])
                                    self.assertFalse(panel["running"])
                                    self.assertEqual(panel["exit_code"], 0)
                                validation = preflight["validation"]
                                self.assertEqual(validation["pytest_passed"], 1)
                                self.assertEqual(validation["budget"], {
                                    "synthetic_updates": 8, "training_camera_forwards": 16,
                                    "contribution_forwards": 8, "reference_contribution_forwards": 2,
                                    "loss_backwards": 16, "analytic_autograd_calls": 2,
                                    "optimizer_group_steps": 96, "coordinated_rejection_checks": 1,
                                    "rng_rejection_checks": 1, "real_image_decodes": 0})
                                self.assertEqual(set(validation["runtime"]), {"0", "1"})
                                for key in ("formal_protocol_prepared", "formal_cases_started",
                                            "observer_finite_scope_admitted", "Q1_real_trajectory_analysis_started",
                                            "long_term_causality_proven"):
                                    self.assertFalse(preflight["scientific_boundary"][key])
                                self.assertEqual(preflight["scientific_boundary"]["old_gate1"], "not_reclassified")
                                digest = preflight["preflight_sha256"]
                                source = repository_path(digest["source"])
                                self.assertRegex(digest["value"], SHA256)
                                if source.is_file():
                                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest["value"])
                                    self.assertEqual(read(source)["budget"], validation["budget"])
                                    self.assertEqual(read(source)["implementation_commit"], preflight["source_commit"]["value"])
                            self.assertTrue(stage_b["requires_independent_preflight"])
                            self.assertTrue(stage_b["stage_A_execution_entry_unchanged"])
                            self.assertNotEqual(stage_b["remote_test_opt_in"], implementation["remote_test_opt_in"])
                            for value in [stage_b["entrypoint"], *stage_b["helpers"],
                                          *stage_b["local_tests"], *stage_b["remote_tests"]]:
                                self.assertTrue(repository_path(value).is_file(), value)
                            kinds = {"phase_A_offline_pointwise_masks"}
                            if stage_b["remote_validation"] == "structural_preflight_passed":
                                kinds.add("phase_B_synthetic_structural_preflight")
                            if stage_b.get("formal_run") == "calibration_blocked":
                                kinds.add("phase_B_formal_observer_admission")
                            self.assertEqual({self.records[repository_path(v)]["kind"] for v in item["run_records"]}, kinds)
                        else:
                            self.assertEqual(implementation["remaining_stage_B"], "design_only")
                    elif item["id"] == "MCMC-Q1-E003":
                        self.assertEqual(item["migration_status"], "new_experiment_remote_executed")
                        self.assertEqual(implementation["phase"], "Q1_minimal_observed_description")
                        self.assertEqual(implementation["profile"], "mcmc_minimal_survival_v1")
                        self.assertTrue(implementation["requires_independent_preflight"])
                        self.assertFalse(implementation["full_numerical_noninterference_proven"])
                        self.assertEqual(implementation["old_gate1"], "not_reclassified")
                        self.assertEqual(implementation["E007"], "calibration_blocked_not_reclassified")
                        self.assertEqual(implementation["remote_test_opt_in"], "IMAGE3D_AUTHORIZE_MINIMAL_SURVIVAL_TESTS")
                        self.assertEqual(implementation["remote_validation_implementation_commit"], "3d5a7df2822bd5dfda617ceb890c3d48f39bb004")
                        recovery = implementation["render_recovery"]
                        self.assertEqual(recovery["remote_validation"], "not_run")
                        self.assertEqual(recovery["new_training_updates"], 0)
                        self.assertTrue(recovery["requires_fresh_process_preflight"])
                        self.assertEqual(recovery["original_suite_status"], "failed_at_isolated_render_not_reclassified")
                        self.assertNotEqual(recovery["remote_test_opt_in"], implementation["remote_test_opt_in"])
                        for value in [recovery["entrypoint"], *recovery["helpers"], *recovery["local_tests"], *recovery["remote_tests"]]:
                            self.assertTrue(repository_path(value).is_file(), value)
                    else:
                        self.assertEqual(item["id"], "MCMC-B1-E006")
                        self.assertIn(item["migration_status"], {"new_experiment_implemented_remote_not_run", "new_experiment_remote_executed"})
                    self.assertEqual(repository_path(implementation["entrypoint"]), path.with_name("run.py"))
                    if item["migration_status"] in {"new_experiment_implemented_remote_not_run",
                                                    "new_experiment_partially_implemented_remote_not_run"}:
                        self.assertEqual(implementation["remote_validation"], "not_run")
                        self.assertEqual(item["run_records"], [])
                    elif item["id"] == "MCMC-Q1-E003":
                        self.assertEqual(implementation["remote_validation"], "training_descriptions_verified_render_failed")
                        self.assertEqual(implementation["preflight_status"], "passed")
                        self.assertEqual(implementation["suite_status"], "failed_at_isolated_render")
                        remote_path = repository_path(implementation["remote_validation_record"])
                        self.assertIn(str(remote_path.relative_to(ROOT)), item["run_records"])
                        remote = self.records[remote_path]
                        self.assertEqual(remote["execution_status"], "failed_at_isolated_render")
                        self.assertEqual(remote["acceptance_status"], implementation["remote_validation"])
                        self.assertEqual(set(remote["panel_statuses"]), {"execution", "monitor"})
                        for panel in remote["panel_statuses"].values():
                            self.assertTrue(panel["finished"])
                            self.assertFalse(panel["running"])
                            self.assertEqual(panel["exit_code"], 1)
                        validation = remote["validation"]
                        self.assertTrue(validation["preflight_passed"])
                        self.assertEqual(validation["training_complete"], ["T0", "T1"])
                        self.assertEqual(validation["global_training_updates"], 400)
                        self.assertEqual(validation["render_T0"], "failed_RNG_guard")
                        self.assertEqual(validation["render_T1"], "not_started")
                        self.assertIsNone(validation["isolated_render_forward_count"])
                        self.assertFalse(validation["suite_passed"])
                        self.assertFalse(validation["original_comparison_exists"])
                        self.assertFalse(validation["precise_RNG_source_identified"])
                        self.assertTrue(remote["scientific_boundary"]["training_descriptions_usable_with_stated_limits"])
                        for key in ("whole_suite_passed", "immediate_render_effect_available",
                                    "full_numerical_noninterference_proven", "long_term_causality_proven",
                                    "excess_source_reuse_causality_proven", "retry_started"):
                            self.assertFalse(remote["scientific_boundary"][key])
                        analysis = repository_path(remote["readonly_analysis"]["path"])
                        self.assertEqual(hashlib.sha256(analysis.read_bytes()).hexdigest(), remote["readonly_analysis"]["sha256"])
                        result = read(analysis)
                        self.assertTrue(result["cpu_only_postrun_reanalysis"])
                        self.assertEqual(result["new_GPU_calls"], 0)
                        self.assertEqual(set(result["training"]), {"T0", "T1"})
                        for trajectory in result["training"].values():
                            for owner in trajectory["owners"].values():
                                self.assertEqual(sum(g["count"] for g in owner["groups"].values()), 1500000)
                                for g in owner["groups"].values():
                                    self.assertEqual(g["alive_after_relocation"] + g["not_alive_after_relocation"], g["count"])
                                    self.assertEqual(g["redeath_fraction_of_initially_alive"],
                                                     g["initially_alive_ever_died"] / g["alive_after_relocation"])
                    elif item["id"] == "MCMC-B1-E007":
                        self.assertEqual(implementation["remote_validation"], "phase_A_pass_no_pointwise_threshold_difference")
                        remote_path = repository_path(implementation["remote_validation_record"])
                        self.assertIn(str(remote_path.relative_to(ROOT)), item["run_records"])
                        remote = self.records[remote_path]
                        self.assertEqual(remote["execution_status"], "completed")
                        self.assertEqual(remote["acceptance_status"], implementation["remote_validation"])
                        for panel in remote["panel_statuses"].values():
                            self.assertTrue(panel["finished"])
                            self.assertFalse(panel["running"])
                            self.assertEqual(panel["exit_code"], 0)
                        self.assertEqual(remote["validation"]["phase"], "A_only")
                        self.assertEqual(remote["validation"]["mask_files"], 360)
                        self.assertEqual(remote["validation"]["xor_count_sum"], 0)
                        self.assertEqual(remote["validation"]["budget"]["new_training_updates"], 0)
                        for key in ("phase_B_started", "independent_nonintervention_passed",
                                    "all_ten_cases_cross_group_mask_equivalence_proven",
                                    "actual_strategy_death_masks_observed", "long_term_causality_proven"):
                            self.assertFalse(remote["scientific_boundary"][key])
                    else:
                        self.assertEqual(implementation["remote_validation"], "same_input_pass_connected_differences_observed")
                        remote_path = repository_path(implementation["remote_validation_record"])
                        self.assertIn(str(remote_path.relative_to(ROOT)), item["run_records"])
                        remote = self.records[remote_path]
                        self.assertEqual(remote["execution_status"], "completed")
                        for panel in remote["panel_statuses"].values():
                            self.assertTrue(panel["finished"])
                            self.assertEqual(panel["exit_code"], 0)
                        self.assertEqual(remote["validation"]["cases_completed"], 10)
                        self.assertTrue(remote["validation"]["same_input_gate_passed"])
                        same = remote["validation"]["comparison_summary"]["same"]
                        self.assertEqual(same["P3_parameters"]["status_counts"], {"bitwise_equal": 36})
                        self.assertEqual(same["updates"]["status_counts"], {"bitwise_equal": 36})
                        self.assertFalse(remote["scientific_boundary"]["long_term_causality_proven"])
                    for value in [implementation["entrypoint"], *implementation["helpers"],
                                  *implementation["local_tests"], *implementation["remote_tests"]]:
                        self.assertTrue(repository_path(value).is_file(), value)
                else:
                    self.assertFalse(path.with_name("run.py").exists())

    def test_e007_formal_calibration_blocked(self):
        item = next(e for e in self.experiments.values() if e["id"] == "MCMC-B1-E007")
        stage = item["implementation"]["stage_B"]
        self.assertEqual(stage["formal_run"], "calibration_blocked")
        self.assertIn(stage["formal_run_record"], item["run_records"])
        record = self.records[repository_path(stage["formal_run_record"])]
        self.assertEqual(record["execution_status"], "admission_blocked")
        self.assertEqual(record["acceptance_status"], "baseline_unstable_or_insufficient")
        self.assertEqual(record["preflight_run"], "admission-preflight-20261027-v1")
        self.assertEqual(set(record["panel_statuses"]), {"execution", "monitor"})
        for panel in record["panel_statuses"].values():
            self.assertTrue(panel["finished"])
            self.assertFalse(panel["running"])
            self.assertEqual(panel["exit_code"], 2)
        validation = record["validation"]
        self.assertEqual(validation["completed_cases"], ["C0", "C1", "C2"])
        self.assertEqual(validation["continuous_steps"], {"C0": 200, "C1": 200, "C2": 200})
        self.assertEqual(validation["exceeded"], ["loss/12100", "loss/12200"])
        self.assertEqual(validation["ratio_metrics_within_ceiling"], 12)
        self.assertLessEqual(validation["maximum_ratio_half_width"], .001)
        self.assertTrue(validation["calibration_recomputed_equal"])
        self.assertTrue(validation["no_retries_or_threshold_changes"])
        for row in validation["loss_details"].values():
            self.assertEqual(row["half_width"], max(row["values"].values()) - min(row["values"].values()))
            self.assertEqual(row["ceiling"], 2e-6 + 1e-3 * abs(row["centre"]))
            self.assertGreater(row["half_width"], row["ceiling"])
        for key, expected in {"global_updates": 600, "training_camera_forwards": 1200,
                              "loss_backwards": 1200, "optimizer_group_steps": 7200,
                              "global_relocation_events": 6, "per_owner_relocation_calls": 12,
                              "contribution_forwards": 0}.items():
            self.assertEqual(validation[key], expected)
        for key, value in record["scientific_boundary"].items():
            if key == "old_gate1":
                self.assertEqual(value, "not_reclassified")
            else:
                self.assertIs(value, False)

    def test_design_only_has_no_execution(self):
        for path, item in self.experiments.items():
            if item["migration_status"] != "new_experiment_design_only":
                continue
            with self.subTest(path=path):
                for key in ("implementation", "gpu_runner", "active_analysis", "entrypoint"):
                    self.assertNotIn(key, item)
                for key in ("legacy_entrypoints", "legacy_tests", "legacy_analysis_scripts",
                            "run_records", "shared_run_records", "reviewed_reports"):
                    self.assertEqual(item[key], [])
                self.assertFalse(list(path.parent.glob("*.py")))
                for name in ("records", "reports"):
                    self.assertFalse(path.with_name(name).exists())

    def test_dependencies_and_research_links(self):
        items = {e["id"]: e for e in self.experiments.values()}
        for key, item in items.items():
            for relation in ("depends_on", "informed_by"):
                self.assertEqual(len(item[relation]), len(set(item[relation])))
                for target in item[relation]:
                    self.assertIn(target, items)
                    self.assertNotEqual(key, target)
        visited = set()

        def visit(key, active):
            self.assertNotIn(key, active, f"执行依赖形成循环：{key}")
            if key in visited:
                return
            for target in items[key]["depends_on"]:
                visit(target, active | {key})
            visited.add(key)

        for key in items:
            visit(key, set())

    def test_single_owner_and_shared_runs(self):
        owners = {}
        run_ids = []
        task_ids = []
        for item in self.experiments.values():
            for value in item["run_records"]:
                path = repository_path(value)
                self.assertNotIn(path, owners, f"重复拥有同一次运行：{value}")
                owners[path] = item["id"]
                self.assertEqual(self.records[path]["experiment_id"], item["id"])
            for value in item["shared_run_records"]:
                self.assertIn(repository_path(value), self.records)
        self.assertEqual(set(owners), set(self.records), "存在未归属运行记录")
        for path, record in self.records.items():
            self.assertEqual(path.stem, record["run_id"])
            run_ids.append(record["run_id"])
            task = record["execution_task"]["value"]
            if task:
                self.assertRegex(task, r"^\d{8}-\d{6}-[0-9a-f]+$")
                task_ids.append(task)
            self.assertTrue(record["execution_status"])
            self.assertTrue(record["acceptance_status"])
            remote = record["remote_evidence"]
            self.assertIn(remote["availability"], {"remote_not_rechecked", "remote_verified_at_record_time"})
            if remote["availability"] == "remote_verified_at_record_time":
                self.assertTrue(remote["verified_at_utc"])
        self.assertEqual(len(run_ids), len(set(run_ids)))
        self.assertEqual(len(task_ids), len(set(task_ids)), "同一执行任务被重复登记")
        for record in self.records.values():
            previous = record["previous_attempt"]
            if previous is not None:
                self.assertIn(previous, run_ids)
                self.assertNotEqual(previous, record["run_id"])

    def test_evidence_hashes_and_availability(self):
        missing = set()
        for record in self.records.values():
            self.assertTrue(record["local_evidence"])
            for evidence in record["local_evidence"]:
                path = repository_path(evidence["path"])
                self.assertRegex(evidence["sha256"], SHA256)
                self.assertEqual(evidence["availability"], "local_verified")
                if not path.is_file():
                    missing.add(evidence["path"])
                    continue
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), evidence["sha256"], str(path))
            for field in ("source_commit", "protocol_sha256", "execution_task", "monitor_task"):
                value = record[field]
                if value["value"] is None:
                    self.assertTrue(value.get("missing_reason"), field)
                else:
                    self.assertTrue(value["source"], field)
                    path = repository_path(value["source"])
                    if not path.is_file():
                        missing.add(value["source"])
            protocol = record["protocol_sha256"]
            if protocol["value"] is not None:
                self.assertRegex(protocol["value"], SHA256)
                path = repository_path(protocol["source"])
                if path.is_file() and protocol["field"] == "file_sha256":
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), protocol["value"])
        if missing:
            message = f"{len(missing)} 份旧本地证据缺席，未校验这些文件；不代表远端可用。"
            if os.environ.get("IMAGE3D_REQUIRE_LOCAL_EVIDENCE") == "1":
                self.fail(message + "\n" + "\n".join(sorted(missing)))
            warnings.warn(message, stacklevel=2)

    def test_markdown_links(self):
        for path in (ROOT / "experiment").rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
                if target.startswith(("http://", "https://", "#")):
                    continue
                destination = (path.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(destination.is_relative_to(ROOT), f"仓库外链接：{path}: {target}")
                if destination.is_relative_to(ROOT / "outputs") and not destination.exists():
                    if os.environ.get("IMAGE3D_REQUIRE_LOCAL_EVIDENCE") != "1":
                        continue
                self.assertTrue(destination.exists(), f"失效链接：{path}: {target}")

    def test_partial_analysis_migration(self):
        for item in self.experiments.values():
            if "active_analysis" not in item:
                continue
            active = item["active_analysis"]
            for key in ("entrypoint", "source_manifest", "verification_report", "verification_record"):
                self.assertTrue(repository_path(active[key]).is_file(), active[key])
            for path in active["tests"]:
                self.assertTrue(repository_path(path).is_file(), path)
            manifest_path = repository_path(active["source_manifest"])
            manifest = read(manifest_path)
            self.assertEqual(manifest["source_run_id"], active["supported_source_run"])
            self.assertIn(manifest["source_run_id"], {r["run_id"] for r in self.records.values()})
            entry = repository_path(active["entrypoint"])
            tree = ast.parse(entry.read_text())
            declared_sha = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "SOURCE_SHA256" for t in n.targets))
            actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            self.assertEqual(declared_sha, actual_sha)
            verification = read(repository_path(active["verification_record"]))
            self.assertEqual(verification["status"], "passed")
            self.assertEqual(verification["analysis_code_sha256"], hashlib.sha256(entry.read_bytes()).hexdigest())
            self.assertEqual(verification["analysis_source_sha256"], actual_sha)
            self.assertTrue(all(v["equal"] for v in verification["outputs"].values()))
            data_root = repository_path(active["source_data_root"])
            for name, identity in manifest["legacy_sources"].items():
                path = repository_path(name)
                if not path.is_file() and os.environ.get("IMAGE3D_REQUIRE_LOCAL_EVIDENCE") != "1":
                    continue
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), identity["sha256"])
            if data_root.is_dir() or os.environ.get("IMAGE3D_REQUIRE_LOCAL_EVIDENCE") == "1":
                for name, identity in manifest["inputs"].items():
                    path = data_root / name
                    self.assertTrue(path.resolve().is_relative_to(data_root.resolve()))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), identity["sha256"])

    def test_gpu_runner_identity_without_execution(self):
        for item in self.experiments.values():
            if "gpu_runner" not in item:
                continue
            gpu = item["gpu_runner"]
            for key in ("entrypoint", "source_manifest", "verification_report", "verification_record"):
                self.assertTrue(repository_path(gpu[key]).is_file(), gpu[key])
            for value in gpu["helpers"] + gpu["local_tests"] + gpu["remote_tests"]:
                self.assertTrue(repository_path(value).is_file(), value)
            self.assertIn(gpu["remote_validation"], {"not_run", "numerical_pass_monitor_incomplete"})
            if gpu["remote_validation"] != "not_run":
                remote_path = repository_path(gpu["remote_validation_record"])
                self.assertIn(remote_path, self.records)
                remote = self.records[remote_path]
                self.assertEqual(remote["acceptance_status"], gpu["remote_validation"])
                self.assertEqual(remote["execution_status"], "completed")
                self.assertTrue(remote["panel_statuses"]["execution"]["value"]["finished"])
                self.assertEqual(remote["panel_statuses"]["execution"]["value"]["exit_code"], 0)
                self.assertFalse(remote["panel_statuses"]["monitor"]["value"]["finished"])
                self.assertIsNone(remote["panel_statuses"]["monitor"]["value"]["exit_code"])
                self.assertEqual(remote["validation"]["monitor_terminal_status"], "missing")
                self.assertEqual(remote["validation"]["within_run_tensor_pairs"], 80)
                self.assertEqual(remote["validation"]["old_helper_statistics_rechecked"], 80)
                self.assertEqual(remote["validation"]["new_vs_old_aligned_trial_pairs"], 40)
                for key in ("within_run_above_tolerance", "within_run_nonfinite",
                            "new_vs_old_above_tolerance", "new_vs_old_nonfinite",
                            "new_forward_calls", "new_optimizer_updates"):
                    self.assertEqual(remote["validation"][key], 0)
            source_path = repository_path(gpu["source_manifest"])
            source = read(source_path)
            self.assertEqual(len(source["legacy_scripts"]), 13)
            self.assertEqual(len(source["files"]), 32)
            self.assertTrue(set(source["legacy_scripts"]) <= source["files"].keys())
            for name, expected in source["files"].items():
                path = repository_path(name)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected["sha256"])
                self.assertEqual(path.stat().st_size, expected["bytes"])
            bridge = repository_path(next(p for p in gpu["helpers"] if p.endswith("/legacy.py")))
            tree = ast.parse(bridge.read_text())
            declared = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "MANIFEST_SHA256" for t in n.targets))
            self.assertEqual(declared, hashlib.sha256(source_path.read_bytes()).hexdigest())
            verify_line = next(n.lineno for n in tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "MANIFEST" for t in n.targets))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("scripts"):
                    self.assertGreater(node.lineno, verify_line)
            record = read(repository_path(gpu["verification_record"]))
            self.assertEqual(record["status"], "local_checks_passed_remote_not_run")
            self.assertEqual(record["remote_validation"], "not_run")
            self.assertEqual(record["new_GPU_calls"], 0)
            self.assertEqual(record["deleted_files"], [])
            for name, expected in record["files"].items():
                self.assertEqual(hashlib.sha256(repository_path(name).read_bytes()).hexdigest(), expected)
            runner = ast.parse(repository_path(gpu["entrypoint"]).read_text())
            constants = {n.targets[0].id: n.value.value for n in runner.body
                         if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                         and isinstance(n.value, ast.Constant)}
            self.assertEqual(constants["PROFILE"], gpu["profile"])
            self.assertEqual(constants["UNIT"], gpu["unit"])
            self.assertEqual(constants["MODULE"], gpu["module"])
            self.assertNotEqual(constants["PROFILE"], source["legacy_profile"])

    def test_failed_attempt_not_relabelled(self):
        records = {r["run_id"]: r for r in self.records.values()}
        first = records["fixed-b0-results-20261022-v1"]
        second = records["fixed-b0-results-20261022-v2"]
        self.assertEqual(first["execution_status"], "preflight_failed_real_replay_not_started")
        self.assertIsNone(first["protocol_sha256"]["value"])
        self.assertEqual(second["previous_attempt"], first["run_id"])
        self.assertEqual(second["historical_status"]["old_gate1"], "not_reclassified")
        native = next(e for e in self.experiments.values() if e["id"] == "MCMC-B1-E001")
        self.assertEqual(native["run_records"], [])
        self.assertEqual(len(native["shared_run_records"]), 1)


if __name__ == "__main__":
    unittest.main()
