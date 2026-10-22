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
                    "indexed_legacy_not_moved", "analysis_migrated_execution_frozen",
                    "gpu_implementation_migrated_remote_not_run",
                })
                if item["migration_status"] == "indexed_legacy_not_moved":
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
                    self.assertEqual(item["migration_status"], "gpu_implementation_migrated_remote_not_run")
                    self.assertEqual(repository_path(item["gpu_runner"]["entrypoint"]), path.with_name("run.py"))
                    self.assertTrue(path.with_name("run.py").is_file())
                else:
                    self.assertFalse(path.with_name("run.py").exists())

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
            self.assertEqual(record["remote_evidence"]["availability"], "remote_not_rechecked")
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
            self.assertEqual(gpu["remote_validation"], "not_run")
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
