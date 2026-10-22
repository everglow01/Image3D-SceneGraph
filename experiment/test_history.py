"""研究归档的字节完整性、原章节跳转与维护退役边界检查。"""
from pathlib import Path
import hashlib
import json
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "experiment/mcmc_shard_imbalance/history"


def payload(text, ident):
    begin = f"<!-- archive:{ident}:begin -->\n"
    end = f"<!-- archive:{ident}:end -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"归档片段边界不唯一：{ident}")
    return text.split(begin, 1)[1].split(end, 1)[0]


class ResearchHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HISTORY / "archive-manifest.json").read_text())

    def test_archived_and_retained_fragments_reconstruct_region(self):
        counts = []
        all_ids = set()
        for source in self.manifest["sources"]:
            pieces = []
            archived = retained = 0
            for part in source["parts"]:
                self.assertNotIn(part["id"], all_ids)
                all_ids.add(part["id"])
                if part["disposition"] == "archived":
                    text = (ROOT / part["archive_path"]).read_text()
                    body = payload(text, part["id"])
                    self.assertIn(f'<a id="{part["anchor"]}"></a>', text)
                    archived += 1
                else:
                    text = (ROOT / source["path"]).read_text()
                    entries = re.findall(r"^- 20\d\d-.*?(?=^- 20\d\d-|\Z)", text, re.M | re.S)
                    matches = [e for e in entries if e.startswith(part["start_prefix"])]
                    self.assertEqual(len(matches), 1)
                    body = matches[0]
                    retained += 1
                self.assertEqual(len(body.encode()), part["bytes"])
                self.assertEqual(hashlib.sha256(body.encode()).hexdigest(), part["sha256"])
                pieces.append(body)
            region = "".join(pieces).encode()
            self.assertEqual(len(region), source["region_bytes"])
            self.assertEqual(hashlib.sha256(region).hexdigest(), source["region_sha256"])
            self.assertEqual(archived, source["archived_parts"])
            self.assertEqual(retained, source["retained_parts"])
            counts.append((archived, retained))
        self.assertEqual(counts, [(72, 1), (19, 0), (5, 0)])

    def test_gateway_headings_links_and_order(self):
        for source in self.manifest["sources"]:
            current = (ROOT / source["path"]).read_text()
            prior = -1
            archive_positions = []
            for part in source["parts"]:
                if part["disposition"] != "archived":
                    continue
                archive = (ROOT / part["archive_path"]).read_text()
                archive_positions.append(archive.index(f'<!-- archive:{part["id"]}:begin -->'))
                if "heading" in part:
                    heading = part["heading"] + "\n"
                    self.assertEqual(current.count(heading), 1)
                    position = current.index(heading)
                    self.assertGreater(position, prior)
                    prior = position
                    self.assertIn(part["source_redirect"], current)
                    target = re.search(r"\]\(([^)]+)\)", part["source_redirect"]).group(1)
                    path, anchor = target.split("#", 1)
                    resolved = ((ROOT / source["path"]).parent / path).resolve()
                    self.assertEqual(resolved, (ROOT / part["archive_path"]).resolve())
                    self.assertEqual(anchor, part["anchor"])
            self.assertEqual(archive_positions, sorted(archive_positions))
            if "source_gateway" in source:
                self.assertIn(source["source_gateway"], current)

    def test_retirement_is_not_deletion_or_new_runtime(self):
        self.assertEqual(self.manifest["deleted_files"], [])
        self.assertFalse(self.manifest["runtime_code_changed"])
        self.assertFalse(self.manifest["remote_accessed"])
        self.assertTrue(self.manifest["historical_constraints_remain_binding"])
        folder = ROOT / "experiment/mcmc_shard_imbalance/branches/b1_observation_reliability/experiments/e004_fixed_b0"
        meta = json.loads((folder / "experiment.json").read_text())
        source = json.loads((folder / "analysis-source.json").read_text())
        self.assertEqual(ROOT / meta["retirement_record"], HISTORY / "archive-manifest.json")
        self.assertEqual(len(self.manifest["maintenance_retirements"]), 2)
        for retired in self.manifest["maintenance_retirements"]:
            self.assertEqual(retired["status"], "historical_frozen_not_maintained")
            self.assertEqual(retired["physical_action"], "retained_unchanged")
            self.assertEqual(retired["replacement"], meta["active_analysis"]["entrypoint"])
            self.assertIn(retired["path"], source["legacy_sources"])
            # 其他 checkout 可缺 Git 忽略的证据；严格可用性由 catalog 检查负责。
            path = ROOT / retired["path"]
            if path.exists():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 source["legacy_sources"][retired["path"]]["sha256"])

    def test_fragment_boundary_rejection(self):
        for text in ("missing", "<!-- archive:x:begin -->\na\n", "<!-- archive:x:begin -->\na\n<!-- archive:x:end --><!-- archive:x:end -->"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                payload(text, "x")


if __name__ == "__main__":
    unittest.main()
