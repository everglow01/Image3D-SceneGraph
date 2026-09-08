#!/usr/bin/env python3
"""Add LightGlue only at the frozen 38-frame component boundary, then map once if bridged."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

from image3d_scenegraph.geometry.colmap import (
    resolve_colmap_feature_profile,
    resolve_colmap_geometric_verification,
    resolve_colmap_local_matcher,
    sha256_file,
)
from run_aliked_score_filter_experiment import (
    database_summary,
    map_and_evaluate,
    run_stage,
    write_json,
)


EXPECTED_BOUNDARY_SHA256 = "1beba232feb2512d9ca256fbe2eb12ff3d2bfb47d0d591c0493ab2b007a78228"
PAIR_BASE = 2147483647


def _digest_rows(connection, query, parameters=()):
    digest = hashlib.sha256()
    for row in connection.execute(query, parameters):
        for value in row:
            if value is None:
                payload = b"N"
            elif isinstance(value, bytes):
                payload = b"B" + value
            else:
                payload = type(value).__name__.encode() + b":" + str(value).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    return digest.hexdigest()


def database_contract(connection, target_pairs):
    placeholders = ",".join("?" for _ in target_pairs)
    feature_queries = (
        "SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras ORDER BY camera_id",
        "SELECT image_id,name,camera_id FROM images ORDER BY image_id",
        "SELECT image_id,rows,cols,data FROM keypoints ORDER BY image_id",
        "SELECT image_id,type,rows,cols,data FROM descriptors ORDER BY image_id",
    )
    feature_hashes = [_digest_rows(connection, query) for query in feature_queries]
    return {
        "features_sha256": hashlib.sha256("".join(feature_hashes).encode()).hexdigest(),
        "nontarget_matches_sha256": _digest_rows(
            connection,
            f"SELECT pair_id,rows,cols,data FROM matches WHERE pair_id NOT IN ({placeholders}) ORDER BY pair_id",
            target_pairs,
        ),
        "nontarget_geometries_sha256": _digest_rows(
            connection,
            f"SELECT pair_id,rows,cols,data,config,F,E,H,qvec,tvec FROM two_view_geometries "
            f"WHERE pair_id NOT IN ({placeholders}) ORDER BY pair_id",
            target_pairs,
        ),
    }


def copy_database_for_pairs(source, destination, target_pairs):
    if destination.exists():
        raise ValueError(f"refusing to overwrite database: {destination}")
    with sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True) as old, sqlite3.connect(destination) as new:
        old.backup(new)
        before = database_contract(old, target_pairs)
        placeholders = ",".join("?" for _ in target_pairs)
        target_before = {
            "candidate": dict(old.execute(
                f"SELECT pair_id,rows FROM matches WHERE pair_id IN ({placeholders})", target_pairs
            )),
            "verified": dict(old.execute(
                f"SELECT pair_id,rows FROM two_view_geometries WHERE pair_id IN ({placeholders})", target_pairs
            )),
        }
        if set(target_before["candidate"]) != set(target_pairs) or set(target_before["verified"]) != set(target_pairs):
            raise ValueError("source database does not contain every target pair")
        if any(target_before[table][pair] for table in target_before for pair in target_pairs):
            raise ValueError("target pair is not empty in frozen source")
        new.execute(f"DELETE FROM matches WHERE pair_id IN ({placeholders})", target_pairs)
        new.execute(f"DELETE FROM two_view_geometries WHERE pair_id IN ({placeholders})", target_pairs)
        new.commit()
        if database_contract(new, target_pairs) != before:
            raise ValueError("non-target database contract changed during copy")
        if new.execute(
            f"SELECT count(*) FROM matches WHERE pair_id IN ({placeholders})", target_pairs
        ).fetchone()[0] or new.execute(
            f"SELECT count(*) FROM two_view_geometries WHERE pair_id IN ({placeholders})", target_pairs
        ).fetchone()[0]:
            raise ValueError("target pair deletion failed")
    return before, target_before


def target_pair_results(database, target_pairs):
    placeholders = ",".join("?" for _ in target_pairs)
    with sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        rows = connection.execute(
            f"SELECT m.pair_id,m.rows,t.rows,t.config FROM matches m JOIN two_view_geometries t USING(pair_id) "
            f"WHERE m.pair_id IN ({placeholders}) ORDER BY m.pair_id",
            target_pairs,
        ).fetchall()
    if len(rows) != len(target_pairs):
        raise ValueError("target matcher did not write every requested pair")
    return [{"pair_id": pair, "candidate": candidate, "verified": verified, "config": config}
            for pair, candidate, verified, config in rows]


def run(project, source_root, boundary_path, output):
    project, source_root, boundary_path, output = map(Path.resolve, (project, source_root, boundary_path, output))
    if output.exists():
        raise ValueError(f"refusing to overwrite experiment: {output}")
    if subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"], text=True,
    ).strip():
        raise ValueError("project tracked files must be clean")
    if sha256_file(boundary_path) != EXPECTED_BOUNDARY_SHA256:
        raise ValueError("boundary diagnosis hash mismatch")
    boundary = json.loads(boundary_path.read_text())
    if boundary.get("profile") != "aliked_component_boundary_v1" or not boundary.get("source_unchanged"):
        raise ValueError("boundary diagnosis is not frozen")
    pairs = boundary["local_bridge_pairs"]
    target_pairs = sorted({int(pair["pair_id"]) for pair in pairs})
    if len(pairs) != len(target_pairs) or len(target_pairs) != 27:
        raise ValueError("expected exactly 27 unique local boundary pairs")
    source_db = source_root / "positive/database.db"
    source_db_hash = boundary["source_hashes"]["positive/database.db"]
    if sha256_file(source_db) != source_db_hash:
        raise ValueError("source database hash mismatch")
    wal = Path(str(source_db) + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("source database has nonempty WAL")
    binary = project / "external/colmap-4-cuda/install/bin/colmap"
    feature = resolve_colmap_feature_profile("aliked_n16rot_v1", project)
    matcher = resolve_colmap_local_matcher(feature, "lightglue", project)
    verification = resolve_colmap_geometric_verification("default_v1")
    timestamps = {item["path"]: item["time_seconds"] for item in json.loads(
        (source_root / "selection.json").read_text()
    )["selected"]}
    output.mkdir(parents=True)
    cell = output / "lightglue_boundary"
    cell.mkdir()
    pair_list = output / "pairs.txt"
    pair_list.write_text("".join(f'{pair["left_name"]} {pair["right_name"]}\n' for pair in pairs))
    protocol = {
        "profile": "aliked_boundary_lightglue_v1",
        "hypothesis": "LightGlue on only the 27 <=2s cross-component pairs may add verified bridge edges.",
        "decision": "Run one unchanged incremental Mapper only if at least one target pair has verified matches.",
        "source_database": str(source_db.relative_to(project)),
        "source_database_sha256": source_db_hash,
        "boundary_report": str(boundary_path.relative_to(project)),
        "boundary_report_sha256": EXPECTED_BOUNDARY_SHA256,
        "pair_count": 27,
        "maximum_pair_time_span_seconds": 2.0,
        "matcher": matcher.name,
        "matcher_profile": matcher.profile_id,
        "matcher_model_sha256": matcher.model_sha256,
        "geometric_verification": verification.profile_id,
        "mapper": "incremental",
        "stage_timeout_seconds": 1200,
        "production_default_changed": False,
        "recovery_applied": False,
        "training_started": False,
        "test_rgb_loaded": False,
        "project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
    }
    write_json(output / "protocol.json", protocol)
    record = {**protocol, "status": "failed", "models": [], "mapping_started": False}
    source_before = sha256_file(source_db)
    try:
        database = cell / "database.db"
        contract, target_before = copy_database_for_pairs(source_db, database, target_pairs)
        record["copied_database_contract"] = contract
        record["target_before"] = target_before
        run_stage(cell, "matching", [
            str(binary), "matches_importer", "--database_path", str(database),
            "--match_list_path", str(pair_list), "--match_type", "pairs",
            *matcher.matching_options, *verification.matching_options,
            "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
            "--FeatureMatching.num_threads", "4", "--default_random_seed", "0",
        ], timeout=1200)
        record["target_after"] = target_pair_results(database, target_pairs)
        with sqlite3.connect(source_db.as_uri() + "?mode=ro&immutable=1", uri=True) as source, sqlite3.connect(
            database.as_uri() + "?mode=ro&immutable=1", uri=True
        ) as modified:
            if database_contract(source, target_pairs) != database_contract(modified, target_pairs):
                raise ValueError("matching changed non-target database content")
        record["added_candidate_pair_count"] = sum(pair["candidate"] > 0 for pair in record["target_after"])
        record["added_verified_pair_count"] = sum(pair["verified"] > 0 for pair in record["target_after"])
        record["added_candidate_count"] = sum(pair["candidate"] for pair in record["target_after"])
        record["added_verified_count"] = sum(pair["verified"] for pair in record["target_after"])
        record["database"] = database_summary(database)
        if record["added_verified_pair_count"]:
            record["mapping_started"] = True
            map_and_evaluate(output, cell, database, str(binary), source_root / "images", timestamps, record)
        else:
            record["status"] = "no_verified_bridge"
            record["product_gate_passed"] = False
            record["acceptance_passed"] = False
    except (RuntimeError, ValueError) as exc:
        record["error"] = str(exc)
        write_json(output / "results.json", record)
        raise
    if sha256_file(source_db) != source_before:
        raise ValueError("source database changed during experiment")
    record["source_database_unchanged"] = True
    record["training_started"] = False
    record["test_rgb_loaded"] = False
    write_json(output / "results.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--boundary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.project, args.source_root, args.boundary_report, args.output)
    print(json.dumps({"status": record["status"],
                      "added_verified_pair_count": record["added_verified_pair_count"],
                      "mapping_started": record["mapping_started"],
                      "primary": record.get("primary")}, indent=2))


if __name__ == "__main__":
    main()
