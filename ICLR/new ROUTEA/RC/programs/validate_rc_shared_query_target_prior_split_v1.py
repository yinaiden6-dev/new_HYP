#!/usr/bin/env python3
"""Qualification-only replay of fixed PAIR64/FULL64 target split boundaries.

No scoring, fitting, split changes, or target identity details are published.
The existing PAIR label identifies its target as winner for HOLD, and its
single challenger for SWITCH.  That physical reference is checked against
the separately hash-bound role identity before identity overlap is measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "registry/rc_shared_query_target_prior_split_qualification_v1_20260909.json"
PINS = {
    "pair_payload": ("results/routea_matched_three_arm_pair64_training_features_v2/payload.pt", "d7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3"),
    "pair_validation": ("results/routea_matched_three_arm_pair64_training_features_v2/independent_validation.json", "657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b"),
    "pair_lineage": ("results/routea_matched_three_arm_pair64_training_features_v2/post_full_lineage.json", "76be3c0cfc839cbe2029e8f7b25f030be15fdb5c37c752ccf9e21e248da33d4e"),
    "pair_producer": ("programs/materialize_routea_matched_three_arm_pair64_training_features_v1.py", "be87a64c708add04f67b664e57cac5caa78b3892d5bf4000b45ae220039ef1ef"),
    "pair_contract": ("plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md", "9ba48038dfa2fb73df15bdb7e47ffbe9596271c5a87497eaf63d3aeb7a95ddd5"),
    "full_manifest": ("results/rc_original_raw_visibility_pv_inputs_v1/manifest.json", "2777f72c1eacc3dd46f1ab4ba7f18c0dda23ddcb95a51f4b87c9fecd67dec2d4"),
    "role_manifest": ("results/cw0_rgh_xf_v2_p0_a0_manifest_v2/role_manifest.json", "2f104f4fbf71bada1b6186fa3d0915fa7f8059c65d6798414ab00043e5835454"),
}
GALLERY_PINS = {
    "gallery_cache_sha256": "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    "raw_path_sequence_sha256": "a15c7a20ca7f1bb26dacb7053d9f896df2d4ad7f6c4408545b8251c08a789b32",
    "legacy_setid_sequence_sha256": "59305a7b787fc1b61277c03fb99154c9edf5cbaec5a84366432b26cfae47dc80",
    "corrected_mapping_sha256": "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
    "repair_contract_sha256": "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
    "repair_manifest_sha256": "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def safe(path):
    path = Path(path)
    need(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), "UNSAFE_PATH")
    need(not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "d1_minimal_intervention", "grozi", "gisc_prerecall_universe")), "PROTECTED_PATH")
    return path


def sha(path):
    digest = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def logical(value):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "logical_sha256"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def qualify():
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.conditional_rep_sources import build_gallery_source

    sources = {}
    for name, (relative, expected) in PINS.items():
        need(sha(ROOT / relative) == expected, "SOURCE_DRIFT:" + name)
        sources[name] = {"path": relative, "sha256": expected}
    sources["validator"] = {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha(__file__)}
    pairv = read(ROOT / PINS["pair_validation"][0])
    lineage = read(ROOT / PINS["pair_lineage"][0])
    need(pairv["status"] == "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED" and all(x is True for x in pairv["checks"].values()), "PAIR_VALIDATION")
    need(pairv["v2_payload_sha256"] == lineage["v2_payload_sha256"] == PINS["pair_payload"][1], "PAIR_PAYLOAD_LINEAGE")
    need(lineage["v1_producer_sha256"] == PINS["pair_producer"][1] and lineage["pair_addendum_sha256"] == PINS["pair_contract"][1], "PAIR_SEMANTICS_LINEAGE")
    pair = torch.load(ROOT / PINS["pair_payload"][0], map_location="cpu", mmap=True, weights_only=True)["records"]
    fullm = read(ROOT / PINS["full_manifest"][0])
    need(fullm["status"] == "RC_ORIGINAL_RAW_VISIBILITY_PV_INPUTS_FULL64_EXACT_EXPORT_PASS", "FULL_MANIFEST")
    full = fullm["records"]
    rolem = read(ROOT / PINS["role_manifest"][0])
    need(rolem["logical_sha256"] == logical(rolem) and rolem["role_shard_count"] == 600, "ROLE_MANIFEST")
    roles = {row["execution_ordinal"]: row for row in rolem["shards"]}
    need(set(roles) == set(range(600)), "ROLE_AXIS")
    populations = {
        "PAIR64": pair,
        "FULL_TRAIN32": [r for r in full if r["role"] == "TRAIN"],
        "FULL_EVAL32": [r for r in full if r["role"] == "EVAL"],
    }
    need([len(populations[k]) for k in populations] == [64, 32, 32], "POPULATION")
    group_sets, role_docs, role_sources = {}, {}, []
    for kind, records in populations.items():
        docs = []
        for record in records:
            execution = record["execution_ordinal"]
            seal = roles[execution]
            path = safe(seal["path"])
            need(sha(path) == seal["sha256"], "ROLE_SHA_DRIFT")
            role = read(path)
            need(role["logical_sha256"] == seal["logical_sha256"] == logical(role), "ROLE_LOGICAL_DRIFT")
            need(role["status"] == "RGH_P0_A0_ROLE_SHARD_READY" and role["query_id"] == record["query_id"] and role["execution_ordinal"] == execution, "ROLE_QUERY_JOIN")
            need(role["target_insertion_count"] == role["target_spatial_supervision_count"] == role["raw_d1_field_count"] == 0, "ROLE_SCOPE")
            need(execution not in role_docs, "DUPLICATE_EXECUTION")
            role_docs[execution] = role
            role_sources.append({"path": str(path.relative_to(ROOT)), "sha256": seal["sha256"], "logical_sha256": seal["logical_sha256"]})
            docs.append(role)
        group_sets[kind] = {name: {row[name] for row in docs} for name in ("query_id", "execution_ordinal", "identity", "supergroup")}
        need(len(group_sets[kind]["query_id"]) == len(records), "DUPLICATE_QUERY")
    need(len(role_sources) == 128, "ROLE_SOURCE_COUNT")
    counts = {kind: {name: len(values) for name, values in sets.items()} for kind, sets in group_sets.items()}
    need([(counts[k]["identity"], counts[k]["supergroup"]) for k in populations] == [(20, 20), (12, 12), (11, 11)], "IDENTITY_GROUP_COUNTS")
    overlaps = {a + "__" + b: {name: len(group_sets[a][name] & group_sets[b][name]) for name in group_sets[a]} for a, b in (("PAIR64", "FULL_EVAL32"), ("PAIR64", "FULL_TRAIN32"), ("FULL_TRAIN32", "FULL_EVAL32"))}
    need(all(count == 0 for row in overlaps.values() for count in row.values()), "SPLIT_OVERLAP")
    gallery = build_gallery_source(verify_cache_file_sha256=True)
    gallery_seals = {key: getattr(gallery, key) for key in GALLERY_PINS}
    need(gallery_seals == GALLERY_PINS, "GALLERY_SEAL_DRIFT")
    matches = 0
    for record in pair:
        need(type(record["switch_label"]) is bool and len(record["challenger_positions"]) == 1, "PAIR_LABEL_SCHEMA")
        winner = int(record["base_winner_position"])
        challenger = int(record["challenger_positions"][0])
        need(winner != challenger and len(record["candidate_physical_rows"]) == 128, "PAIR_ORIGINAL_AXIS")
        target = challenger if record["switch_label"] else winner
        physical = int(record["candidate_physical_rows"][target])
        need(gallery.corrected_identities[physical] == role_docs[record["execution_ordinal"]]["identity"], "PAIR_ACTUAL_SUPERVISION_TARGET_MISMATCH")
        matches += 1
    need(matches == 64, "PAIR_TARGET_MATCH_COUNT")
    return {
        "status": "RC_SHARED_QUERY_TARGET_PRIOR_SPLIT_QUALIFICATION_V1_PASS",
        "schema": "rc_shared_query_target_prior_split_qualification_v1",
        "sources": sources,
        "gallery_six_sha256": gallery_seals,
        "role_sources": sorted(role_sources, key=lambda item: item["path"]),
        "role_shards_physically_and_logically_verified": 128,
        "counts": counts,
        "overlap_counts": overlaps,
        "PAIR_actual_supervision_target_role_identity_matches": 64,
        "PAIR_actual_supervision_target_role_identity_mismatches": 0,
        "PAIR_target_reconstruction": "switch_label_false_uses_base_winner_true_uses_unique_challenger",
        "old_protocol_scope": "PAIR_materialization_checked_execution_disjointness;_old_matched_consumer_identity_check_covered_FULL_TRAIN_vs_EVAL;_this_receipt_adds_PAIR_identity_and_supergroup_verification",
        "evaluation_identity_or_target_details_exported": 0,
        "scoring_calls": 0,
        "training_updates": 0,
        "episodes_dropped_or_reassigned": 0,
        "HYP_GO_claimed": False,
        "scientific_GO_or_NO_GO": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("write", "validate"), required=True)
    args = parser.parse_args()
    if args.phase == "write":
        need(not OUT.exists(), "APPEND_ONLY_RECEIPT_EXISTS")
    result = qualify()
    if args.phase == "write":
        with OUT.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        OUT.chmod(0o444)
    else:
        need(read(OUT) == result, "RECEIPT_REPLAY_DRIFT")
    print(json.dumps({"status": result["status"], "receipt_sha256": sha(OUT), "counts": result["counts"], "overlap_counts": result["overlap_counts"], "PAIR_target_matches": result["PAIR_actual_supervision_target_role_identity_matches"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
