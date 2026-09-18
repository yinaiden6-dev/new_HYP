#!/usr/bin/env python3
"""Independent validator for current64 J1 TRAIN32/EVAL32 label ledgers."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any, Mapping
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT_V1_20260904.md"
)
TWO_GATE = (
    ROOT
    / "plan/ROUTEA_N2_CANDIDATE_RECALL_AND_OWNERSHIP_TWO_GATE_ADDENDUM_V1_20260903.md"
)
CURRENT_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"
)
CURRENT_AGGREGATE = CURRENT_ROOT / "validation.json"
SOURCE = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json"
)
SOURCE_VALIDATION = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/independent_validation.json"
)
LABEL_ROOT = ROOT / "results/routea_matched_three_arm_n2_d1_label_join_authority_v1"
LABEL_RESULT = LABEL_ROOT / "result.json"
LABEL_VALIDATION = LABEL_ROOT / "independent_validation.json"
PAIR_FIXED = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1/fixed_pairs.json"
)
PAIR_VALIDATION = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_pair64_postseal_fixed_pairs_v1/independent_validation.json"
)
PAIR_AGGREGATE = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_pair64_training_features_v1/independent_aggregate_validation.json"
)
GALLERY = (
    ROOT.parents[2]
    / "colnomic/difficult/raw_gallery_7b/cache/colnomic_gallery_emb_difficult.pt"
)
REPAIR_REGISTRY = ROOT / "registry/gallery_identity_repair_v1.json"
REPAIR_RUNTIME = ROOT / "src/rc_aslo_xf/gallery_identity_repair.py"
REPAIR_CONTRACT = (
    ROOT / "protocols/L0_C0_GALLERY_IDENTITY_REPAIR_CONTRACT_V1_20260808.json"
)
PRODUCER = (
    ROOT
    / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1.py"
)
OUT_ROOT = (
    ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1"
)
TRAIN = OUT_ROOT / "J1_TRAIN32_LABEL_LEDGER.json"
EVAL = OUT_ROOT / "J1_EVAL32_LABEL_LEDGER.json"
RESULT = OUT_ROOT / "result.json"
OUT = OUT_ROOT / "independent_validation.json"
VERSION = "routea_n2_fresh_d1_matched_three_arm_current64_postseal_join_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_READY"
VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_POSTSEAL_JOIN_VALIDATED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING"
FOLD_HASHES = {
    0: "0a9c72582f0af05b9a83d28a6217ef62670a2a89b7bf433247e35e46d0e983f3",
    1: "c411861a4ca6b8f41792893f747d5f66561af718c6681b3d4c4bbece6b40ab72",
    2: "08b89f23d61012460da86191951202dbdb4e7aa4cffb75e2c673e5bb8c5f9a0f",
    3: "a559a678fe056353de607ba92b2bda1e675470be7a477ccdc54513c1395afb94",
    4: "d6014cb1c141efd7cb57a19eed9692b406efddf8ef0f52449690e06b1090c6b3",
}
EXPECTED = {
    "TRAIN": {
        "query_count": 32,
        "identity_count": 12,
        "supergroup_count": 12,
        "TARGET_WINNER": 27,
        "TARGET_NONWINNER": 5,
        "TARGET_ABSENT": 0,
    },
    "EVAL": {
        "query_count": 32,
        "identity_count": 11,
        "supergroup_count": 11,
        "TARGET_WINNER": 24,
        "TARGET_NONWINNER": 8,
        "TARGET_ABSENT": 0,
    },
}
FIXED_HASHES = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    TWO_GATE: "cb39aeaf3b0826b408b84063f0345bbcb384cb8ceef9ff5ae8760e60d1a86765",
    CURRENT_AGGREGATE: "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    SOURCE: "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    SOURCE_VALIDATION: "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
    LABEL_RESULT: "519cea43968abf8083f3d6c4b98e108d592f29976c470b01f464054d7c9a5861",
    LABEL_VALIDATION: "0b78e01d6940c19db40d606ce6012be190628b31d56e71851c22a97e83cfe3a8",
    PAIR_FIXED: "b89b7b43bf3a74353b2482451d73be999a6e1f36fb81cd4233c5867eb7daccad",
    PAIR_VALIDATION: "786c3fd06dc4cb3f4761ea4783a57c60e55bcce1a6c46dc43e38ab51e11d1633",
    PAIR_AGGREGATE: "90e83f9598f3c9cfcf1a1d80282666be213a70abd98c4f8465ae01a3d6c3074e",
    GALLERY: "11713d62d649143c05899bf89714eb768095b95b69e60792e266e64de17e9fcc",
    REPAIR_REGISTRY: "9dc7df14922b88afeba0ee321c168f92ed363cfc3695c7579e662985d9767c3f",
    REPAIR_RUNTIME: "995c43fe36ef946bef6686af0809b3e1d0f9c758eb01d8784ca8115f1d17a34d",
    REPAIR_CONTRACT: "867128101656c41438e6d5c89695ea78bc922c0a59d5a83f78f567364a3fe650",
}


class ValidationError(RuntimeError):
    pass


def req(x: bool, m: str) -> None:
    if not x:
        raise ValidationError(m)


def sha(p: Path) -> str:
    req(p.is_file() and not p.is_symlink(), f"absent {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def canon(x: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            x, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def logical(x: Mapping[str, Any]) -> str:
    y = dict(x)
    y.pop("logical_sha256", None)
    return canon(y)


def thash(x: torch.Tensor) -> str:
    t = torch.as_tensor(x).detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(json.dumps(list(t.shape), separators=(",", ":")).encode())
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def read(p: Path) -> dict[str, Any]:
    x = json.loads(p.read_text())
    req(isinstance(x, dict), f"root {p}")
    return x


def atomic(p: Path, x: Mapping[str, Any]) -> None:
    req(not p.exists(), f"validation exists {p}")
    s = (
        json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    fd, n = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".partial", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(s)
            f.flush()
            os.fsync(f.fileno())
        os.link(n, p)
    finally:
        Path(n).unlink(missing_ok=True)


def labels() -> tuple[str, ...]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from rc_aslo_xf.n2_corrected_d1_runtime_v1 import (
        corrected_labels_from_legacy,
        validate_corrected_identity_axis,
    )

    x = corrected_labels_from_legacy(
        torch.load(GALLERY, map_location="cpu", weights_only=False, mmap=True)["setids"]
    )
    a = validate_corrected_identity_axis(x)
    req(
        a["corrected_identity_count"] == 5412
        and a["corrected_axis_sha256"]
        == "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "axis",
    )
    return x


def current() -> (
    tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]
):
    agg = read(CURRENT_AGGREGATE)
    src = read(SOURCE)
    srcv = read(SOURCE_VALIDATION)
    req(
        agg.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
        and all(agg.get("checks", {}).values())
        and agg.get("logical_sha256") == logical(agg)
        and src.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and srcv.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and srcv.get("manifest_sha256") == sha(SOURCE),
        "current authority",
    )
    seals = {int(x["shard"]): x for x in agg["shards"]}
    records = []
    bindings = []
    for s in range(8):
        d = CURRENT_ROOT / f"shard{s:02d}"
        pp = d / "payload.pt"
        rp = d / "receipt.json"
        vp = d / "validation.json"
        z = seals[s]
        req(
            z["payload_sha256"] == sha(pp)
            and z["receipt_sha256"] == sha(rp)
            and z["validation_sha256"] == sha(vp),
            f"shard{s}",
        )
        payload = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
        records.extend(payload["records"])
        bindings.append(
            {
                "shard": s,
                "payload_sha256": z["payload_sha256"],
                "receipt_sha256": z["receipt_sha256"],
                "validation_sha256": z["validation_sha256"],
            }
        )
    by = {str(x["query_id"]): x for x in src["records"]}
    req(
        len(records) == len(by) == 64
        and {str(x["query_id"]) for x in records} == set(by),
        "population",
    )
    return records, by, bindings


def consensus(
    qids: set[str], source: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    lr, lv = read(LABEL_RESULT), read(LABEL_VALIDATION)
    req(
        lr.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_READY"
        and lv.get("status") == "ROUTEA_N2_D1_LABEL_JOIN_AUTHORITY_VALIDATED"
        and lv.get("result_sha256") == sha(LABEL_RESULT)
        and all(lv.get("checks", {}).values()),
        "labels",
    )
    seals = {int(x["fold"]): x for x in lv["folds"]}
    occ = defaultdict(list)
    bindings = []
    for f in range(5):
        p = LABEL_ROOT / f"fold_{f}/payload.pt"
        req(sha(p) == FOLD_HASHES[f] == seals[f]["payload_sha256"], f"fold{f}")
        payload = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        for x in payload["records"]:
            if x["query_id"] in qids:
                occ[str(x["query_id"])].append((f, x))
        bindings.append({"fold": f, "payload_sha256": FOLD_HASHES[f]})
    out = {}
    for q in qids:
        s = source[q]
        h = int(s["heldout_fold"])
        rows = occ[q]
        req(len(rows) == 4 and {f for f, _ in rows} == set(range(5)) - {h}, f"occ {q}")
        vals = {
            (
                str(x["target_identity"]),
                str(x["supergroup"]),
                int(x["target_physical_row"]),
                int(x["query_ordinal"]),
                int(x["heldout_fold"]),
                str(x["track"]),
            )
            for _, x in rows
        }
        req(len(vals) == 1, f"cons {q}")
        t, g, tr, o, h2, track = next(iter(vals))
        req(o == int(s["oof_query_ordinal"]) and h2 == h, f"role {q}")
        out[q] = {
            "target_identity": t,
            "target_supergroup": g,
            "target_physical_row_provenance": tr,
            "canonical_query_ordinal": o,
            "canonical_heldout_fold": h,
            "canonical_track": track,
            "source_folds": sorted(f for f, _ in rows),
        }
    return out, bindings


def expected_ledger(
    role: str,
    records: list[dict[str, Any]],
    targets: Mapping[str, Mapping[str, Any]],
    lab: tuple[str, ...],
) -> dict[str, Any]:
    out = []
    for x in sorted(
        (r for r in records if r["role_membership"] == role),
        key=lambda r: int(r["execution_ordinal"]),
    ):
        q = str(x["query_id"])
        t = targets[q]
        axis = list(map(int, x["candidate_physical_rows"]))
        ids = [lab[r] for r in axis]
        ps = [i for i, v in enumerate(ids) if v == t["target_identity"]]
        req(len(ps) <= 1, f"position {q}")
        pos = ps[0] if ps else None
        state = (
            "TARGET_WINNER"
            if pos == 0
            else "TARGET_NONWINNER" if pos is not None else "TARGET_ABSENT"
        )
        req(
            lab[t["target_physical_row_provenance"]] == t["target_identity"]
            and x["base_winner_position"] == 0,
            f"provenance {q}",
        )
        out.append(
            {
                "query_id": q,
                "execution_ordinal": int(x["execution_ordinal"]),
                "canonical_query_ordinal": t["canonical_query_ordinal"],
                "canonical_heldout_fold": t["canonical_heldout_fold"],
                "canonical_track": t["canonical_track"],
                "target_identity": t["target_identity"],
                "target_supergroup": t["target_supergroup"],
                "target_physical_row_provenance": t["target_physical_row_provenance"],
                "target_state": state,
                "target_candidate_position": pos,
                "target_representative_physical_row": (
                    axis[pos] if pos is not None else None
                ),
                "base_winner_position": 0,
                "base_winner_representative_physical_row": axis[0],
                "base_winner_corrected_identity": ids[0],
                "n2_label_source_folds": t["source_folds"],
                "targetfree_record_binding": {
                    "candidate_axis_sha256": x["candidate_axis_sha256"],
                    "base_scores_sha256": thash(x["base_scores"]),
                    "feature_sha256": x["feature_sha256"],
                    "adapted_image_tokens_sha256": x["adapted_image_tokens_sha256"],
                    "physical_row_scores_sha256": x["physical_row_scores_sha256"],
                    "oof_checkpoint_sha256": x["oof_checkpoint_sha256"],
                },
                "target_insertion_count": 0,
            }
        )
    c = Counter(x["target_state"] for x in out)
    i = {x["target_identity"] for x in out}
    g = {x["target_supergroup"] for x in out}
    e = EXPECTED[role]
    req(
        len(out) == e["query_count"]
        and len(i) == e["identity_count"]
        and len(g) == e["supergroup_count"]
        and all(
            c.get(s, 0) == e[s]
            for s in ("TARGET_WINNER", "TARGET_NONWINNER", "TARGET_ABSENT")
        ),
        f"{role} gate",
    )
    policy = {
        "TRAIN": {
            "readable_by": ["T0_NATIVE7_TRAINING"],
            "forbidden_to": ["A0_TARGET_FREE_ACTION_SEAL", "E0_POSTSEAL_EVALUATION"],
        },
        "EVAL": {
            "readable_by": ["E0_POSTSEAL_EVALUATION"],
            "forbidden_to": ["T0_NATIVE7_TRAINING", "A0_TARGET_FREE_ACTION_SEAL"],
        },
    }[role]
    v = {
        "version": VERSION,
        "status": f"ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_J1_{role}32_LABEL_LEDGER_READY",
        "claim_level": f"POSTSEAL_{role}32_LABEL_LEDGER_NO_MODEL_OR_ACTION",
        "role": role,
        "population": {
            "query_count": len(out),
            "identity_count": len(i),
            "supergroup_count": len(g),
            "target_winner_count": c.get("TARGET_WINNER", 0),
            "target_nonwinner_count": c.get("TARGET_NONWINNER", 0),
            "target_absent_count": c.get("TARGET_ABSENT", 0),
        },
        "records": out,
        "record_sequence_sha256": canon(out),
        "access_policy": policy,
        "target_insertion_count": 0,
        "model_update_count": 0,
        "logical_sha256": "",
    }
    v["logical_sha256"] = logical(v)
    return v


def main() -> None:
    for path, expected in FIXED_HASHES.items():
        req(sha(path) == expected, f"fixed hash {path}")
    records, source, shards = current()
    targets, folds = consensus({str(x["query_id"]) for x in records}, source)
    lab = labels()
    train = expected_ledger("TRAIN", records, targets, lab)
    evals = expected_ledger("EVAL", records, targets, lab)
    observed_train, observed_eval, result = read(TRAIN), read(EVAL), read(RESULT)
    req(observed_train == train and observed_eval == evals, "physical ledgers differ")
    pair = read(PAIR_FIXED)
    sets = [
        (
            {x["query_id"] for x in pair["records"]},
            {x["target_identity"] for x in pair["records"]},
            {x["target_supergroup"] for x in pair["records"]},
        ),
        (
            {x["query_id"] for x in train["records"]},
            {x["target_identity"] for x in train["records"]},
            {x["target_supergroup"] for x in train["records"]},
        ),
        (
            {x["query_id"] for x in evals["records"]},
            {x["target_identity"] for x in evals["records"]},
            {x["target_supergroup"] for x in evals["records"]},
        ),
    ]
    req(
        all(
            not (sets[a][k] & sets[b][k])
            for a, b in ((0, 1), (0, 2), (1, 2))
            for k in range(3)
        ),
        "disjointness",
    )
    expected_bindings = {
        "contract_sha256": sha(CONTRACT),
        "two_gate_sha256": sha(TWO_GATE),
        "current64_aggregate_sha256": sha(CURRENT_AGGREGATE),
        "current64_aggregate_logical_sha256": read(CURRENT_AGGREGATE)["logical_sha256"],
        "current64_source_sha256": sha(SOURCE),
        "current64_source_validation_sha256": sha(SOURCE_VALIDATION),
        "n2_label_result_sha256": sha(LABEL_RESULT),
        "n2_label_validation_sha256": sha(LABEL_VALIDATION),
        "n2_label_fold_payloads": folds,
        "pair64_fixed_pairs_sha256": sha(PAIR_FIXED),
        "pair64_validation_sha256": sha(PAIR_VALIDATION),
        "pair64_training_input_aggregate_sha256": sha(PAIR_AGGREGATE),
        "corrected_axis_sha256": "935ce029e3c8177fd9bc4b51f7b41fff8e241c3d587977c31d516b2f300efca4",
        "current64_shards": shards,
        "producer_sha256": sha(PRODUCER),
    }
    expected_access = {
        "targetfree_current64_query_count": 64,
        "n2_label_record_occurrence_count": 256,
        "target_identity_read_count": 64,
        "target_supergroup_read_count": 64,
        "target_physical_row_provenance_read_count": 64,
        "historical_candidate_position_read_count": 0,
        "target_insertion_count": 0,
        "model_forward_count": 0,
        "model_update_count": 0,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }
    req(
        result.get("status") == READY
        and result.get("train32_ledger", {}).get("sha256") == sha(TRAIN)
        and result.get("eval32_ledger", {}).get("sha256") == sha(EVAL)
        and result.get("population", {}).get("TRAIN32") == train["population"]
        and result.get("population", {}).get("EVAL32") == evals["population"]
        and all(
            v == 0
            for block in result.get("three_population_overlap_counts", {}).values()
            for v in block.values()
        )
        and result.get("bindings") == expected_bindings
        and result.get("access") == expected_access
        and result.get("logical_sha256") == logical(result)
        and result.get("next_authorized_stage") == NEXT,
        "result",
    )
    checks = {
        "producer_not_imported": True,
        "current64_targetfree_aggregate_and_shards": True,
        "n2_fourfold_consensus_64x4": True,
        "corrected_5412_target_positions": True,
        "train32_population_27_5_0": True,
        "eval32_population_24_8_0": True,
        "physical_train_eval_ledgers_distinct": TRAIN.resolve() != EVAL.resolve()
        and sha(TRAIN) != sha(EVAL),
        "three_population_query_identity_group_disjoint": True,
        "target_insertion_zero": True,
        "no_model_update": True,
        "access_allowlists": train["access_policy"]["readable_by"]
        == ["T0_NATIVE7_TRAINING"]
        and evals["access_policy"]["readable_by"] == ["E0_POSTSEAL_EVALUATION"],
        "no_scientific_claim": result.get("scientific_GO_or_NO_GO") is None
        and result.get("ownership_GO_or_NO_GO") is None,
    }
    req(all(checks.values()), "checks")
    out = {
        "version": VERSION,
        "status": VALIDATED,
        "claim_level": "INDEPENDENT_CURRENT64_POSTSEAL_LABEL_JOIN_VALIDATION_NO_MODEL_OR_ACTION",
        "checks": checks,
        "population": result["population"],
        "three_population_overlap_counts": result["three_population_overlap_counts"],
        "train32_ledger_sha256": sha(TRAIN),
        "train32_ledger_logical_sha256": train["logical_sha256"],
        "eval32_ledger_sha256": sha(EVAL),
        "eval32_ledger_logical_sha256": evals["logical_sha256"],
        "result_sha256": sha(RESULT),
        "result_logical_sha256": result["logical_sha256"],
        "producer_sha256": sha(PRODUCER),
        "validator_sha256": sha(Path(__file__).resolve()),
        "bindings": {
            "contract_sha256": sha(CONTRACT),
            "current64_aggregate_sha256": sha(CURRENT_AGGREGATE),
            "current64_shards": shards,
            "n2_label_fold_payloads": folds,
            "pair64_fixed_pairs_sha256": sha(PAIR_FIXED),
            "pair64_training_input_aggregate_sha256": sha(PAIR_AGGREGATE),
        },
        "target_insertion_count": 0,
        "model_update_count": 0,
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "logical_sha256": "",
    }
    out["logical_sha256"] = logical(out)
    atomic(OUT, out)
    print(json.dumps({"status": VALIDATED, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
