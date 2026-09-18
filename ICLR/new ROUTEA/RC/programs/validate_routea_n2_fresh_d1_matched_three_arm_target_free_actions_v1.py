#!/usr/bin/env python3
"""Independent replay of the target-free NATIVE7 EVAL32 action seal."""

from __future__ import annotations
import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
from typing import Any, Mapping
import torch

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "plan/ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_SEVEN_PARAMETER_CROSSFIT_CONTRACT_V1_20260904.md"
)
CURRENT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_current64_prejoin_v1"
CURRENT_VALID = CURRENT / "validation.json"
SOURCE = (
    ROOT
    / "results/routea_n2_fresh_d1_matched_three_arm_source_manifest_v1/manifest.json"
)
SOURCE_VALID = SOURCE.parent / "independent_validation.json"
T0_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_native7_training_v1"
T0 = T0_ROOT / "result.json"
T0_VALID = T0_ROOT / "independent_validation.json"
PRODUCER = (
    ROOT
    / "programs/materialize_routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1.py"
)
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1"
ACTIONS = OUT_ROOT / "actions.json"
OUT = OUT_ROOT / "independent_validation.json"
VERSION = "routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_READY"
VALID = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_VALIDATED"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_POSTSEAL_EVALUATION_REDUCER"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
EXPECTED = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    CURRENT_VALID: "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    SOURCE: "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    SOURCE_VALID: "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
}
FORBIDDEN = {
    "target_identity",
    "target_position",
    "target_state",
    "target_supergroup",
    "correctness",
    "base_correct",
    "final_correct",
    "outcome",
    "switch_label",
}


class VError(RuntimeError):
    pass


def req(x: bool, m: str) -> None:
    if not x:
        raise VError(m)


def sha(p: Path) -> str:
    req(p.is_file() and not p.is_symlink(), f"missing {p}")
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


def read(p: Path) -> dict[str, Any]:
    x = json.loads(p.read_text())
    req(isinstance(x, dict), "json root")
    return x


def walk(x: Any) -> set[str]:
    if isinstance(x, dict):
        return set(map(str, x)) | set().union(*(walk(v) for v in x.values()), set())
    if isinstance(x, (list, tuple)):
        return set().union(*(walk(v) for v in x), set())
    return set()


def atomic(p: Path, x: Mapping[str, Any]) -> None:
    s = (
        json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        req(p.read_text() == s, "validation drift")
        return
    fd, n = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".partial", dir=p.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(s)
            f.flush()
            os.fsync(f.fileno())
        os.link(n, p)
    finally:
        Path(n).unlink(missing_ok=True)


def psha(w: list[float], b: float) -> str:
    vector = torch.tensor([*[float(x) for x in w], float(b)], dtype=torch.float64)
    digest = hashlib.sha256()
    digest.update(str(vector.dtype).encode())
    digest.update(json.dumps(list(vector.shape), separators=(",", ":")).encode())
    digest.update(vector.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def replay(
    rows: list[int],
    base: list[float],
    winner: int,
    chall: list[int],
    features: torch.Tensor,
    w: list[float],
    bias: float,
) -> dict[str, Any]:
    x = torch.as_tensor(features, dtype=torch.float64)
    z = (x @ torch.tensor(w, dtype=torch.float64) + bias).tolist()
    i = max(range(127), key=lambda k: (float(z[k]), -rows[chall[k]]))
    c = chall[i]
    decision = "SWITCH" if float(z[i]) > 0 else "HOLD"
    order = sorted(range(128), key=lambda k: (-base[k], rows[k]))
    req(order[0] == winner, "winner")
    final = list(order)
    if decision == "SWITCH":
        final.remove(c)
        final.insert(0, c)
    return {
        "candidate_logits": [
            {
                "candidate_position": chall[k],
                "representative_physical_row": rows[chall[k]],
                "logit": float(z[k]),
            }
            for k in range(127)
        ],
        "proposed_challenger_position": c,
        "proposed_challenger_physical_row": rows[c],
        "strongest_logit": float(z[i]),
        "decision": decision,
        "final_ranking_positions": final,
        "final_ranking_physical_rows": [rows[k] for k in final],
    }


def reorder(
    rows: list[int],
    base: list[float],
    winner: int,
    chall: list[int],
    features: torch.Tensor,
    w: list[float],
    bias: float,
) -> dict[str, Any]:
    p = list(reversed(range(128)))
    inv = {old: new for new, old in enumerate(p)}
    by = {chall[i]: torch.as_tensor(features)[i] for i in range(127)}
    nc = sorted(set(range(128)) - {inv[winner]})
    nf = torch.stack([by[p[n]] for n in nc])
    return replay(
        [rows[i] for i in p], [base[i] for i in p], inv[winner], nc, nf, w, bias
    )


def selftest() -> None:
    r = list(range(128))
    b = [2.0] + [1 - i / 1000 for i in range(127)]
    c = list(range(1, 128))
    x = torch.zeros(127, 6)
    x[3, 0] = 2
    w = [1.0, 0, 0, 0, 0, 0]
    a = replay(r, b, 0, c, x, w, -1)
    q = reorder(r, b, 0, c, x, w, -1)
    req(
        a["decision"] == "SWITCH"
        and a["final_ranking_physical_rows"] == q["final_ranking_physical_rows"],
        "fixture",
    )
    print('{"status":"A0_VALIDATOR_SYNTHETIC_SELF_TEST_PASS"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-self-test", action="store_true")
    a = ap.parse_args()
    if a.synthetic_self_test:
        selftest()
        return
    for p, h in EXPECTED.items():
        req(sha(p) == h, f"fixed hash {p}")
    out = read(ACTIONS)
    req(
        out.get("version") == VERSION
        and out.get("status") == READY
        and out.get("logical_sha256") == logical(out)
        and out.get("bindings", {}).get("producer_sha256") == sha(PRODUCER),
        "action envelope",
    )
    req(not (FORBIDDEN & walk(out["records"])), "protected output")
    t, tv = read(T0), read(T0_VALID)
    req(
        t.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_COMPLETE"
        and t.get("logical_sha256") == logical(t)
        and tv.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_VALIDATED"
        and tv.get("result_sha256") == sha(T0)
        and tv.get("result_logical_sha256") == t["logical_sha256"]
        and tv.get("logical_sha256") == logical(tv)
        and all(tv.get("checks", {}).values()),
        "T0",
    )
    heads = {}
    req(set(t["heads"]) == set(ARMS), "arms")
    for arm, h in t["heads"].items():
        w = list(map(float, h["weight"]))
        b = float(h["bias"])
        req(
            len(w) == 6
            and h["parameter_count"] == 7
            and h["parameter_sha256"] == psha(w, b),
            "head",
        )
        heads[arm] = (w, b)
    cur = read(CURRENT_VALID)
    src = read(SOURCE)
    srcv = read(SOURCE_VALID)
    req(
        cur.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
        and cur.get("logical_sha256") == logical(cur)
        and all(cur.get("checks", {}).values())
        and src.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and src.get("logical_sha256") == logical(src)
        and srcv.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and srcv.get("manifest_sha256") == sha(SOURCE)
        and srcv.get("logical_sha256") == logical(srcv)
        and all(srcv.get("checks", {}).values()),
        "inputs",
    )
    roles = {str(x["query_id"]): str(x["role"]) for x in src["records"]}
    expected = []
    used_seals = []
    seals = {int(x["shard"]): x for x in cur["shards"]}
    for s in range(8):
        d = CURRENT / f"shard{s:02d}"
        pp = d / "payload.pt"
        rp = d / "receipt.json"
        vp = d / "validation.json"
        q = seals[s]
        req(
            sha(pp) == q["payload_sha256"]
            and sha(rp) == q["receipt_sha256"]
            and sha(vp) == q["validation_sha256"],
            "shard seal",
        )
        used_seals.append(
            {
                "shard": s,
                "payload_sha256": sha(pp),
                "receipt_sha256": sha(rp),
                "validation_sha256": sha(vp),
            }
        )
        p = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
        for r in p["records"]:
            qid = str(r["query_id"])
            req(roles[qid] == r["role_membership"], "role")
            if roles[qid] != "EVAL":
                continue
            rows = list(map(int, r["candidate_physical_rows"]))
            base = torch.as_tensor(r["base_scores"], dtype=torch.float64).tolist()
            winner = int(r["base_winner_position"])
            chall = list(map(int, r["challenger_positions"]))
            acts = {}
            for arm in ARMS:
                acts[arm] = {}
                for path, key in (
                    ("REAL", "real_native_features"),
                    ("C_BIND", "cbind_native_features"),
                ):
                    w, b = heads[arm]
                    z = replay(rows, base, winner, chall, r[key][arm], w, b)
                    zz = reorder(rows, base, winner, chall, r[key][arm], w, b)
                    req(
                        z["decision"] == zz["decision"]
                        and z["proposed_challenger_physical_row"]
                        == zz["proposed_challenger_physical_row"]
                        and z["final_ranking_physical_rows"]
                        == zz["final_ranking_physical_rows"],
                        "reorder",
                    )
                    acts[arm][path] = z
                req(
                    torch.equal(
                        torch.as_tensor(r["real_native_features"][arm])[:, 0],
                        torch.as_tensor(r["cbind_native_features"][arm])[:, 0],
                    ),
                    "gap",
                )
            order = sorted(range(128), key=lambda i: (-base[i], rows[i]))
            expected.append(
                {
                    "execution_ordinal": int(r["execution_ordinal"]),
                    "query_id": qid,
                    "canonical_heldout_fold": int(r["heldout_fold"]),
                    "candidate_physical_rows": rows,
                    "base_scores": [float(x) for x in base],
                    "base_winner_position": winner,
                    "base_ranking_positions": order,
                    "base_ranking_physical_rows": [rows[i] for i in order],
                    "actions": acts,
                    "candidate_reorder_invariance": True,
                    "cbind_base_and_gap_unchanged": True,
                }
            )
    expected.sort(key=lambda x: x["execution_ordinal"])
    expected_bindings = {
        "current64_validation_sha256": sha(CURRENT_VALID),
        "current64_validation_logical_sha256": cur["logical_sha256"],
        "current64_shards": used_seals,
        "source_manifest_sha256": sha(SOURCE),
        "source_manifest_logical_sha256": src["logical_sha256"],
        "source_validation_sha256": sha(SOURCE_VALID),
        "source_validation_logical_sha256": srcv["logical_sha256"],
        "t0_result_sha256": sha(T0),
        "t0_result_logical_sha256": t["logical_sha256"],
        "t0_validation_sha256": sha(T0_VALID),
        "t0_validation_logical_sha256": tv["logical_sha256"],
        "producer_sha256": sha(PRODUCER),
    }
    expected_access = {
        "eval_role_membership_read_count": 32,
        "train_role_membership_read_count": 32,
        "label_ledger_open_count": 0,
        "pair_or_fixed_pair_open_count": 0,
        "target_identity_read_count": 0,
        "target_position_read_count": 0,
        "correctness_read_count": 0,
        "outcome_read_count": 0,
        "model_update_count": 0,
        "external_read_count": 0,
        "sealed_read_count": 0,
    }
    checks = {
        "producer_not_imported": True,
        "envelope_population_and_sequence": out.get("contract_sha256") == sha(CONTRACT)
        and out.get("population")
        == {
            "eval_query_count": 32,
            "arm_count": 3,
            "path_count": 2,
            "candidate_logit_count": 32 * 3 * 2 * 127,
        }
        and out.get("record_sequence_sha256")
        == canon([[x["execution_ordinal"], x["query_id"]] for x in expected]),
        "bindings_exact": out.get("bindings") == expected_bindings,
        "access_exact": out.get("access") == expected_access,
        "records_exact": out.get("records") == expected,
        "eval32_only": len(expected) == 32 == len({x["query_id"] for x in expected}),
        "all_127_logits_six_paths": out.get("population", {}).get(
            "candidate_logit_count"
        )
        == 32 * 3 * 2 * 127,
        "candidate_reorder_invariance": all(
            x["candidate_reorder_invariance"] for x in expected
        ),
        "cbind_base_gap_unchanged": all(
            x["cbind_base_and_gap_unchanged"] for x in expected
        ),
        "strict_switch_exact_hold": all(
            a["decision"] == ("SWITCH" if a["strongest_logit"] > 0 else "HOLD")
            and (
                a["decision"] != "HOLD"
                or a["final_ranking_physical_rows"] == x["base_ranking_physical_rows"]
            )
            for x in expected
            for arm in ARMS
            for a in x["actions"][arm].values()
        ),
        "no_protected_fields": not (FORBIDDEN & walk(out["records"])),
        "no_claim_or_advance": out.get("scientific_GO_or_NO_GO") is None
        and out.get("ownership_GO_or_NO_GO") is None
        and out.get("automatic_stage_advance") is False
        and out.get("next_authorized_stage")
        == "N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_INDEPENDENT_VALIDATION",
    }
    req(all(checks.values()), "checks")
    v = {
        "version": VERSION,
        "status": VALID,
        "claim_level": "INDEPENDENT_TARGET_FREE_EVAL32_ACTION_REPLAY_NO_LABELS",
        "checks": checks,
        "query_count": 32,
        "candidate_logit_count": 32 * 3 * 2 * 127,
        "actions_sha256": sha(ACTIONS),
        "actions_logical_sha256": out["logical_sha256"],
        "producer_sha256": sha(PRODUCER),
        "validator_sha256": sha(Path(__file__).resolve()),
        "access": out["access"],
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "logical_sha256": "",
    }
    v["logical_sha256"] = logical(v)
    atomic(OUT, v)
    print(json.dumps({"status": VALID, "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
