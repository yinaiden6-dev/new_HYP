#!/usr/bin/env python3
"""Seal target-free NATIVE7 actions on the frozen current64 EVAL population."""

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
OUT_ROOT = ROOT / "results/routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1"
OUT = OUT_ROOT / "actions.json"
VERSION = "routea_n2_fresh_d1_matched_three_arm_target_free_actions_v1_20260904"
READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_READY"
NEXT = "N2_FRESH_D1_MATCHED_THREE_ARM_TARGET_FREE_ACTIONS_INDEPENDENT_VALIDATION"
T0_READY = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_COMPLETE"
T0_VALIDATED = "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_NATIVE7_TRAINING_VALIDATED"
ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
PATHS = ("REAL", "C_BIND")
EXPECTED = {
    CONTRACT: "1e129df85e71a14eefa814b477bf304771eeda631d14d4d68cc6c2a3b8342f79",
    CURRENT_VALID: "24ea4c6e416fc9d5e0a57d22aaa957d5bb3f29e4f01022369a62dd7ff48b7859",
    SOURCE: "da7ff4c0ee24bb72d1fa026e3e103e01e3cf3f9cd489e8b33f26835b4065ab58",
    SOURCE_VALID: "654453f5884fc2bc1848a54fe67cde37947f01cdc1b6493d3e764cda4cfb737e",
}


class A0Error(RuntimeError):
    pass


def req(x: bool, m: str) -> None:
    if not x:
        raise A0Error(m)


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
    req(p.is_file() and not p.is_symlink(), f"missing json {p}")
    x = json.loads(p.read_text())
    req(isinstance(x, dict), "json root")
    return x


def atomic(p: Path, x: Mapping[str, Any]) -> None:
    s = (
        json.dumps(x, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        req(
            p.is_file() and not p.is_symlink() and p.read_text() == s,
            "immutable action drift",
        )
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


def action(
    rows: list[int],
    base: list[float],
    winner: int,
    challengers: list[int],
    features: torch.Tensor,
    w: list[float],
    bias: float,
) -> dict[str, Any]:
    req(
        len(rows) == len(base) == 128
        and len(set(rows)) == 128
        and len(challengers) == 127
        and set(challengers) == set(range(128)) - {winner},
        "axis",
    )
    x = torch.as_tensor(features, dtype=torch.float64)
    weight = torch.tensor(w, dtype=torch.float64)
    req(
        x.shape == (127, 6)
        and weight.shape == (6,)
        and bool(torch.isfinite(x).all() and torch.isfinite(weight).all())
        and math.isfinite(bias),
        "head/features",
    )
    logits = (x @ weight + float(bias)).tolist()
    idx = max(range(127), key=lambda i: (float(logits[i]), -rows[challengers[i]]))
    chosen = challengers[idx]
    decision = "SWITCH" if float(logits[idx]) > 0.0 else "HOLD"
    order = sorted(range(128), key=lambda i: (-float(base[i]), rows[i]))
    req(order[0] == winner, "winner/tie")
    final = list(order)
    if decision == "SWITCH":
        final.remove(chosen)
        final.insert(0, chosen)
    return {
        "candidate_logits": [
            {
                "candidate_position": challengers[i],
                "representative_physical_row": rows[challengers[i]],
                "logit": float(logits[i]),
            }
            for i in range(127)
        ],
        "proposed_challenger_position": chosen,
        "proposed_challenger_physical_row": rows[chosen],
        "strongest_logit": float(logits[idx]),
        "decision": decision,
        "final_ranking_positions": final,
        "final_ranking_physical_rows": [rows[i] for i in final],
    }


def reordered_replay(
    rows: list[int],
    base: list[float],
    winner: int,
    challengers: list[int],
    features: torch.Tensor,
    w: list[float],
    bias: float,
) -> dict[str, Any]:
    perm = list(reversed(range(128)))
    old_to_new = {old: new for new, old in enumerate(perm)}
    nr = [rows[i] for i in perm]
    nb = [base[i] for i in perm]
    nw = old_to_new[winner]
    by_old = {challengers[i]: torch.as_tensor(features)[i] for i in range(127)}
    nc = sorted(set(range(128)) - {nw})
    nf = torch.stack([by_old[perm[new]] for new in nc])
    return action(nr, nb, nw, nc, nf, w, bias)


def load_heads() -> (
    tuple[dict[str, tuple[list[float], float]], dict[str, Any], dict[str, Any]]
):
    t, v = read(T0), read(T0_VALID)
    req(
        t.get("status") == T0_READY
        and t.get("logical_sha256") == logical(t)
        and v.get("status") == T0_VALIDATED
        and v.get("result_sha256") == sha(T0)
        and v.get("result_logical_sha256") == t["logical_sha256"]
        and v.get("logical_sha256") == logical(v)
        and all(v.get("checks", {}).values()),
        "T0 seal",
    )
    heads = {}
    req(set(t.get("heads", {})) == set(ARMS), "head arms")
    for arm, h in t["heads"].items():
        w = list(map(float, h["weight"]))
        b = float(h["bias"])
        req(
            len(w) == 6
            and h.get("parameter_count") == 7
            and h.get("parameter_sha256") == psha(w, b)
            and all(map(math.isfinite, w))
            and math.isfinite(b),
            f"head {arm}",
        )
        heads[arm] = (w, b)
    return heads, t, v


def build() -> dict[str, Any]:
    for p, h in EXPECTED.items():
        req(sha(p) == h, f"fixed hash {p}")
    cur, src, srcv = read(CURRENT_VALID), read(SOURCE), read(SOURCE_VALID)
    req(
        cur.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_PREJOIN_AGGREGATE_VALIDATED"
        and cur.get("logical_sha256") == logical(cur)
        and all(cur.get("checks", {}).values()),
        "current aggregate",
    )
    req(
        src.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_READY"
        and src.get("logical_sha256") == logical(src)
        and srcv.get("status")
        == "ROUTEA_N2_FRESH_D1_MATCHED_THREE_ARM_CURRENT64_SOURCE_MANIFEST_VALIDATED"
        and srcv.get("manifest_sha256") == sha(SOURCE)
        and srcv.get("logical_sha256") == logical(srcv)
        and all(srcv.get("checks", {}).values()),
        "source seal",
    )
    heads, t0, t0v = load_heads()
    roles = {str(x["query_id"]): str(x["role"]) for x in src["records"]}
    records = []
    shards = []
    seals = {int(x["shard"]): x for x in cur["shards"]}
    for s in range(8):
        d = CURRENT / f"shard{s:02d}"
        pp = d / "payload.pt"
        rp = d / "receipt.json"
        vp = d / "validation.json"
        seal = seals[s]
        req(
            sha(pp) == seal["payload_sha256"]
            and sha(rp) == seal["receipt_sha256"]
            and sha(vp) == seal["validation_sha256"],
            f"shard{s} seal",
        )
        payload = torch.load(pp, map_location="cpu", weights_only=False, mmap=True)
        for r in payload["records"]:
            q = str(r["query_id"])
            req(roles[q] == r["role_membership"], "role mismatch")
            if roles[q] != "EVAL":
                continue
            rows = list(map(int, r["candidate_physical_rows"]))
            base = torch.as_tensor(r["base_scores"], dtype=torch.float64).tolist()
            winner = int(r["base_winner_position"])
            chall = list(map(int, r["challenger_positions"]))
            paths = {}
            for arm in ARMS:
                paths[arm] = {}
                for path, key in (
                    ("REAL", "real_native_features"),
                    ("C_BIND", "cbind_native_features"),
                ):
                    w, b = heads[arm]
                    a = action(rows, base, winner, chall, r[key][arm], w, b)
                    z = reordered_replay(rows, base, winner, chall, r[key][arm], w, b)
                    req(
                        a["decision"] == z["decision"]
                        and a["proposed_challenger_physical_row"]
                        == z["proposed_challenger_physical_row"]
                        and a["final_ranking_physical_rows"]
                        == z["final_ranking_physical_rows"],
                        "candidate reorder",
                    )
                    paths[arm][path] = a
                req(
                    torch.equal(
                        torch.as_tensor(r["real_native_features"][arm])[:, 0],
                        torch.as_tensor(r["cbind_native_features"][arm])[:, 0],
                    ),
                    "C_BIND gap",
                )
            base_order = sorted(range(128), key=lambda i: (-base[i], rows[i]))
            records.append(
                {
                    "execution_ordinal": int(r["execution_ordinal"]),
                    "query_id": q,
                    "canonical_heldout_fold": int(r["heldout_fold"]),
                    "candidate_physical_rows": rows,
                    "base_scores": [float(x) for x in base],
                    "base_winner_position": winner,
                    "base_ranking_positions": base_order,
                    "base_ranking_physical_rows": [rows[i] for i in base_order],
                    "actions": paths,
                    "candidate_reorder_invariance": True,
                    "cbind_base_and_gap_unchanged": True,
                }
            )
        shards.append(
            {
                "shard": s,
                "payload_sha256": sha(pp),
                "receipt_sha256": sha(rp),
                "validation_sha256": sha(vp),
            }
        )
    records.sort(key=lambda x: x["execution_ordinal"])
    req(len(records) == 32 and len({x["query_id"] for x in records}) == 32, "EVAL32")
    out = {
        "version": VERSION,
        "status": READY,
        "claim_level": "TARGET_FREE_EVAL32_ACTION_SEAL_NO_LABELS_CORRECTNESS_OR_SCIENTIFIC_CLAIM",
        "contract_sha256": sha(CONTRACT),
        "population": {
            "eval_query_count": 32,
            "arm_count": 3,
            "path_count": 2,
            "candidate_logit_count": 32 * 3 * 2 * 127,
        },
        "records": records,
        "record_sequence_sha256": canon(
            [[x["execution_ordinal"], x["query_id"]] for x in records]
        ),
        "bindings": {
            "current64_validation_sha256": sha(CURRENT_VALID),
            "current64_validation_logical_sha256": cur["logical_sha256"],
            "current64_shards": shards,
            "source_manifest_sha256": sha(SOURCE),
            "source_manifest_logical_sha256": src["logical_sha256"],
            "source_validation_sha256": sha(SOURCE_VALID),
            "source_validation_logical_sha256": srcv["logical_sha256"],
            "t0_result_sha256": sha(T0),
            "t0_result_logical_sha256": t0["logical_sha256"],
            "t0_validation_sha256": sha(T0_VALID),
            "t0_validation_logical_sha256": t0v["logical_sha256"],
            "producer_sha256": sha(Path(__file__).resolve()),
        },
        "access": {
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
        },
        "checks": {
            "candidate_reorder_invariance": True,
            "cbind_preserves_base_and_gap": True,
            "strict_positive_switch_else_exact_hold": True,
            "one_action_maximum": True,
        },
        "scientific_GO_or_NO_GO": None,
        "ownership_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": NEXT,
        "logical_sha256": "",
    }
    out["logical_sha256"] = logical(out)
    return out


def selftest() -> None:
    rows = list(range(100, 228))
    base = [2.0] + [1.0 - i / 1000 for i in range(127)]
    chall = list(range(1, 128))
    x = torch.zeros(127, 6)
    x[8, 0] = 2
    w = [1, 0, 0, 0, 0, 0]
    a = action(rows, base, 0, chall, x, w, -1)
    z = reordered_replay(rows, base, 0, chall, x, w, -1)
    req(
        a["decision"] == "SWITCH"
        and a["final_ranking_physical_rows"] == z["final_ranking_physical_rows"],
        "switch fixture",
    )
    h = action(rows, base, 0, chall, torch.zeros(127, 6), w, -1)
    expected = [rows[i] for i in sorted(range(128), key=lambda i: (-base[i], rows[i]))]
    req(
        h["decision"] == "HOLD" and h["final_ranking_physical_rows"] == expected,
        "hold fixture",
    )
    print('{"status":"A0_SYNTHETIC_SELF_TEST_PASS"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic-self-test", action="store_true")
    a = ap.parse_args()
    if a.synthetic_self_test:
        selftest()
        return
    x = build()
    atomic(OUT, x)
    print(json.dumps({"status": READY, "queries": 32}, sort_keys=True))


if __name__ == "__main__":
    main()
