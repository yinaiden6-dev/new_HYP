#!/usr/bin/env python3
"""Materialize hash-bound, training-only matched A/B/C pair64 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_romav2_colnomic_visibility_xf_six_case_v1 as core  # noqa: E402
from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as frozen  # noqa: E402
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa: E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import (  # noqa: E402
    FrozenColNomicBaseV1,
)
from romav2 import RoMaV2  # noqa: E402


CONTRACT = (
    ROOT
    / "plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md"
)
PRIMARY_CONTRACT = (
    ROOT
    / "plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md"
)
PAIR_V1 = ROOT / "results/romav2_colnomic_visibility_xf_balanced32_v1/result.json"
PAIR_V2 = ROOT / "results/romav2_colnomic_visibility_xf_balanced32_v2/result.json"
PAIR_V1_PRODUCER = ROOT / "programs/run_romav2_colnomic_visibility_xf_balanced32_v1.py"
PAIR_V2_PRODUCER = ROOT / "programs/run_romav2_colnomic_visibility_xf_balanced32_v2.py"
SOURCE_MANIFEST = ROOT / "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json"
GEOMETRY = ROOT / "cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"
FOLDS = ROOT / "protocols/dino_rcde_prejoin_folds_600_v1_2_20260812.json"
CURRENT_ROOT = ROOT / "results/routea_d1_current_runtime_64_prejoin_v1"
CURRENT_VALIDATION = CURRENT_ROOT / "validation.json"
FROZEN_VALIDATION = (
    ROOT / "results/romav2_colnomic_frozen_gate_definition_v1/validation.json"
)
SOURCE_MODULE = ROOT / "src/rc_aslo_xf/rgh_full600_source_v1.py"
BASE_MODULE = ROOT / "src/rc_aslo_xf/rgh_v9_frozen_colnomic_base_v1.py"
FROZEN_MODULE = ROOT / "src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py"
CORE_MODULE = ROOT / "programs/run_romav2_colnomic_visibility_xf_six_case_v1.py"
ROMA_WEIGHTS = WORKSPACE / "third_party/model_cache/torch/hub/checkpoints/romav2.0.1.pt"
OUT_ROOT = ROOT / "results/routea_matched_three_arm_pair64_training_features_v1"

ARMS = ("A_ALL", "B_QUERY", "C_PAIRED")
COMMON_FEATURE_NAMES = tuple(frozen.FEATURE_NAMES[:2])
NATIVE_FEATURE_NAMES = tuple(frozen.FEATURE_NAMES)
EXPECTED_SHA256 = {
    PAIR_V1: "77367e63ac835bb38340c1ae9c9d16018fde1c22b16a14a12f346d56201e404a",
    PAIR_V2: "fd02c73f508f3f16acbafefb5b703dc864a7793ff7f688ff70efffdffa80f108",
    PAIR_V1_PRODUCER: "8d25f5840b5da6e7ae9a26b7bdba4d46dbfa3d3b5059df0d040dc32da0714bb4",
    PAIR_V2_PRODUCER: "e87240aaeae6308e70f6a6588496d43bdde69d320114d86b3a6de39c3c43aa43",
    SOURCE_MANIFEST: "e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23",
    GEOMETRY: "fff5b980ffa88997ed9bb2a509686800a3448d18a93a714c4ef89007fb24329b",
    FOLDS: "44df51fe97a8db6de1a1101d0b310320da398d1eb88dd21b282c33cbcc6c6bc8",
    CURRENT_VALIDATION: "6c031e5a82b8057a48b48794d53dd76f4888b76ed03f081a925425c64e9621cb",
    FROZEN_VALIDATION: "3dc6fee548ae7171ae6fe0c018cdfe4d31d2e7efecd3790fe0c43e6235f5c372",
    SOURCE_MODULE: "d5d0ceb14d3f2761ebf71ffb0cbd5d960e2492c634ea81509dbdeada9527b525",
    BASE_MODULE: "58787d13530d6694217244f1728d1452cf83b4b387e50e901d5ef2ebc0c698bf",
    FROZEN_MODULE: "96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7",
    CORE_MODULE: "fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c",
    ROMA_WEIGHTS: "1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7",
}
COHORTS = (
    {
        "name": "BALANCED32_V1",
        "path": PAIR_V1,
        "schema": "rc_romav2_colnomic_visibility_xf_balanced32_v1_20260831",
        "status": "ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_NO_GO",
        "inner_fold": 3,
        "control_kind": "ROLL_ONE",
    },
    {
        "name": "BALANCED32_V2",
        "path": PAIR_V2,
        "schema": "rc_romav2_colnomic_visibility_xf_balanced32_v2_20260831",
        "status": "ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_GO",
        "inner_fold": 4,
        "control_kind": "HALF_MAP_ROLL",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: dict) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_direct_hashes() -> None:
    for path, expected in EXPECTED_SHA256.items():
        require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected,
            f"pair64 direct input hash drift: {path}",
        )


def current64_membership() -> tuple[set[int], list[dict[str, object]]]:
    validation = json.loads(CURRENT_VALIDATION.read_text())
    require(
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("logical_sha256") == logical_sha256(validation)
        and validation.get("query_count") == 64
        and validation.get("target_label_read_count") == 0
        and all(validation.get("checks", {}).values()),
        "current64 target-free validation drift",
    )
    executions: set[int] = set()
    seals: list[dict[str, object]] = []
    for item in validation["shards"]:
        shard = int(item["shard"])
        payload_path = CURRENT_ROOT / f"shard{shard:02d}/payload.pt"
        receipt_path = CURRENT_ROOT / f"shard{shard:02d}/receipt.json"
        validation_path = CURRENT_ROOT / f"shard{shard:02d}/validation.json"
        require(
            item.get("payload_sha256") == sha256_file(payload_path)
            and item.get("receipt_sha256") == sha256_file(receipt_path)
            and item.get("validation_sha256") == sha256_file(validation_path),
            "current64 shard seal drift",
        )
        payload = torch.load(
            payload_path, map_location="cpu", weights_only=False, mmap=True
        )
        shard_executions = [
            int(record["execution_ordinal"]) for record in payload.get("records", [])
        ]
        require(
            len(shard_executions) == 8
            and len(set(shard_executions)) == 8
            and not (executions & set(shard_executions)),
            "current64 execution membership drift",
        )
        executions.update(shard_executions)
        seals.append(
            {
                "shard": shard,
                "payload_sha256": item["payload_sha256"],
                "receipt_sha256": item["receipt_sha256"],
                "validation_sha256": item["validation_sha256"],
            }
        )
    require(len(executions) == 64, "current64 population drift")
    return executions, seals


def load_pair_authorities() -> tuple[list[tuple[dict, dict]], set[int], list[dict]]:
    validate_direct_hashes()
    current, current_seals = current64_membership()
    loaded: list[tuple[dict, dict]] = []
    pair_executions: set[int] = set()
    for cohort in COHORTS:
        document = json.loads(Path(cohort["path"]).read_text())
        require(
            document.get("schema_version") == cohort["schema"]
            and document.get("status") == cohort["status"]
            and document.get("logical_sha256") == logical_sha256(document)
            and len(document.get("rows", [])) == 32
            and len(document.get("selection", [])) == 32
            and sum(bool(row.get("base_correct")) for row in document.get("rows", []))
            == 16
            and document.get("model_update_count") == 0
            and document.get("training_count") == 0
            and document.get("full_c128_scoring_count") == 0,
            f"{cohort['name']} authority envelope drift",
        )
        row_executions = [int(row["execution_ordinal"]) for row in document["rows"]]
        selection_executions = [
            int(row["execution_ordinal"]) for row in document["selection"]
        ]
        require(
            len(set(row_executions)) == 32
            and row_executions == selection_executions
            and not (pair_executions & set(row_executions)),
            "balanced pair cohorts overlap or selection order drift",
        )
        pair_executions.update(row_executions)
        loaded.append((cohort, document))
    require(
        len(pair_executions) == 64
        and not (pair_executions & current),
        "pair64 is not disjoint from validated current64",
    )
    frozen_validation = json.loads(FROZEN_VALIDATION.read_text())
    require(
        frozen_validation.get("status")
        == "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS"
        and frozen_validation.get("feature_max_abs") == 0.0
        and frozen_validation.get("module_sha256") == EXPECTED_SHA256[FROZEN_MODULE]
        and tuple(frozen_validation.get("feature_names", ())) == NATIVE_FEATURE_NAMES,
        "frozen candidate-feature authority drift",
    )
    return loaded, current, current_seals


def control_shifts(
    cohort_name: str, query_map: torch.Tensor, reference_map: torch.Tensor
) -> tuple[int, int]:
    if cohort_name == "BALANCED32_V1":
        return 1, 1
    if cohort_name == "BALANCED32_V2":
        return max(1, query_map.numel() // 2), max(1, reference_map.numel() // 2)
    raise RuntimeError("unknown pair cohort")


def evidence_and_derived(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    query_shift: int,
    reference_shift: int,
) -> tuple[dict[str, float], dict[str, float]]:
    real, mass, _ = core.score(
        query_tokens, reference_tokens, query_weight, reference_weight
    )
    query_control, _, _ = core.score(
        query_tokens,
        reference_tokens,
        query_weight.roll(query_shift),
        reference_weight,
    )
    reference_control, _, _ = core.score(
        query_tokens,
        reference_tokens,
        query_weight,
        reference_weight.roll(reference_shift),
    )
    evidence = {
        "real_score": float(real),
        "visibility_mass": float(mass),
        "query_control_score": float(query_control),
        "reference_control_score": float(reference_control),
    }
    derived = {
        "score_over_mass": evidence["real_score"]
        / max(evidence["visibility_mass"], 1e-12),
        "query_response": evidence["real_score"]
        - evidence["query_control_score"],
        "reference_response": evidence["real_score"]
        - evidence["reference_control_score"],
    }
    require(
        all(math.isfinite(value) for value in (*evidence.values(), *derived.values()))
        and evidence["visibility_mass"] >= 0.0,
        "non-finite pair64 arm evidence",
    )
    return evidence, derived


def make_feature(
    raw_scores: list[float],
    evidence: dict[int, dict[str, float]],
    challenger: int,
    winner: int,
) -> torch.Tensor:
    mean = sum(map(float, raw_scores)) / len(raw_scores)
    std = (
        sum((float(value) - mean) ** 2 for value in raw_scores) / len(raw_scores)
    ) ** 0.5
    challenger_evidence = evidence[challenger]
    winner_evidence = evidence[winner]

    def symmetric(left: float, right: float) -> float:
        return (float(left) - float(right)) / (
            abs(float(left)) + abs(float(right)) + 1e-12
        )

    challenger_normalized = float(challenger_evidence["real_score"]) / max(
        float(challenger_evidence["visibility_mass"]), 1e-12
    )
    winner_normalized = float(winner_evidence["real_score"]) / max(
        float(winner_evidence["visibility_mass"]), 1e-12
    )
    challenger_query = float(challenger_evidence["real_score"]) - float(
        challenger_evidence["query_control_score"]
    )
    winner_query = float(winner_evidence["real_score"]) - float(
        winner_evidence["query_control_score"]
    )
    challenger_reference = float(challenger_evidence["real_score"]) - float(
        challenger_evidence["reference_control_score"]
    )
    winner_reference = float(winner_evidence["real_score"]) - float(
        winner_evidence["reference_control_score"]
    )
    return torch.tensor(
        [
            (float(raw_scores[challenger]) - float(raw_scores[winner]))
            / max(std, 1e-12),
            symmetric(
                challenger_evidence["real_score"], winner_evidence["real_score"]
            ),
            symmetric(
                challenger_evidence["visibility_mass"],
                winner_evidence["visibility_mass"],
            ),
            symmetric(challenger_normalized, winner_normalized),
            symmetric(challenger_query, winner_query),
            symmetric(challenger_reference, winner_reference),
        ],
        dtype=torch.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate immutable authorities and population without running RoMaV2",
    )
    args = parser.parse_args()
    if OUT_ROOT.exists() and not args.preflight_only:
        raise RuntimeError(f"immutable pair64 output exists: {OUT_ROOT}")

    pair_authorities, current_executions, current_seals = load_pair_authorities()
    base = FrozenColNomicBaseV1(ROOT)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "ROUTEA_MATCHED_THREE_ARM_PAIR64_INPUT_PREFLIGHT_PASS",
                    "pair_query_count": 64,
                    "current_query_count": len(current_executions),
                    "overlap_count": 0,
                },
                sort_keys=True,
            )
        )
        return

    loader = RGHFull600SourceLoaderV1(
        rc_root=ROOT,
        geometry_payload_path=GEOMETRY,
        prejoin_schedule_path=FOLDS,
    )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(17)
    model = RoMaV2()
    records: list[dict] = []
    c_candidate_replay_count = 0
    c_feature_replay_count = 0
    c_feature_max_abs = 0.0

    for cohort, pair_document in pair_authorities:
        cohort_name = str(cohort["name"])
        for pair_row_ordinal, old_row in enumerate(pair_document["rows"]):
            execution = int(old_row["execution_ordinal"])
            source = loader.load_execution(execution)
            axis = [int(item.physical_row) for item in source.candidates]
            raw_tensor = base.scores(execution, axis)
            raw_scores = raw_tensor.tolist()
            old_candidates = {
                int(item["candidate_position"]): item
                for item in old_row["candidates"]
            }
            target_candidates = [
                item for item in old_row["candidates"] if item.get("role") == "TARGET"
            ]
            competitor_candidates = [
                item
                for item in old_row["candidates"]
                if item.get("role") == "COMPETITOR"
            ]
            require(
                source.query_id == old_row["query_id"]
                and source.inner_fold == cohort["inner_fold"]
                and len(target_candidates) == len(competitor_candidates) == 1
                and len(old_candidates) == 2,
                "fixed pair source/role drift",
            )
            target_position = int(target_candidates[0]["candidate_position"])
            competitor_position = int(
                competitor_candidates[0]["candidate_position"]
            )
            ranking = sorted(
                range(128), key=lambda position: (-raw_scores[position], axis[position])
            )
            target_rank = ranking.index(target_position) + 1
            expected_competitor = sorted(
                (position for position in range(128) if position != target_position),
                key=lambda position: (-raw_scores[position], axis[position]),
            )[0]
            require(
                target_position != competitor_position
                and competitor_position == expected_competitor
                and int(old_row["base_rank"]) == target_rank
                and bool(old_row["base_correct"]) == (target_rank == 1)
                and axis[target_position] == int(target_candidates[0]["physical_row"])
                and axis[competitor_position]
                == int(competitor_candidates[0]["physical_row"]),
                "fixed TARGET/COMPETITOR row drift",
            )
            if target_rank == 1:
                winner, challenger, switch_label = (
                    target_position,
                    competitor_position,
                    False,
                )
            else:
                winner, challenger, switch_label = (
                    competitor_position,
                    target_position,
                    True,
                )

            query_image = core.oriented(source.query.source_path)
            evidence: dict[str, dict[int, dict[str, float]]] = {
                arm: {} for arm in ARMS
            }
            derived_evidence: dict[str, dict[int, dict[str, float]]] = {
                arm: {} for arm in ARMS
            }
            fixed_candidate_maps: dict[int, dict] = {}
            for position in (winner, challenger):
                reference = source.candidates[position].source
                prediction = model.match(
                    query_image, core.oriented(reference.source_path)
                )
                query_map = core.cell_means(
                    prediction["overlap_AB"][0, ..., 0].detach().cpu(),
                    source.query.colnomic_geometry,
                ).contiguous()
                reference_map = core.cell_means(
                    prediction["overlap_BA"][0, ..., 0].detach().cpu(),
                    reference.colnomic_geometry,
                ).contiguous()
                query_shift, reference_shift = control_shifts(
                    cohort_name, query_map, reference_map
                )
                ones_query = torch.ones_like(query_map)
                ones_reference = torch.ones_like(reference_map)
                effective = {
                    "A_ALL": (ones_query, ones_reference),
                    "B_QUERY": (query_map, ones_reference),
                    "C_PAIRED": (query_map, reference_map),
                }
                for arm in ARMS:
                    arm_evidence, arm_derived = evidence_and_derived(
                        source.query.tokens,
                        reference.tokens,
                        effective[arm][0],
                        effective[arm][1],
                        query_shift,
                        reference_shift,
                    )
                    evidence[arm][position] = arm_evidence
                    derived_evidence[arm][position] = arm_derived

                old = old_candidates[position]
                c_evidence = evidence["C_PAIRED"][position]
                query_hash = tensor_sha256(query_map)
                reference_hash = tensor_sha256(reference_map)
                expected_query_shift = int(old.get("query_control_shift", 1))
                expected_reference_shift = int(old.get("reference_control_shift", 1))
                require(
                    query_hash == old["query_map_sha256"]
                    and reference_hash == old["reference_map_sha256"]
                    and query_shift == expected_query_shift
                    and reference_shift == expected_reference_shift
                    and c_evidence["real_score"] == float(old["real_score"])
                    and c_evidence["visibility_mass"]
                    == float(old["visibility_mass"])
                    and c_evidence["query_control_score"]
                    == float(old["query_control_score"])
                    and c_evidence["reference_control_score"]
                    == float(old["reference_control_score"]),
                    "C candidate map/evidence is not an exact historical replay",
                )
                c_candidate_replay_count += 1
                fixed_candidate_maps[position] = {
                    "candidate_position": position,
                    "physical_row": axis[position],
                    "query_map": query_map,
                    "reference_map": reference_map,
                    "query_map_sha256": query_hash,
                    "reference_map_sha256": reference_hash,
                    "query_control_shift": query_shift,
                    "reference_control_shift": reference_shift,
                    "query_tokens_sha256": source.query.tokens_sha256,
                    "reference_tokens_sha256": reference.tokens_sha256,
                    "query_source_image_sha256": source.query.source_image_sha256,
                    "reference_source_image_sha256": reference.source_image_sha256,
                    "query_source_logical_sha256": source.query.source_logical_sha256,
                    "reference_source_logical_sha256": reference.source_logical_sha256,
                }

            real_native_features: dict[str, torch.Tensor] = {}
            real_common_features: dict[str, torch.Tensor] = {}
            for arm in ARMS:
                native = make_feature(
                    raw_scores, evidence[arm], challenger, winner
                ).unsqueeze(0)
                real_native_features[arm] = native
                real_common_features[arm] = native[:, :2].contiguous()
            expected_frozen_feature = frozen.candidate_feature(
                raw_scores, old_candidates, challenger, winner
            ).unsqueeze(0)
            observed_c_feature = real_native_features["C_PAIRED"]
            feature_abs = float(
                (observed_c_feature - expected_frozen_feature).abs().max()
            )
            c_feature_max_abs = max(c_feature_max_abs, feature_abs)
            require(
                torch.equal(observed_c_feature, expected_frozen_feature),
                "C native feature is not bit-exact to frozen candidate_feature",
            )
            c_feature_replay_count += 1
            records.append(
                {
                    "pair_cohort": cohort_name,
                    "pair_row_ordinal": pair_row_ordinal,
                    "execution_ordinal": execution,
                    "query_id": source.query_id,
                    "inner_fold": int(source.inner_fold),
                    "candidate_physical_rows": axis,
                    "base_scores": raw_tensor,
                    "base_winner_position": winner,
                    "challenger_positions": [challenger],
                    "switch_label": switch_label,
                    "evidence": evidence,
                    "derived_evidence": derived_evidence,
                    "fixed_candidate_maps": fixed_candidate_maps,
                    "real_common_features": real_common_features,
                    "real_native_features": real_native_features,
                    "c_frozen_candidate_feature": expected_frozen_feature,
                    "c_feature_exact": True,
                    "pair_role_output_count": 0,
                    "target_identity_output_count": 0,
                    "eval_candidate_generation_count": 0,
                    "model_update_count": 0,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "pair64_training_feature_ready",
                        "cohort": cohort_name,
                        "pair_row_ordinal": pair_row_ordinal,
                        "execution_ordinal": execution,
                        "done": len(records),
                        "total": 64,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    require(
        len(records) == 64
        and len({record["execution_ordinal"] for record in records}) == 64
        and sum(bool(record["switch_label"]) for record in records) == 32
        and c_candidate_replay_count == 128
        and c_feature_replay_count == 64
        and c_feature_max_abs == 0.0,
        "pair64 final population or C replay drift",
    )
    value = {
        "schema_version": "routea_matched_three_arm_pair64_training_features_v1_20260902",
        "status": "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY",
        "claim_level": "TRAINING_ONLY_PAIR64_FEATURES_NO_EVAL_CANDIDATE_GENERATION",
        "arms": list(ARMS),
        "feature_names": {
            "common3_inputs": list(COMMON_FEATURE_NAMES),
            "native7_inputs": list(NATIVE_FEATURE_NAMES),
        },
        "control_conventions": {
            "BALANCED32_V1": {
                "query": "ROLL_ONE",
                "reference": "ROLL_ONE",
            },
            "BALANCED32_V2": {
                "query": "HALF_MAP_ROLL",
                "reference": "HALF_MAP_ROLL",
            },
            "same_shift_for_all_arms_within_cohort": True,
        },
        "records": records,
        "population": {
            "pair_query_count": 64,
            "fixed_candidate_count": 128,
            "switch_label_count": 64,
            "positive_switch_label_count": 32,
            "negative_switch_label_count": 32,
            "current64_membership_count": len(current_executions),
            "pair64_current64_overlap_count": 0,
        },
        "c_exact_replay": {
            "candidate_count": c_candidate_replay_count,
            "feature_count": c_feature_replay_count,
            "feature_max_abs": c_feature_max_abs,
            "map_hash_mismatch_count": 0,
            "scalar_mismatch_count": 0,
        },
        "bindings": {
            "contract_sha256": sha256_file(CONTRACT),
            "primary_contract_sha256": sha256_file(PRIMARY_CONTRACT),
            "producer_sha256": sha256_file(Path(__file__).resolve()),
            "pair_v1_sha256": sha256_file(PAIR_V1),
            "pair_v2_sha256": sha256_file(PAIR_V2),
            "pair_v1_producer_sha256": sha256_file(PAIR_V1_PRODUCER),
            "pair_v2_producer_sha256": sha256_file(PAIR_V2_PRODUCER),
            "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
            "geometry_payload_sha256": sha256_file(GEOMETRY),
            "fold_schedule_sha256": sha256_file(FOLDS),
            "current64_validation_sha256": sha256_file(CURRENT_VALIDATION),
            "current64_shards": current_seals,
            "current64_execution_set_sha256": canonical_sha256(
                sorted(current_executions)
            ),
            "source_module_sha256": sha256_file(SOURCE_MODULE),
            "base_module_sha256": sha256_file(BASE_MODULE),
            "core_module_sha256": sha256_file(CORE_MODULE),
            "frozen_gate_module_sha256": sha256_file(FROZEN_MODULE),
            "frozen_gate_validation_sha256": sha256_file(FROZEN_VALIDATION),
            "roma_checkpoint_sha256": sha256_file(ROMA_WEIGHTS),
        },
        "usage_policy": {
            "training_only": True,
            "allowed_consumers": [
                "MATCHED_THREE_ARM_COMMON3_CROSSFIT_TRAINING",
                "MATCHED_THREE_ARM_NATIVE7_CROSSFIT_TRAINING",
                "FROZEN_C_FEATURE_REGRESSION",
            ],
            "eval_candidate_generation_authorized": False,
            "current_axis_mutation_authorized": False,
            "scientific_result_authorized": False,
        },
        "access": {
            "target_bearing_authority_read_count": 2,
            "pair_query_read_count": 64,
            "pair_role_field_read_count": 128,
            "target_candidate_role_read_count": 64,
            "competitor_candidate_role_read_count": 64,
            "target_identity_read_count": 0,
            "supergroup_read_count": 0,
            "current_membership_read_count": 64,
            "current_candidate_value_consumption_count": 0,
            "switch_label_output_count": 64,
            "pair_role_output_count": 0,
            "eval_candidate_generation_count": 0,
            "model_update_count": 0,
        },
    }

    OUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".pair64-training-features-", dir=OUT_ROOT.parent)
    )
    stage_payload = staging / "payload.pt"
    stage_receipt = staging / "receipt.json"
    torch.save(value, stage_payload)
    receipt = {
        "schema_version": "routea_matched_three_arm_pair64_training_features_receipt_v1_20260902",
        "status": value["status"],
        "claim_level": value["claim_level"],
        "query_count": 64,
        "fixed_candidate_count": 128,
        "switch_label_count": 64,
        "c_feature_max_abs": c_feature_max_abs,
        "payload_sha256": sha256_file(stage_payload),
        "eval_candidate_generation_authorized": False,
        "model_update_count": 0,
        "next_authorized_stage": "PAIR64_TRAINING_FEATURE_INDEPENDENT_VALIDATION",
        "logical_sha256": "",
    }
    receipt["logical_sha256"] = logical_sha256(receipt)
    try:
        stage_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.rename(staging, OUT_ROOT)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
