#!/usr/bin/env python3
"""Independent validation of matched A/B/C pair64 training-only features."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rc_aslo_xf import romav2_colnomic_frozen_gate_v1 as frozen  # noqa: E402
from rc_aslo_xf.rgh_full600_source_v1 import RGHFull600SourceLoaderV1  # noqa: E402
from rc_aslo_xf.rgh_v9_frozen_colnomic_base_v1 import (  # noqa: E402
    FrozenColNomicBaseV1,
)


CONTRACT = (
    ROOT
    / "plan/ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_ADDENDUM_V1_20260902.md"
)
PRIMARY_CONTRACT = (
    ROOT
    / "plan/ROUTEA_MATCHED_THREE_ARM_SAME_CAPACITY_7PARAM_CROSSFIT_V1_20260902.md"
)
PRODUCER = ROOT / "programs/materialize_routea_matched_three_arm_pair64_training_features_v1.py"
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
PAYLOAD_PATH = OUT_ROOT / "payload.pt"
RECEIPT_PATH = OUT_ROOT / "receipt.json"
OUT = OUT_ROOT / "independent_validation.json"

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
    },
    {
        "name": "BALANCED32_V2",
        "path": PAIR_V2,
        "schema": "rc_romav2_colnomic_visibility_xf_balanced32_v2_20260831",
        "status": "ROMAV2_COLNOMIC_VISIBILITY_XF_BALANCED32_V2_GO",
        "inner_fold": 4,
    },
)
PAYLOAD_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "arms",
    "feature_names",
    "control_conventions",
    "records",
    "population",
    "c_exact_replay",
    "bindings",
    "usage_policy",
    "access",
}
RECORD_KEYS = {
    "pair_cohort",
    "pair_row_ordinal",
    "execution_ordinal",
    "query_id",
    "inner_fold",
    "candidate_physical_rows",
    "base_scores",
    "base_winner_position",
    "challenger_positions",
    "switch_label",
    "evidence",
    "derived_evidence",
    "fixed_candidate_maps",
    "real_common_features",
    "real_native_features",
    "c_frozen_candidate_feature",
    "c_feature_exact",
    "pair_role_output_count",
    "target_identity_output_count",
    "eval_candidate_generation_count",
    "model_update_count",
}
MAP_KEYS = {
    "candidate_position",
    "physical_row",
    "query_map",
    "reference_map",
    "query_map_sha256",
    "reference_map_sha256",
    "query_control_shift",
    "reference_control_shift",
    "query_tokens_sha256",
    "reference_tokens_sha256",
    "query_source_image_sha256",
    "reference_source_image_sha256",
    "query_source_logical_sha256",
    "reference_source_logical_sha256",
}
EVIDENCE_KEYS = {
    "real_score",
    "visibility_mass",
    "query_control_score",
    "reference_control_score",
}
DERIVED_KEYS = {"score_over_mass", "query_response", "reference_response"}
RECEIPT_KEYS = {
    "schema_version",
    "status",
    "claim_level",
    "query_count",
    "fixed_candidate_count",
    "switch_label_count",
    "c_feature_max_abs",
    "payload_sha256",
    "eval_candidate_generation_authorized",
    "model_update_count",
    "next_authorized_stage",
    "logical_sha256",
}


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


def atomic_json(path: Path, value: dict) -> None:
    file_descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(file_descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def independent_score(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_weight: torch.Tensor,
    reference_weight: torch.Tensor,
) -> tuple[float, float]:
    query_unit = F.normalize(query_tokens.to(torch.float64), dim=1)
    reference_unit = F.normalize(reference_tokens.to(torch.float64), dim=1)
    similarity = query_unit @ reference_unit.T
    local = (similarity * reference_weight[None]).max(dim=1).values
    mass = torch.sqrt(query_weight.mean() * reference_weight.mean())
    score = mass * (query_weight * local).sum() / query_weight.sum().clamp_min(1e-12)
    return float(score), float(mass)


def independent_evidence(
    query_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    query_weight: torch.Tensor,
    reference_weight: torch.Tensor,
    query_shift: int,
    reference_shift: int,
) -> tuple[dict[str, float], dict[str, float]]:
    real, mass = independent_score(
        query_tokens, reference_tokens, query_weight, reference_weight
    )
    query_control, _ = independent_score(
        query_tokens,
        reference_tokens,
        query_weight.roll(query_shift),
        reference_weight,
    )
    reference_control, _ = independent_score(
        query_tokens,
        reference_tokens,
        query_weight,
        reference_weight.roll(reference_shift),
    )
    evidence = {
        "real_score": real,
        "visibility_mass": mass,
        "query_control_score": query_control,
        "reference_control_score": reference_control,
    }
    derived = {
        "score_over_mass": real / max(mass, 1e-12),
        "query_response": real - query_control,
        "reference_response": real - reference_control,
    }
    return evidence, derived


def independent_feature(
    raw_scores: list[float],
    evidence: dict[int, dict[str, float]],
    challenger: int,
    winner: int,
) -> torch.Tensor:
    mean = sum(map(float, raw_scores)) / len(raw_scores)
    standard_deviation = (
        sum((float(value) - mean) ** 2 for value in raw_scores) / len(raw_scores)
    ) ** 0.5
    left = evidence[challenger]
    right = evidence[winner]

    def symmetric(a: float, b: float) -> float:
        return (float(a) - float(b)) / (abs(float(a)) + abs(float(b)) + 1e-12)

    left_normalized = float(left["real_score"]) / max(
        float(left["visibility_mass"]), 1e-12
    )
    right_normalized = float(right["real_score"]) / max(
        float(right["visibility_mass"]), 1e-12
    )
    left_query = float(left["real_score"]) - float(left["query_control_score"])
    right_query = float(right["real_score"]) - float(right["query_control_score"])
    left_reference = float(left["real_score"]) - float(
        left["reference_control_score"]
    )
    right_reference = float(right["real_score"]) - float(
        right["reference_control_score"]
    )
    return torch.tensor(
        [
            (float(raw_scores[challenger]) - float(raw_scores[winner]))
            / max(standard_deviation, 1e-12),
            symmetric(left["real_score"], right["real_score"]),
            symmetric(left["visibility_mass"], right["visibility_mass"]),
            symmetric(left_normalized, right_normalized),
            symmetric(left_query, right_query),
            symmetric(left_reference, right_reference),
        ],
        dtype=torch.float64,
    )


def load_current_membership() -> tuple[set[int], list[dict[str, object]]]:
    validation = json.loads(CURRENT_VALIDATION.read_text())
    if not (
        validation.get("status") == "ROUTEA_D1_CURRENT_RUNTIME_64_PREJOIN_VALIDATED"
        and validation.get("logical_sha256") == logical_sha256(validation)
        and validation.get("query_count") == 64
        and validation.get("target_label_read_count") == 0
        and all(validation.get("checks", {}).values())
    ):
        raise RuntimeError("current64 validation drift")
    executions: set[int] = set()
    seals: list[dict[str, object]] = []
    for seal in validation["shards"]:
        shard = int(seal["shard"])
        payload_path = CURRENT_ROOT / f"shard{shard:02d}/payload.pt"
        receipt_path = CURRENT_ROOT / f"shard{shard:02d}/receipt.json"
        validation_path = CURRENT_ROOT / f"shard{shard:02d}/validation.json"
        if not (
            seal.get("payload_sha256") == sha256_file(payload_path)
            and seal.get("receipt_sha256") == sha256_file(receipt_path)
            and seal.get("validation_sha256") == sha256_file(validation_path)
        ):
            raise RuntimeError("current64 shard seal drift")
        current_payload = torch.load(
            payload_path, map_location="cpu", weights_only=False, mmap=True
        )
        shard_executions = {
            int(record["execution_ordinal"])
            for record in current_payload.get("records", [])
        }
        if len(shard_executions) != 8 or executions & shard_executions:
            raise RuntimeError("current64 execution-set drift")
        executions.update(shard_executions)
        seals.append(
            {
                "shard": shard,
                "payload_sha256": seal["payload_sha256"],
                "receipt_sha256": seal["receipt_sha256"],
                "validation_sha256": seal["validation_sha256"],
            }
        )
    if len(executions) != 64:
        raise RuntimeError("current64 population drift")
    return executions, seals


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable pair64 independent validation exists: {OUT}")
    for path, expected in EXPECTED_SHA256.items():
        if not (
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected
        ):
            raise RuntimeError(f"pair64 validator input hash drift: {path}")

    payload = torch.load(
        PAYLOAD_PATH, map_location="cpu", weights_only=False, mmap=True
    )
    receipt = json.loads(RECEIPT_PATH.read_text())
    current_executions, current_seals = load_current_membership()
    pair_documents: dict[str, dict] = {}
    old_rows: dict[tuple[str, int], dict] = {}
    pair_executions: set[int] = set()
    pair_authority_checks: list[bool] = []
    for cohort in COHORTS:
        document = json.loads(Path(cohort["path"]).read_text())
        row_executions = [int(row["execution_ordinal"]) for row in document["rows"]]
        selection_executions = [
            int(row["execution_ordinal"]) for row in document["selection"]
        ]
        pair_authority_checks.append(
            document.get("schema_version") == cohort["schema"]
            and document.get("status") == cohort["status"]
            and document.get("logical_sha256") == logical_sha256(document)
            and len(document.get("rows", [])) == 32
            and len(set(row_executions)) == 32
            and sum(bool(row.get("base_correct")) for row in document.get("rows", []))
            == 16
            and row_executions == selection_executions
            and not (pair_executions & set(row_executions))
            and document.get("model_update_count") == 0
            and document.get("training_count") == 0
            and document.get("full_c128_scoring_count") == 0
        )
        pair_executions.update(row_executions)
        pair_documents[str(cohort["name"])] = document
        old_rows.update(
            {
                (str(cohort["name"]), ordinal): row
                for ordinal, row in enumerate(document["rows"])
            }
        )

    loader = RGHFull600SourceLoaderV1(
        rc_root=ROOT,
        geometry_payload_path=GEOMETRY,
        prejoin_schedule_path=FOLDS,
    )
    base = FrozenColNomicBaseV1(ROOT)
    records = payload.get("records", [])
    record_checks: list[bool] = []
    formula_max_abs = 0.0
    c_feature_max_abs = 0.0
    exact_c_candidate_count = 0
    exact_c_feature_count = 0
    switch_label_count = 0
    fixed_candidate_count = 0
    for record in records:
        if set(record) != RECORD_KEYS:
            raise RuntimeError("pair64 record schema drift")
        cohort_name = str(record["pair_cohort"])
        pair_row_ordinal = int(record["pair_row_ordinal"])
        old_row = old_rows[(cohort_name, pair_row_ordinal)]
        cohort = next(item for item in COHORTS if item["name"] == cohort_name)
        execution = int(record["execution_ordinal"])
        source = loader.load_execution(execution)
        axis = [int(item.physical_row) for item in source.candidates]
        raw_tensor = base.scores(execution, axis)
        raw_scores = raw_tensor.tolist()
        old_candidates = {
            int(item["candidate_position"]): item for item in old_row["candidates"]
        }
        target = next(
            item for item in old_row["candidates"] if item.get("role") == "TARGET"
        )
        competitor = next(
            item
            for item in old_row["candidates"]
            if item.get("role") == "COMPETITOR"
        )
        target_position = int(target["candidate_position"])
        competitor_position = int(competitor["candidate_position"])
        ranking = sorted(
            range(128), key=lambda position: (-raw_scores[position], axis[position])
        )
        target_rank = ranking.index(target_position) + 1
        expected_competitor = sorted(
            (position for position in range(128) if position != target_position),
            key=lambda position: (-raw_scores[position], axis[position]),
        )[0]
        if target_rank == 1:
            winner, challenger, expected_label = (
                target_position,
                competitor_position,
                False,
            )
        else:
            winner, challenger, expected_label = (
                competitor_position,
                target_position,
                True,
            )
        switch_label_count += int(bool(record["switch_label"]))
        stored_maps = record["fixed_candidate_maps"]
        if set(map(int, stored_maps)) != {winner, challenger}:
            raise RuntimeError("pair64 fixed map positions drift")
        recomputed_evidence: dict[str, dict[int, dict[str, float]]] = {
            arm: {} for arm in ARMS
        }
        recomputed_derived: dict[str, dict[int, dict[str, float]]] = {
            arm: {} for arm in ARMS
        }
        map_checks: list[bool] = []
        for position in (winner, challenger):
            fixed_candidate_count += 1
            entry = stored_maps[position]
            if set(entry) != MAP_KEYS:
                raise RuntimeError("pair64 fixed map schema drift")
            reference = source.candidates[position].source
            query_map = torch.as_tensor(entry["query_map"])
            reference_map = torch.as_tensor(entry["reference_map"])
            if cohort_name == "BALANCED32_V1":
                expected_query_shift = expected_reference_shift = 1
            else:
                expected_query_shift = max(1, query_map.numel() // 2)
                expected_reference_shift = max(1, reference_map.numel() // 2)
            ones_query = torch.ones_like(query_map)
            ones_reference = torch.ones_like(reference_map)
            effective = {
                "A_ALL": (ones_query, ones_reference),
                "B_QUERY": (query_map, ones_reference),
                "C_PAIRED": (query_map, reference_map),
            }
            for arm in ARMS:
                computed, derived = independent_evidence(
                    source.query.tokens,
                    reference.tokens,
                    effective[arm][0],
                    effective[arm][1],
                    expected_query_shift,
                    expected_reference_shift,
                )
                recomputed_evidence[arm][position] = computed
                recomputed_derived[arm][position] = derived
                stored_evidence = record["evidence"][arm][position]
                stored_derived = record["derived_evidence"][arm][position]
                if set(stored_evidence) != EVIDENCE_KEYS or set(stored_derived) != DERIVED_KEYS:
                    raise RuntimeError("pair64 evidence schema drift")
                formula_max_abs = max(
                    formula_max_abs,
                    *(abs(computed[key] - float(stored_evidence[key])) for key in EVIDENCE_KEYS),
                    *(abs(derived[key] - float(stored_derived[key])) for key in DERIVED_KEYS),
                )
            old = old_candidates[position]
            stored_c = record["evidence"]["C_PAIRED"][position]
            c_exact = (
                entry["query_map_sha256"] == old["query_map_sha256"]
                and entry["reference_map_sha256"] == old["reference_map_sha256"]
                and stored_c["real_score"] == float(old["real_score"])
                and stored_c["visibility_mass"] == float(old["visibility_mass"])
                and stored_c["query_control_score"]
                == float(old["query_control_score"])
                and stored_c["reference_control_score"]
                == float(old["reference_control_score"])
            )
            exact_c_candidate_count += int(c_exact)
            map_checks.append(
                entry["candidate_position"] == position
                and entry["physical_row"] == axis[position]
                and query_map.dtype == reference_map.dtype == torch.float64
                and query_map.ndim == reference_map.ndim == 1
                and query_map.shape == (source.query.tokens.shape[0],)
                and reference_map.shape == (reference.tokens.shape[0],)
                and bool(torch.isfinite(query_map).all())
                and bool(torch.isfinite(reference_map).all())
                and float(query_map.min()) >= 0.0
                and float(query_map.max()) <= 1.0
                and float(reference_map.min()) >= 0.0
                and float(reference_map.max()) <= 1.0
                and entry["query_map_sha256"] == tensor_sha256(query_map)
                and entry["reference_map_sha256"] == tensor_sha256(reference_map)
                and entry["query_control_shift"] == expected_query_shift
                and entry["reference_control_shift"] == expected_reference_shift
                and entry["query_tokens_sha256"] == source.query.tokens_sha256
                and entry["reference_tokens_sha256"] == reference.tokens_sha256
                and entry["query_source_image_sha256"]
                == source.query.source_image_sha256
                and entry["reference_source_image_sha256"]
                == reference.source_image_sha256
                and entry["query_source_logical_sha256"]
                == source.query.source_logical_sha256
                and entry["reference_source_logical_sha256"]
                == reference.source_logical_sha256
                and c_exact
            )

        feature_checks: list[bool] = []
        for arm in ARMS:
            expected_native = independent_feature(
                raw_scores, recomputed_evidence[arm], challenger, winner
            ).unsqueeze(0)
            stored_native = torch.as_tensor(record["real_native_features"][arm])
            stored_common = torch.as_tensor(record["real_common_features"][arm])
            feature_checks.append(
                stored_native.dtype == stored_common.dtype == torch.float64
                and stored_native.shape == (1, 6)
                and stored_common.shape == (1, 2)
                and torch.equal(stored_native, expected_native)
                and torch.equal(stored_common, expected_native[:, :2])
                and torch.equal(stored_common, stored_native[:, :2])
            )
        expected_frozen = frozen.candidate_feature(
            raw_scores, old_candidates, challenger, winner
        ).unsqueeze(0)
        stored_c_native = torch.as_tensor(record["real_native_features"]["C_PAIRED"])
        stored_c_frozen = torch.as_tensor(record["c_frozen_candidate_feature"])
        c_feature_max_abs = max(
            c_feature_max_abs,
            float((stored_c_native - expected_frozen).abs().max()),
            float((stored_c_frozen - expected_frozen).abs().max()),
        )
        c_feature_exact = (
            torch.equal(stored_c_native, expected_frozen)
            and torch.equal(stored_c_frozen, expected_frozen)
            and record["c_feature_exact"] is True
        )
        exact_c_feature_count += int(c_feature_exact)
        a_native = torch.as_tensor(record["real_native_features"]["A_ALL"])
        b_native = torch.as_tensor(record["real_native_features"]["B_QUERY"])
        role_output_ok = (
            record["pair_role_output_count"] == 0
            and record["target_identity_output_count"] == 0
            and record["eval_candidate_generation_count"] == 0
            and record["model_update_count"] == 0
            and not (
                {"role", "target_position", "base_correct", "target_identity", "supergroup"}
                & set(record)
            )
        )
        record_checks.append(
            cohort_name in pair_documents
            and old_row["execution_ordinal"] == execution
            and old_row["query_id"] == source.query_id == record["query_id"]
            and source.inner_fold == cohort["inner_fold"] == record["inner_fold"]
            and axis == record["candidate_physical_rows"]
            and torch.equal(torch.as_tensor(record["base_scores"]), raw_tensor)
            and competitor_position == expected_competitor
            and int(old_row["base_rank"]) == target_rank
            and bool(old_row["base_correct"]) == (target_rank == 1)
            and record["base_winner_position"] == winner
            and record["challenger_positions"] == [challenger]
            and record["switch_label"] is expected_label
            and set(record["evidence"]) == set(ARMS)
            and set(record["derived_evidence"]) == set(ARMS)
            and all(set(map(int, record["evidence"][arm])) == {winner, challenger} for arm in ARMS)
            and all(set(map(int, record["derived_evidence"][arm])) == {winner, challenger} for arm in ARMS)
            and set(record["real_common_features"]) == set(ARMS)
            and set(record["real_native_features"]) == set(ARMS)
            and all(map_checks)
            and all(feature_checks)
            and c_feature_exact
            and bool(torch.equal(a_native[:, 4:], torch.zeros((1, 2), dtype=torch.float64)))
            and float(b_native[0, 5]) == 0.0
            and role_output_ok
        )

    expected_bindings = {
        "contract_sha256": sha256_file(CONTRACT),
        "primary_contract_sha256": sha256_file(PRIMARY_CONTRACT),
        "producer_sha256": sha256_file(PRODUCER),
        "pair_v1_sha256": sha256_file(PAIR_V1),
        "pair_v2_sha256": sha256_file(PAIR_V2),
        "pair_v1_producer_sha256": sha256_file(PAIR_V1_PRODUCER),
        "pair_v2_producer_sha256": sha256_file(PAIR_V2_PRODUCER),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "geometry_payload_sha256": sha256_file(GEOMETRY),
        "fold_schedule_sha256": sha256_file(FOLDS),
        "current64_validation_sha256": sha256_file(CURRENT_VALIDATION),
        "current64_shards": current_seals,
        "current64_execution_set_sha256": canonical_sha256(sorted(current_executions)),
        "source_module_sha256": sha256_file(SOURCE_MODULE),
        "base_module_sha256": sha256_file(BASE_MODULE),
        "core_module_sha256": sha256_file(CORE_MODULE),
        "frozen_gate_module_sha256": sha256_file(FROZEN_MODULE),
        "frozen_gate_validation_sha256": sha256_file(FROZEN_VALIDATION),
        "roma_checkpoint_sha256": sha256_file(ROMA_WEIGHTS),
    }
    expected_population = {
        "pair_query_count": 64,
        "fixed_candidate_count": 128,
        "switch_label_count": 64,
        "positive_switch_label_count": 32,
        "negative_switch_label_count": 32,
        "current64_membership_count": 64,
        "pair64_current64_overlap_count": 0,
    }
    expected_access = {
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
    }
    expected_usage = {
        "training_only": True,
        "allowed_consumers": [
            "MATCHED_THREE_ARM_COMMON3_CROSSFIT_TRAINING",
            "MATCHED_THREE_ARM_NATIVE7_CROSSFIT_TRAINING",
            "FROZEN_C_FEATURE_REGRESSION",
        ],
        "eval_candidate_generation_authorized": False,
        "current_axis_mutation_authorized": False,
        "scientific_result_authorized": False,
    }
    expected_c_replay = {
        "candidate_count": 128,
        "feature_count": 64,
        "feature_max_abs": 0.0,
        "map_hash_mismatch_count": 0,
        "scalar_mismatch_count": 0,
    }
    frozen_validation = json.loads(FROZEN_VALIDATION.read_text())
    checks = {
        "payload_envelope": set(payload) == PAYLOAD_KEYS
        and payload.get("schema_version")
        == "routea_matched_three_arm_pair64_training_features_v1_20260902"
        and payload.get("status")
        == "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY"
        and payload.get("claim_level")
        == "TRAINING_ONLY_PAIR64_FEATURES_NO_EVAL_CANDIDATE_GENERATION"
        and tuple(payload.get("arms", ())) == ARMS,
        "direct_input_hashes": all(
            sha256_file(path) == expected for path, expected in EXPECTED_SHA256.items()
        ),
        "pair_authorities": all(pair_authority_checks),
        "pair64_population_disjoint": len(pair_executions) == 64
        and len(current_executions) == 64
        and not (pair_executions & current_executions),
        "feature_names": payload.get("feature_names")
        == {
            "common3_inputs": list(COMMON_FEATURE_NAMES),
            "native7_inputs": list(NATIVE_FEATURE_NAMES),
        },
        "control_conventions": payload.get("control_conventions")
        == {
            "BALANCED32_V1": {"query": "ROLL_ONE", "reference": "ROLL_ONE"},
            "BALANCED32_V2": {
                "query": "HALF_MAP_ROLL",
                "reference": "HALF_MAP_ROLL",
            },
            "same_shift_for_all_arms_within_cohort": True,
        },
        "frozen_feature_authority": frozen_validation.get("status")
        == "ROMAV2_COLNOMIC_FROZEN_GATE_DEFINITION_VALIDATION_PASS"
        and frozen_validation.get("feature_max_abs") == 0.0
        and frozen_validation.get("module_sha256") == sha256_file(FROZEN_MODULE)
        and tuple(frozen_validation.get("feature_names", ()))
        == NATIVE_FEATURE_NAMES,
        "record_population": len(records) == 64
        and len({int(record["execution_ordinal"]) for record in records}) == 64
        and sum(record["pair_cohort"] == "BALANCED32_V1" for record in records) == 32
        and sum(record["pair_cohort"] == "BALANCED32_V2" for record in records) == 32,
        "record_formula_maps_features": all(record_checks)
        and formula_max_abs <= 1e-12,
        "c_exact_candidate_replay": exact_c_candidate_count == 128,
        "c_exact_feature_replay": exact_c_feature_count == 64
        and c_feature_max_abs == 0.0,
        "population": payload.get("population") == expected_population
        and fixed_candidate_count == 128
        and len(records) == 64
        and switch_label_count == 32,
        "c_replay_summary": payload.get("c_exact_replay") == expected_c_replay,
        "bindings": payload.get("bindings") == expected_bindings,
        "usage_policy": payload.get("usage_policy") == expected_usage,
        "access": payload.get("access") == expected_access,
        "receipt": set(receipt) == RECEIPT_KEYS
        and receipt.get("schema_version")
        == "routea_matched_three_arm_pair64_training_features_receipt_v1_20260902"
        and receipt.get("status")
        == "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_READY"
        and receipt.get("claim_level")
        == "TRAINING_ONLY_PAIR64_FEATURES_NO_EVAL_CANDIDATE_GENERATION"
        and receipt.get("logical_sha256") == logical_sha256(receipt)
        and receipt.get("query_count") == 64
        and receipt.get("fixed_candidate_count") == 128
        and receipt.get("switch_label_count") == 64
        and receipt.get("c_feature_max_abs") == 0.0
        and receipt.get("payload_sha256") == sha256_file(PAYLOAD_PATH)
        and receipt.get("eval_candidate_generation_authorized") is False
        and receipt.get("model_update_count") == 0
        and receipt.get("next_authorized_stage")
        == "PAIR64_TRAINING_FEATURE_INDEPENDENT_VALIDATION",
        "claim_boundary": payload.get("usage_policy", {}).get(
            "eval_candidate_generation_authorized"
        )
        is False
        and payload.get("access", {}).get("pair_role_output_count") == 0
        and payload.get("access", {}).get("target_identity_read_count") == 0
        and payload.get("access", {}).get("model_update_count") == 0,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "routea_matched_three_arm_pair64_training_features_independent_validation_v1_20260902",
        "status": (
            "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_INDEPENDENT_VALIDATION_PASS"
            if passed
            else "ROUTEA_MATCHED_THREE_ARM_PAIR64_TRAINING_FEATURES_INDEPENDENT_VALIDATION_FAIL"
        ),
        "checks": checks,
        "query_count": len(records),
        "fixed_candidate_count": fixed_candidate_count,
        "positive_switch_label_count": switch_label_count,
        "c_exact_candidate_count": exact_c_candidate_count,
        "c_exact_feature_count": exact_c_feature_count,
        "formula_max_abs": formula_max_abs,
        "c_feature_max_abs": c_feature_max_abs,
        "producer_payload_sha256": sha256_file(PAYLOAD_PATH),
        "producer_receipt_sha256": sha256_file(RECEIPT_PATH),
        "training_only": True,
        "eval_candidate_generation_authorized": False,
        "model_update_count": 0,
        "scientific_GO_or_NO_GO": None,
        "next_authorized_stage": (
            "MATCHED_THREE_ARM_COMMON3_NATIVE7_CROSSFIT_TRAINING"
            if passed
            else None
        ),
        "logical_sha256": "",
    }
    result["logical_sha256"] = logical_sha256(result)
    atomic_json(OUT, result)
    print(json.dumps(result, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
