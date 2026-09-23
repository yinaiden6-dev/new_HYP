from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
from typing import Any

import pytest


RC_ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = RC_ROOT / "programs"
SRC = RC_ROOT / "src"
for import_root in (str(PROGRAMS), str(SRC)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import run_dino_rcde_r1_main_train_v1_8_single_scope as runner  # noqa: E402
import validate_dino_rcde_r1_main_train_v1_8_single_scope as validator  # noqa: E402


STAGE = "R1_MAIN_BAG_FOLDS2_4_V18_ACCELERATED_TRAINING"
ROLES_RELATIVE = "protocols/dino_rcde_training_roles_600_v1_2_20260812.json"
LEDGER_RELATIVE = (
    "results/dino_rcde_p0_v1_2/finalize_repair_job5069473/data/"
    "fold_access_and_episode_receipt.json"
)
SHAPE_RELATIVE = (
    "results/dino_rcde_p0_v1_2/finalize_repair_job5069473/data/"
    "resource_shape_ledger.json"
)
PROTOCOL_TEMPLATE = (
    "protocols/dino_rcde_r1_main_train_v1_8_bag_fold{fold}_20260814.json"
)
AUTHORITY_RELATIVE = "registry/current_authority_v14_20260814.json"
SLURM_RELATIVE = (
    "slurm/dino_rcde_r1_main_bag_folds2_4_v1_8_accelerated_4h.sbatch"
)
OUTPUT_TEMPLATE = (
    "results/dino_rcde_r1_main_v1_8_accelerated_folds2_4/RCDE_BAG_fold{fold}"
)


def _artifact_contract() -> dict[str, dict[str, str]]:
    return {
        name: {
            "schema_version": f"rc_test_{name}_v1_8",
            "status": f"TEST_{name.upper()}_V1_8",
        }
        for name in sorted(runner.ARTIFACT_CONTRACT_KEYS)
    }


def _scope(fold: int, *, arm: str = "RCDE_BAG", stage: str = STAGE) -> dict[str, Any]:
    return {
        "stage": stage,
        "arm": arm,
        "outer_fold": fold,
        "optimizer_updates_total": 2_048,
        "seed": 17,
        "maximum_submitted_jobs_in_chain": 1,
        "maximum_concurrently_running_tasks": 3,
    }


def _authority() -> dict[str, Any]:
    return {
        "next_authorized_stage": STAGE,
        "authorized_scopes": [_scope(fold) for fold in (2, 3, 4)],
    }


def _protocol(fold: int, *, arm: str = "RCDE_BAG", stage: str = STAGE) -> dict[str, Any]:
    return {
        "stage": stage,
        "training": {"arms": [arm], "outer_folds": [fold]},
        "artifact_contract": _artifact_contract(),
        "execution": {"runner_runtime_guard_seconds": 13_800},
        "bindings": {
            "training_roles": {"path": ROLES_RELATIVE},
            "episode_ledger": {"path": LEDGER_RELATIVE},
            "resource_shape_ledger": {"path": SHAPE_RELATIVE},
        },
    }


@pytest.fixture(scope="module")
def frozen_ledgers() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return tuple(
        json.loads((RC_ROOT / relative).read_text(encoding="utf-8"))
        for relative in (ROLES_RELATIVE, LEDGER_RELATIVE, SHAPE_RELATIVE)
    )  # type: ignore[return-value]


@pytest.mark.parametrize("fold", (2, 3, 4))
def test_authority_exactly_accepts_each_frozen_bag_scope(fold: int) -> None:
    protocol = _protocol(fold)
    authority = _authority()
    assert runner.single_scope(protocol, authority) == ("RCDE_BAG", fold)
    assert (
        validator.authorized_scope_member(
            authority, arm="RCDE_BAG", fold=fold, stage=STAGE
        )
        == authority["authorized_scopes"][fold - 2]
    )


@pytest.mark.parametrize(
    ("protocol", "message"),
    (
        (_protocol(1), "not authorized"),
        (_protocol(2, arm="RCDE_CONTEXT"), "not authorized"),
        (_protocol(2, stage="WRONG_STAGE"), "not authorized"),
    ),
)
def test_authority_rejects_fold1_arm_and_stage_mismatches(
    protocol: dict[str, Any], message: str
) -> None:
    authority = _authority()
    with pytest.raises(runner.ContractRepairAbort, match=message):
        runner.single_scope(protocol, authority)
    with pytest.raises(validator.ValidationAbort, match="not one exact authorized member"):
        validator.authorized_scope_member(
            authority,
            arm=str(protocol["training"]["arms"][0]),
            fold=int(protocol["training"]["outer_folds"][0]),
            stage=str(protocol["stage"]),
        )


def test_authority_rejects_duplicate_scope_members() -> None:
    authority = _authority()
    authority["authorized_scopes"].append(copy.deepcopy(authority["authorized_scopes"][0]))
    with pytest.raises(runner.ContractRepairAbort, match="duplicate scopes"):
        runner.single_scope(_protocol(2), authority)
    with pytest.raises(validator.ValidationAbort, match="duplicate scope"):
        validator.authorized_scope_member(
            authority, arm="RCDE_BAG", fold=2, stage=STAGE
        )


@pytest.mark.parametrize(
    ("arms", "folds", "message"),
    (
        (["RCDE_BAG", "RCDE_CONTEXT"], [2], "exactly one allowed arm"),
        (["RCDE_BAG"], [2, 3], "exactly one outer fold"),
        ([], [2], "exactly one allowed arm"),
        (["RCDE_BAG"], [], "exactly one outer fold"),
    ),
)
def test_protocol_must_freeze_one_arm_and_one_fold(
    arms: list[str], folds: list[int], message: str
) -> None:
    protocol = _protocol(2)
    protocol["training"] = {"arms": arms, "outer_folds": folds}
    with pytest.raises(runner.ContractRepairAbort, match=message):
        runner.single_scope(protocol, _authority())


def test_nested_nine_item_artifact_contract_is_accepted_by_both_programs() -> None:
    protocol = {"artifact_contract": _artifact_contract()}
    nested = runner.artifact_contract(protocol)
    flattened = validator.artifact_contract(protocol)
    assert set(nested) == set(runner.ARTIFACT_CONTRACT_KEYS)
    assert set(flattened) == {
        f"{name}_{field}"
        for name in runner.ARTIFACT_CONTRACT_KEYS
        for field in ("schema", "status")
    }
    for name, spec in nested.items():
        assert flattened[f"{name}_schema"] == spec["schema_version"]
        assert flattened[f"{name}_status"] == spec["status"]


def _artifact_mutations() -> list[dict[str, Any]]:
    valid = _artifact_contract()
    missing = copy.deepcopy(valid)
    missing.pop("validation")
    extra = copy.deepcopy(valid)
    extra["undeclared"] = {
        "schema_version": "rc_test_undeclared_v1_8",
        "status": "UNDECLARED",
    }
    old_flat = {
        f"{name}_{field}": value["schema_version" if field == "schema" else "status"]
        for name, value in valid.items()
        for field in ("schema", "status")
    }
    missing_nested_field = copy.deepcopy(valid)
    missing_nested_field["result"].pop("status")
    extra_nested_field = copy.deepcopy(valid)
    extra_nested_field["result"]["path"] = "train_result.json"
    old_schema = copy.deepcopy(valid)
    old_schema["result"]["schema_version"] = "rc_test_result_v1_7"
    return [
        missing,
        extra,
        old_flat,  # type: ignore[list-item]
        missing_nested_field,
        extra_nested_field,
        old_schema,
    ]


@pytest.mark.parametrize("contract", _artifact_mutations())
def test_artifact_contract_rejects_missing_extra_flat_or_non_v18_forms(
    contract: dict[str, Any]
) -> None:
    protocol = {"artifact_contract": contract}
    with pytest.raises(runner.ContractRepairAbort):
        runner.artifact_contract(protocol)
    with pytest.raises(validator.ValidationAbort):
        validator.artifact_contract(protocol)


class _ReachedInputClosure(RuntimeError):
    pass


def _invoke_runner_through_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    arm: str,
    fold: int,
    runtime_guard: int,
) -> None:
    protocol = {
        "execution": {"runner_runtime_guard_seconds": 13_800},
        "artifact_contract": _artifact_contract(),
    }
    monkeypatch.setattr(
        runner,
        "validate_contract",
        lambda *_: (protocol, "RCDE_BAG", 2),
    )
    monkeypatch.setattr(
        runner,
        "input_closure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_ReachedInputClosure()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            runner.__file__,
            "--protocol",
            str(RC_ROOT / "protocols/never_read.json"),
            "--authority",
            str(RC_ROOT / "registry/never_read.json"),
            "--cache-root",
            str(RC_ROOT / "runtime"),
            "--output-dir",
            str(RC_ROOT / "results/v18_unit_never_created"),
            "--arm",
            arm,
            "--fold",
            str(fold),
            "--max-runtime-seconds",
            str(runtime_guard),
        ],
    )
    runner.main()


@pytest.mark.parametrize(
    ("arm", "fold", "runtime_guard", "message"),
    (
        ("RCDE_CONTEXT", 2, 13_800, "CLI arm/fold"),
        ("RCDE_BAG", 3, 13_800, "CLI arm/fold"),
        ("RCDE_BAG", 2, 13_799, "CLI runtime guard"),
    ),
)
def test_cli_scope_and_runtime_guard_must_match_protocol_exactly(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    fold: int,
    runtime_guard: int,
    message: str,
) -> None:
    with pytest.raises(runner.ContractRepairAbort, match=message):
        _invoke_runner_through_cli(
            monkeypatch, arm=arm, fold=fold, runtime_guard=runtime_guard
        )


def test_exact_cli_scope_and_runtime_reach_input_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(_ReachedInputClosure):
        _invoke_runner_through_cli(
            monkeypatch, arm="RCDE_BAG", fold=2, runtime_guard=13_800
        )


def test_folds2_3_4_real_populations_and_schedules_rebuild_independently(
    frozen_ledgers: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    roles, ledger, shape = frozen_ledgers
    observed_schedule_hashes: set[str] = set()
    observed_query_sets: set[tuple[int, ...]] = set()
    for fold in (2, 3, 4):
        correct, wrong = runner.ordered_pools(ledger, roles, shape, fold)
        population = validator.common.build_population(_protocol(fold), fold)
        assert [row["query_id"] for row in correct] == [
            row["query_id"] for row in population["correct"]
        ]
        assert [row["query_id"] for row in wrong] == [
            row["query_id"] for row in population["wrong"]
        ]
        assert sorted(
            int(row["execution_ordinal"]) for row in correct + wrong
        ) == population["eligible_queries"]
        assert not set(population["allowed_references"]).intersection(
            population["heldout_references"]
        )
        assert not population["train_identities"].intersection(
            population["heldout_identities"]
        )
        assert not population["train_groups"].intersection(
            population["heldout_groups"]
        )

        reconstructed = validator.common.expected_schedule(population, fold)
        runner_sha = runner.schedule_prefix(
            correct, wrong, fold, runner.base.UPDATE_COUNT
        ).hexdigest()
        assert runner_sha == reconstructed["sha256"]
        assert reconstructed["pair_count"] == 65_536
        observed_schedule_hashes.add(runner_sha)
        observed_query_sets.add(tuple(population["eligible_queries"]))

    assert len(observed_schedule_hashes) == 3
    assert len(observed_query_sets) == 3


def test_v17_frozen_execution_files_are_byte_unchanged() -> None:
    expected = {
        "programs/run_dino_rcde_r1_main_train_v1_7_contract_repair.py": (
            "2eabbb95f0981299d56cbbaccb20d518b9a7d9b7ea7a767f7139069a68d70139"
        ),
        "programs/validate_dino_rcde_r1_main_train_v1_7_contract_repair.py": (
            "1ec6dd73697ebfb91f3db61acbca66c85427a1337068c4bd670f489275fc1f5e"
        ),
        "programs/smoke_dino_rcde_r1_main_v1_7_gpu.py": (
            "4f65810129ae0d6cc81141c8381b08ee6ee37caa212f302b95f890794f576289"
        ),
        "tests/test_dino_rcde_r1_main_v1_7_contract_repair.py": (
            "3510271a45bb3049944cd7972985dc9fdd6b8943960e05541e155a40cb2bbc25"
        ),
        "slurm/dino_rcde_r1_main_bag_fold1_v1_7_contract_repair_dev_1h.sbatch": (
            "f5bd89b6f94b8b727114397f611f603318351dd5008aff247e91f29a13629272"
        ),
    }
    assert {
        relative: validator.common.file_sha256(RC_ROOT / relative)
        for relative in expected
    } == expected


@pytest.mark.skipif(
    not (RC_ROOT / SLURM_RELATIVE).is_file(),
    reason="production V1.8 accelerated array entry has not been frozen yet",
)
def test_production_array_maps_each_fold_to_unique_protocol_output_lock_and_tmp() -> None:
    text = (RC_ROOT / SLURM_RELATIVE).read_text(encoding="utf-8")
    assert re.search(r"^#SBATCH\s+--partition=accelerated\s*$", text, re.MULTILINE)
    assert re.search(r"^#SBATCH\s+--array=2-4%3\s*$", text, re.MULTILINE)
    assert re.search(r"rc_fold=.*SLURM_ARRAY_TASK_ID", text)
    assert "dino_rcde_r1_main_train_v1_8_bag_fold${rc_fold}_20260814.json" in text
    assert "RCDE_BAG_fold${rc_fold}" in text
    assert ".RCDE_BAG_fold${rc_fold}.runner.lock" in text
    runtime_line = next(line for line in text.splitlines() if "rc_runtime_cache=" in line)
    assert (
        "SLURM_ARRAY_TASK_ID" in runtime_line
        or "SLURM_JOB_ID" in runtime_line
        or ("SLURM_ARRAY_JOB_ID" in runtime_line and "rc_fold" in runtime_line)
    )
    assert "--arm RCDE_BAG" in text
    assert re.search(r"--fold\s+\"?\$\{?rc_fold", text)
    assert re.search(r"--max-runtime-seconds\s+13800", text)
    # Validator must validate the same isolated scope; no cross-fold shared output.
    assert text.count("--output-dir \"$rc_output\"") >= 2
    assert re.search(r"--arm\s+RCDE_BAG", text)


PRODUCTION_PROTOCOLS_READY = all(
    (RC_ROOT / PROTOCOL_TEMPLATE.format(fold=fold)).is_file() for fold in (2, 3, 4)
) and (RC_ROOT / AUTHORITY_RELATIVE).is_file()


@pytest.mark.skipif(
    not PRODUCTION_PROTOCOLS_READY,
    reason="production V1.8 protocols/authority have not been frozen yet",
)
def test_production_protocols_freeze_three_isolated_single_scopes() -> None:
    authority = json.loads((RC_ROOT / AUTHORITY_RELATIVE).read_text(encoding="utf-8"))
    keys = {
        (scope["arm"], int(scope["outer_fold"]), scope["stage"])
        for scope in authority["authorized_scopes"]
    }
    assert keys == {("RCDE_BAG", fold, STAGE) for fold in (2, 3, 4)}
    outputs: set[str] = set()
    for fold in (2, 3, 4):
        path = RC_ROOT / PROTOCOL_TEMPLATE.format(fold=fold)
        protocol = json.loads(path.read_text(encoding="utf-8"))
        assert runner.single_scope(protocol, authority) == ("RCDE_BAG", fold)
        runner.artifact_contract(protocol)
        validator.artifact_contract(protocol)
        output = str(protocol["resume"]["persistent_output"])
        assert output == OUTPUT_TEMPLATE.format(fold=fold)
        outputs.add(output)
        assert protocol["execution"]["runner_runtime_guard_seconds"] == 13_800
        assert protocol["execution"]["slurm_path"] == SLURM_RELATIVE
    assert len(outputs) == 3
