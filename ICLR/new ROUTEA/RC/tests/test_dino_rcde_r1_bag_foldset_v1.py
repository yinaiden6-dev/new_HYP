from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pytest
import torch


RC_ROOT = Path(__file__).resolve().parents[1]
for path in (RC_ROOT / "programs", RC_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import materialize_dino_rcde_r1_bag_foldset_v1 as materializer  # noqa: E402
import validate_dino_rcde_r1_bag_foldset_v1 as validator  # noqa: E402
from rc_aslo_xf.dino_rcde_v1_2_resource_core import DINO_RCDE_V1_2  # noqa: E402


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def binding(root: Path, path: Path) -> dict[str, str]:
    return {"path": rel(root, path), "sha256": sha(path)}


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def protected() -> dict[str, int]:
    return dict(materializer.PROTECTED_ZERO)


def training(fold: int) -> dict[str, Any]:
    return {
        "arms": ["RCDE_BAG"], "outer_folds": [fold], "seed": 17,
        "parameter_count": 65125, "optimizer": "AdamW",
        "deterministic_algorithms": True, "learning_rate": 0.0003,
        "weight_decay": 0.0001, "gradient_clip_l2": 1.0,
        "updates_total": 2048, "episodes_per_update": 4,
        "raw_correct_per_update": 2, "raw_wrong_per_update": 2,
        "warmup_updates": 128, "final_learning_rate": 0.00003,
        "loss": "mean_softplus_negative_signed_pair_logit_tau_1",
        "query_tile_rows": 8, "reference_tile_rows": 16,
        "query_order_namespace": "RCDE_QUERY_ORDER_V1_1",
        "query_order_key": "source_image_sha256",
        "query_order_tie_break": "stable_episode_ledger_order",
        "candidate_direction_source": "P0_frozen_model_candidate_order",
        "cache_query_key": "execution_ordinal", "outer_train_only": True,
        "heldout_target_join_allowed": False,
    }


CACHE = {
    "root": "runtime/dino_rcde_p0_v1_2/stable_fp16_cache",
    "query_count": 600, "reference_count": 4748,
    "query_key": "execution_ordinal",
    "model_checkpoint_logical_sha256": "f901d9bd056bb65e5fcb02c72e869f6d0cf8d7472467d22fbc340442feca034c",
}
FORBIDDEN = {
    "paths": ["C8", "S8", "opened", "sealed"], "home_write": False,
    "other_fold_or_arm_training": False, "control_training": False,
    "heldout_label_join": False, "evaluation": False,
    "scientific_decision": False,
}
BAG = {
    "enabled": True, "fixed_point_free_cyclic_shift": True,
    "query_binding_namespace": "DINO_RCDE_R1_BAG_QUERY_PERMUTATION_V1_5",
    "reference_binding_namespace": "DINO_RCDE_R1_BAG_REFERENCE_PERMUTATION_V1_5",
    "valid_tokens_only": True,
}


def build_tree(root: Path, *, variant: str | None = None) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    common: dict[str, dict[str, str]] = {}
    for name in materializer.COMMON_BINDING_NAMES:
        path = root / "common" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixed-{name}\n", encoding="utf-8")
        common[name] = binding(root, path)

    # Foldset implementations are content-bound, without importing them.
    implementation_bindings = {}
    for name, source in {
        "materializer": RC_ROOT / "programs/materialize_dino_rcde_r1_bag_foldset_v1.py",
        "validator": RC_ROOT / "programs/validate_dino_rcde_r1_bag_foldset_v1.py",
        "tests": RC_ROOT / "tests/test_dino_rcde_r1_bag_foldset_v1.py",
    }.items():
        target = root / ("programs" if name != "tests" else "tests") / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        implementation_bindings[name] = binding(root, target)

    foldset_authority_path = root / "registry/foldset_authority.json"
    foldset_status = "DINO_RCDE_R1_MAIN_BAG_FOLDS1_4_FOLDSET_CLOSURE_AUTHORIZED"
    write(foldset_authority_path, {
        "status": foldset_status,
        "next_authorized_stage": materializer.PROTOCOL_STAGE,
        "natural_training_authorized": False,
        "protected_access_counts": protected(),
        "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None,
    })

    initial_model = DINO_RCDE_V1_2()
    initial_state = initial_model.state_dict()
    assert state_sha(dict(initial_state)) == materializer.INITIAL_STATE_SHA256
    ineligible = {25, 26, 101, 346, 354, 470}
    eligible_universe = sorted(set(range(600)) - ineligible)
    heldout_queries = [
        set(eligible_universe[:149]), set(eligible_universe[149:297]),
        set(eligible_universe[297:446]), set(eligible_universe[446:]),
    ]
    ref_parts = [set(range(0, 13)), set(range(13, 25)), set(range(25, 37)), set(range(37, 50))]
    fold_inputs = []
    for fold in (1, 2, 3, 4):
        directory = root / f"results/fold{fold}"
        directory.mkdir(parents=True)
        authority_path = root / f"registry/source_authority_fold{fold}.json"
        authority_status = f"SOURCE_FOLD{fold}_AUTHORIZED"
        stage = f"SOURCE_STAGE_FOLD{fold}"
        scope = {"stage": stage, "arm": "RCDE_BAG", "outer_fold": fold,
                 "optimizer_updates_total": 2048, "seed": 17}
        authority = {
            "status": authority_status, "next_authorized_stage": stage,
            "protected_access_counts": protected(), "automatic_stage_advance": False,
            "scientific_GO_or_NO_GO": None,
        }
        authority["authorized_scope" if fold == 1 else "authorized_scopes"] = scope if fold == 1 else [scope]
        write(authority_path, authority)

        protocol_path = root / f"protocols/source_fold{fold}.json"
        source_protocol = {
            "schema_version": (
                "rc_dino_rcde_r1_main_train_protocol_v1_7_contract_repair_20260813"
                if fold == 1 else "rc_dino_rcde_r1_main_train_protocol_v1_8_single_scope_20260814"
            ),
            "stage": stage,
            "authority": {**binding(root, authority_path), "required_status": authority_status},
            "bindings": copy.deepcopy(common), "cache": copy.deepcopy(CACHE),
            "training": training(fold), "forbidden": copy.deepcopy(FORBIDDEN),
            "resume": {"persistent_output": rel(root, directory)},
            "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None,
        }
        if variant == "core_drift" and fold == 4:
            drift = root / "common/rcde_core_fold4.txt"; drift.write_text("drift\n")
            source_protocol["bindings"]["rcde_core"] = binding(root, drift)
        write(protocol_path, source_protocol)
        protocol_sha = sha(protocol_path); authority_sha = sha(authority_path)

        run_identity_path = directory / "run_identity.json"
        run_identity = {
            "schema_version": (
                "rc_dino_rcde_r1_main_run_identity_v1_7" if fold == 1
                else "rc_dino_rcde_r1_run_identity_v1_8"
            ),
            "arm": "RCDE_BAG", "outer_fold": fold,
            "protocol_sha256": protocol_sha, "authority_sha256": authority_sha,
        }
        if fold != 1: run_identity["status"] = "DINO_RCDE_R1_RUN_IDENTITY_FROZEN"
        write(run_identity_path, run_identity)

        access_path = directory / "signed_reference_access_manifest.json"
        heldout_refs = sorted(ref_parts[fold - 1]); allowed_refs = sorted(set(range(50)) - set(heldout_refs))
        eligible = sorted(set(eligible_universe) - heldout_queries[fold - 1])
        access = {
            "schema_version": (
                "rc_dino_rcde_signed_reference_access_manifest_v1_7" if fold == 1
                else "rc_dino_rcde_signed_reference_access_manifest_v1_8"
            ),
            "status": "DINO_RCDE_SIGNED_REFERENCE_ACCESS_FROZEN",
            "outer_fold": fold, "eligible_query_execution_ordinals": eligible,
            "eligible_query_count": len(eligible),
            "allowed_reference_physical_rows": allowed_refs,
            "allowed_reference_count": len(allowed_refs),
            "heldout_reference_physical_rows": heldout_refs,
            "heldout_reference_count": len(heldout_refs),
            "reference_row_intersection_count": 0,
            "protected_access_counts": protected(),
        }
        if variant == "protected_nonzero" and fold == 3:
            access["protected_access_counts"]["C8_runtime_read_count"] = 1
        access["logical_sha256"] = canonical(access)
        write(access_path, access)

        final_state = {name: tensor.detach().clone() for name, tensor in initial_state.items()}
        final_state["beta"] = final_state["beta"] + fold
        if variant == "bad_state_schema" and fold == 4:
            final_state.pop("pair_fc2.weight")
        final_sha = state_sha(final_state)
        checkpoint_path = directory / "checkpoint_update2048.pt"
        checkpoint = {
            "schema_version": (
                "rc_dino_rcde_r1_main_checkpoint_v1_7" if fold == 1
                else "rc_dino_rcde_r1_main_checkpoint_v1_8"
            ),
            "arm": "RCDE_BAG", "outer_fold": fold, "update": 2048, "seed": 17,
            "protocol_sha256": protocol_sha, "authority_sha256": authority_sha,
            "initial_state_sha256": materializer.INITIAL_STATE_SHA256,
            "final_state_sha256": final_sha, "model_state_dict": final_state,
        }
        if fold != 1: checkpoint["status"] = "DINO_RCDE_R1_MAIN_CHECKPOINT_COMPLETE"
        torch.save(checkpoint, checkpoint_path)

        result_path = directory / "train_result.json"
        result = {
            "schema_version": (
                "rc_dino_rcde_r1_main_train_result_v1_7" if fold == 1
                else "rc_dino_rcde_r1_main_train_result_v1_8"
            ),
            "status": (
                "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAIN_COMPLETE" if fold == 1
                else "DINO_RCDE_R1_MAIN_ARM_FOLD_TRAINING_COMPLETE"
            ),
            "arm": "RCDE_BAG", "outer_fold": fold, "parameter_count": 65125,
            "initial_state_sha256": materializer.INITIAL_STATE_SHA256,
            "final_state_sha256": final_sha, "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha(checkpoint_path),
            "protocol_sha256": protocol_sha, "authority_sha256": authority_sha,
            "run_identity_sha256": sha(run_identity_path),
            "signed_reference_access_manifest_sha256": sha(access_path),
            "bag_permutation": copy.deepcopy(BAG),
            "training": {"update_count": 2048, "pair_count": 65536, "seed": 17,
                         "schedule_sha256": hashlib.sha256(f"schedule{fold}".encode()).hexdigest()},
            "runtime_access": {"heldout_reference_intersection_count": 0},
            "protected_access_counts": protected(),
            "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None,
        }
        write(result_path, result)

        validation_path = directory / "training_validation.json"
        source_validation = {
            "schema_version": (
                "rc_dino_rcde_r1_main_train_validation_v1_7_contract_repair_20260813"
                if fold == 1 else "rc_dino_rcde_r1_main_training_validation_v1_8"
            ),
            "status": "DINO_RCDE_R1_MAIN_ARM_FOLD_VALIDATION_PASS",
            "arm": "RCDE_BAG", "outer_fold": fold, "update_count": 2048,
            "pair_count": 65536, "eligible_query_count": len(eligible),
            "allowed_reference_count": len(allowed_refs),
            "heldout_reference_intersection_count": 0,
            "schedule_sha256": result["training"]["schedule_sha256"],
            "protocol_sha256": protocol_sha, "authority_sha256": authority_sha,
            "train_result_sha256": sha(result_path),
            "run_identity_sha256": sha(run_identity_path),
            "signed_reference_access_manifest_sha256": sha(access_path),
            "checkpoint_sha256": sha(checkpoint_path),
            "binding_hashes": {name: spec["sha256"] for name, spec in source_protocol["bindings"].items()},
            "checks": {"fixture_independent_validation": True},
            "protected_access_counts": protected(),
            "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None,
        }
        write(validation_path, source_validation)
        fold_inputs.append({
            "outer_fold": fold, "authority": binding(root, authority_path),
            "protocol": binding(root, protocol_path),
            "run_identity": binding(root, run_identity_path),
            "access_manifest": binding(root, access_path),
            "train_result": binding(root, result_path),
            "training_validation": binding(root, validation_path),
            "checkpoint": binding(root, checkpoint_path),
        })

    if variant == "duplicate_fold":
        fold_inputs[3] = copy.deepcopy(fold_inputs[2])
    foldset_protocol_path = root / "protocols/foldset.json"
    write(foldset_protocol_path, {
        "schema_version": materializer.PROTOCOL_SCHEMA,
        "stage": materializer.PROTOCOL_STAGE,
        "claim_level": materializer.CLAIM_LEVEL,
        "authority": {**binding(root, foldset_authority_path), "required_status": foldset_status},
        "implementation_bindings": implementation_bindings,
        "output_dir": "results/foldset",
        "fold_inputs": fold_inputs,
        "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None,
    })
    return foldset_protocol_path, root / "results/foldset"


def use_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(materializer, "ROOT", root)
    monkeypatch.setattr(validator, "ROOT", root)


def test_two_file_end_to_end_and_checkpoint_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, output_dir = build_tree(tmp_path)
    use_root(monkeypatch, tmp_path)
    manifest = materializer.materialize(protocol, output_dir)
    validation = validator.validate(protocol, manifest, output_dir / "foldset_validation.json")
    assert {p.name for p in output_dir.iterdir()} == {"foldset_manifest.json", "foldset_validation.json"}
    assert json.loads(validation.read_text())["status"] == validator.VALIDATION_STATUS
    assert json.loads(manifest.read_text())["cross_fold_closure"]["eligible_query_union_count"] == 594
    assert not any(p.suffix == ".pt" for p in output_dir.iterdir())


@pytest.mark.parametrize("variant", ["core_drift", "protected_nonzero", "duplicate_fold"])
def test_materializer_fails_closed_on_cross_fold_or_protected_drift(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, output_dir = build_tree(tmp_path, variant=variant)
    use_root(monkeypatch, tmp_path)
    with pytest.raises(materializer.FoldsetAbort):
        materializer.materialize(protocol, output_dir)
    assert not output_dir.exists()


def test_validator_loads_checkpoint_and_rejects_state_schema_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, output_dir = build_tree(tmp_path, variant="bad_state_schema")
    use_root(monkeypatch, tmp_path)
    manifest = materializer.materialize(protocol, output_dir)
    with pytest.raises(validator.FoldsetValidationAbort, match="state cardinality"):
        validator.validate(protocol, manifest, output_dir / "foldset_validation.json")
    assert {p.name for p in output_dir.iterdir()} == {"foldset_manifest.json"}


def test_independent_validator_does_not_import_materializer() -> None:
    source = (RC_ROOT / "programs/validate_dino_rcde_r1_bag_foldset_v1.py").read_text()
    assert "import materialize_dino_rcde_r1_bag_foldset_v1" not in source
