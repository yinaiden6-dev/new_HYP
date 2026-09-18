from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTEA = ROOT.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "programs"), str(ROUTEA / "route_a_core")]

from rc_aslo_xf.n2_current_runtime_d1_training_v1 import (
    CORRECTED_ELIGIBLE_IDENTITY_COUNTS,
    EXPECTED_ALGORITHM_DEPENDENCY_HASHES,
    EXPECTED_PARAMETER_SCHEMA_SHA256,
    algorithm_dependency_hashes,
    canonical_sha256,
    corrected_mask_audit,
    label_join_dependency_hashes,
    load_gallery,
    parameter_manifest,
    raw_prejoin_dependency_hashes,
    select_corrected_top63,
)
from route_a import o1_c6direct_m1_d1 as d1
import run_routea_matched_three_arm_n2_current_runtime_d1_fold_v1 as trainer
import validate_routea_matched_three_arm_n2_current_runtime_d1_fold_v1 as validator


PREFLIGHT = ROUTEA / "results/route_a_v3_1_o1_c6direct_m1_v1/preflight/preflight_seed17_v1.json"


def test_exact_dependency_layers_and_parameter_schema() -> None:
    assert algorithm_dependency_hashes(ROOT) == EXPECTED_ALGORITHM_DEPENDENCY_HASHES
    assert set(raw_prejoin_dependency_hashes(ROOT)) == {
        "maxsim_runtime_sha256", "corrected_d1_runtime_sha256",
        "gallery_identity_repair_runtime_sha256",
        "gallery_identity_repair_registry_sha256",
        "gallery_identity_repair_contract_sha256",
    }
    assert set(label_join_dependency_hashes(ROOT)) == {
        "corrected_d1_runtime_sha256", "gallery_identity_repair_runtime_sha256",
        "gallery_identity_repair_registry_sha256",
        "gallery_identity_repair_contract_sha256",
    }
    manifest = parameter_manifest(d1.build_fresh_d1_adapter())
    schema = [{"name": row["name"], "shape": row["shape"], "dtype": row["dtype"]} for row in manifest]
    assert len(manifest) == 8
    assert sum(row["numel"] for row in manifest) == 49_792
    assert canonical_sha256(schema) == EXPECTED_PARAMETER_SCHEMA_SHA256


def test_corrected_legal_counts_and_biogen_representative() -> None:
    _, labels, _ = load_gallery(ROOT)
    preflight = json.loads(PREFLIGHT.read_text())
    for fold in range(5):
        excluded = set(preflight["gallery_masks"][fold]["excluded_row_indices"])
        legal = torch.tensor([row not in excluded for row in range(5_413)])
        assert corrected_mask_audit(labels, legal)["eligible_corrected_identities"] == CORRECTED_ELIGIBLE_IDENTITY_COUNTS[fold]
    fold = 1
    excluded = set(preflight["gallery_masks"][fold]["excluded_row_indices"])
    legal = torch.tensor([row not in excluded for row in range(5_413)])
    scores = -torch.arange(5_413, dtype=torch.float32) / 10_000
    scores[714] = scores[715] = 10.0
    target = preflight["folds"][fold]["train_identities"][0]
    selected = select_corrected_top63(scores, labels, legal, target_identity=target)
    if bool(legal[714]):
        assert int(selected[0]) == 714
        assert 715 not in selected.tolist()
    assert len({labels[int(row)] for row in selected}) == 63


def test_schedule_implementations_are_identical_and_init_is_fixed_seed17() -> None:
    records = []
    for track, identity_count in (("outcome", 4), ("difficult", 2), ("new_difficult_train", 2)):
        for identity in range(identity_count):
            for query in range(2):
                records.append(
                    {
                        "query_id": f"{track}-{identity}-{query}",
                        "track": track,
                        "target_identity": f"{track}-id-{identity}",
                    }
                )
    assert trainer.build_schedule(records, 1) == validator.independent_schedule(records, 1)
    assert trainer.build_schedule(records, 1) != trainer.build_schedule(records, 2)
    torch.manual_seed(17)
    first = {name: value.clone() for name, value in d1.build_fresh_d1_adapter().state_dict().items()}
    torch.manual_seed(17)
    second = d1.build_fresh_d1_adapter().state_dict()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_latest_checkpoint_rejects_filename_payload_step_mismatch(tmp_path: Path) -> None:
    torch.save({"completed_steps": 25}, tmp_path / "step_0025.pt")
    assert trainer.latest_checkpoint(tmp_path).name == "step_0025.pt"
    torch.save({"completed_steps": 24}, tmp_path / "step_0050.pt")
    with pytest.raises(Exception, match="filename/payload step mismatch"):
        trainer.latest_checkpoint(tmp_path)


def test_validator_independently_reconstructs_corrected_top63() -> None:
    _, labels, _ = load_gallery(ROOT)
    fold = 1
    preflight = json.loads(PREFLIGHT.read_text())
    excluded = set(preflight["gallery_masks"][fold]["excluded_row_indices"])
    legal = torch.tensor([row not in excluded for row in range(5_413)])
    generator = torch.Generator().manual_seed(314159)
    scores = torch.randn(5_413, generator=generator, dtype=torch.float32)
    scores[714] = scores[715] = 50.0
    target = preflight["folds"][fold]["train_identities"][0]
    shared = select_corrected_top63(scores, labels, legal, target_identity=target)
    independent = validator.independent_select_corrected_top63(
        scores, labels, legal, target_identity=target
    )
    assert torch.equal(independent, shared)
    if bool(legal[714]):
        assert int(independent[0]) == 714
        assert 715 not in independent.tolist()


def test_optimizer_envelope_checks_step_recipe_and_moments() -> None:
    adapter = d1.build_fresh_d1_adapter()
    optimizer = d1.build_fresh_d1_optimizer(adapter)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in adapter.parameters()).backward()
        optimizer.step()
    state = optimizer.state_dict()
    validator.validate_optimizer_envelope(
        state, adapter.state_dict(), completed_steps=3
    )
    corrupted = copy.deepcopy(state)
    corrupted["state"][0]["step"] = torch.tensor(2.0)
    with pytest.raises(Exception, match="optimizer moment/step envelope drift"):
        validator.validate_optimizer_envelope(
            corrupted, adapter.state_dict(), completed_steps=3
        )


def test_scorer_and_trainer_do_not_open_path_bearing_query_authorities() -> None:
    protected = ("query_ledger.json", "OUTCOME_MANIFEST", "DIFFICULT_MANIFEST", "UPSTREAMS")
    for relative in (
        "programs/materialize_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py",
        "programs/validate_routea_matched_three_arm_n2_d1_raw_prejoin_shard_v1.py",
        "programs/validate_routea_matched_three_arm_n2_d1_raw_prejoin_aggregate_v1.py",
        "programs/run_routea_matched_three_arm_n2_current_runtime_d1_fold_v1.py",
    ):
        source = (ROOT / relative).read_text()
        assert all(term not in source for term in protected)
