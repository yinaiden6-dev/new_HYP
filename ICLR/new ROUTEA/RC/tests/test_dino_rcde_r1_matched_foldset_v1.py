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
for source_dir in (RC_ROOT / "programs", RC_ROOT / "src"):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import materialize_dino_rcde_r1_matched_foldset_v1 as materializer  # noqa: E402
import validate_dino_rcde_r1_matched_foldset_v1 as validator  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def binding(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def required(root: Path, path: Path, status: str) -> dict[str, str]:
    return {**binding(root, path), "required_status": status}


def state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0"); digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def copy_relative(root: Path, relative: str) -> Path:
    source, target = RC_ROOT / relative, root / relative
    if not source.is_file():
        pytest.skip(f"production fixture source missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def copy_bound(root: Path, spec: dict[str, Any]) -> Path:
    return copy_relative(root, str(spec["path"]))


def build_tree(root: Path) -> tuple[Path, Path]:
    # Copy immutable production evidence into an isolated fixture tree.  This
    # both keeps the test compact and exercises the exact V1.7/V1.8 boundary.
    v16 = copy_relative(root, materializer.SOURCE_CONTEXT_AUTHORITY_PATH)
    amendment = copy_relative(root, materializer.AMENDMENT_PATH)
    receipt = copy_relative(root, materializer.RECEIPT_PATH)
    bag_manifest_path = copy_relative(root, materializer.BAG_MANIFEST_PATH)
    bag_validation_path = copy_relative(root, materializer.BAG_VALIDATION_PATH)
    bag_manifest = json.loads(bag_manifest_path.read_text())
    for name in materializer.COMMON_BINDINGS:
        copy_bound(root, bag_manifest["shared_training_contract"]["common_bindings"][name])
    for row in bag_manifest["folds"]:
        for name in ("authority", "protocol", "run_identity", "access_manifest", "train_result", "training_validation", "checkpoint"):
            copy_bound(root, row[name])
        source_protocol = json.loads((root / row["protocol"]["path"]).read_text())
        for name in materializer.COMMON_BINDINGS:
            copy_bound(root, source_protocol["bindings"][name])

    context_root = "results/dino_rcde_r1_main_v1_8_accelerated_context_folds1_4"
    context_inputs: list[dict[str, Any]] = []
    for fold in materializer.FOLDS:
        protocol_path = copy_relative(root, f"protocols/dino_rcde_r1_main_train_v1_8_context_fold{fold}_20260814.json")
        directory = f"{context_root}/RCDE_CONTEXT_fold{fold}"
        paths = {
            "protocol": protocol_path,
            "run_identity": copy_relative(root, f"{directory}/run_identity.json"),
            "access_manifest": copy_relative(root, f"{directory}/signed_reference_access_manifest.json"),
            "train_result": copy_relative(root, f"{directory}/train_result.json"),
            "training_validation": copy_relative(root, f"{directory}/training_validation.json"),
            "checkpoint": copy_relative(root, f"{directory}/checkpoint_update2048.pt"),
        }
        source_protocol = json.loads(protocol_path.read_text())
        for name in materializer.COMMON_BINDINGS:
            copy_bound(root, source_protocol["bindings"][name])
        context_inputs.append({"outer_fold": fold, **{name: binding(root, path) for name, path in paths.items()}})

    implementations: dict[str, dict[str, str]] = {}
    for name, source in {
        "materializer": RC_ROOT / "programs/materialize_dino_rcde_r1_matched_foldset_v1.py",
        "validator": RC_ROOT / "programs/validate_dino_rcde_r1_matched_foldset_v1.py",
        "tests": RC_ROOT / "tests/test_dino_rcde_r1_matched_foldset_v1.py",
    }.items():
        target = root / ("tests" if name == "tests" else "programs") / source.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        implementations[name] = binding(root, target)

    authority_status = "DINO_RCDE_R1_MAIN_BAG_CONTEXT_FOLDS1_4_MATCHED_FOLDSET_CLOSURE_AUTHORIZED"
    authority_path = root / "registry/current_authority_v17_fixture.json"
    write(authority_path, {"status": authority_status, "next_authorized_stage": materializer.PROTOCOL_STAGE,
                           "natural_training_authorized": False, "protected_access_counts": copy.deepcopy(materializer.PROTECTED_ZERO),
                           "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None})
    protocol_path = root / "protocols/matched_foldset_fixture.json"
    write(protocol_path, {
        "schema_version": materializer.PROTOCOL_SCHEMA, "stage": materializer.PROTOCOL_STAGE,
        "claim_level": materializer.CLAIM_LEVEL,
        "authority": required(root, authority_path, authority_status),
        "source_context_authority": required(root, v16, materializer.SOURCE_CONTEXT_AUTHORITY_STATUS),
        "runtime_scheduler_amendment": required(root, amendment, materializer.AMENDMENT_STATUS),
        "runtime_scheduler_receipt": required(root, receipt, materializer.RECEIPT_STATUS),
        "scheduler_completion": copy.deepcopy(materializer.SCHEDULER_COMPLETION),
        "bag_foldset": {
            "manifest": required(root, bag_manifest_path, materializer.BAG_MANIFEST_STATUS),
            "validation": required(root, bag_validation_path, materializer.BAG_VALIDATION_STATUS),
        },
        "context_fold_inputs": context_inputs, "implementation_bindings": implementations,
        "output_dir": "results/matched_fixture", "forbidden": copy.deepcopy(materializer.FORBIDDEN),
        "automatic_stage_advance": False, "scientific_GO_or_NO_GO": None,
    })
    return protocol_path, root / "results/matched_fixture"


def use_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(materializer, "ROOT", root)
    monkeypatch.setattr(validator, "ROOT", root)


@pytest.fixture()
def matched_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    protocol, output_dir = build_tree(tmp_path)
    use_root(monkeypatch, tmp_path)
    return tmp_path, protocol, output_dir


def rebind_context(root: Path, protocol_path: Path, fold: int) -> None:
    protocol = json.loads(protocol_path.read_text())
    entry = next(row for row in protocol["context_fold_inputs"] if row["outer_fold"] == fold)
    for name in ("access_manifest", "train_result", "training_validation", "checkpoint"):
        entry[name] = binding(root, root / entry[name]["path"])
    write(protocol_path, protocol)


def test_two_file_end_to_end_cpu_checkpoint_and_matched_closure(
    matched_tree: tuple[Path, Path, Path],
) -> None:
    _, protocol, output_dir = matched_tree
    manifest = materializer.materialize(protocol, output_dir)
    validation = validator.validate(protocol, manifest, output_dir / "matched_foldset_validation.json")
    assert {path.name for path in output_dir.iterdir()} == {"matched_foldset_manifest.json", "matched_foldset_validation.json"}
    assert not any(path.suffix == ".pt" for path in output_dir.iterdir())
    observed = json.loads(validation.read_text())
    assert observed["status"] == validator.VALIDATION_STATUS
    assert observed["checkpoint_files_loaded_cpu"] == 8
    assert observed["checkpoint_files_copied"] == observed["heldout_forward_count"] == observed["heldout_label_join_count"] == 0


def test_materializer_rejects_cross_arm_access_drift(
    matched_tree: tuple[Path, Path, Path],
) -> None:
    root, protocol_path, output_dir = matched_tree
    protocol = json.loads(protocol_path.read_text())
    entry = next(row for row in protocol["context_fold_inputs"] if row["outer_fold"] == 4)
    access_path = root / entry["access_manifest"]["path"]
    result_path = root / entry["train_result"]["path"]
    validation_path = root / entry["training_validation"]["path"]
    access = json.loads(access_path.read_text()); removed = access["eligible_query_execution_ordinals"].pop()
    access["eligible_query_count"] -= 1; access_without_hash = dict(access); access_without_hash.pop("logical_sha256"); access["logical_sha256"] = canonical(access_without_hash); write(access_path, access)
    result = json.loads(result_path.read_text()); result["runtime_access"]["query_read_union"].remove(removed); result["signed_reference_access_manifest_sha256"] = sha(access_path); write(result_path, result)
    source_validation = json.loads(validation_path.read_text()); source_validation["eligible_query_count"] -= 1; source_validation["signed_reference_access_manifest_sha256"] = sha(access_path); source_validation["train_result_sha256"] = sha(result_path); write(validation_path, source_validation)
    rebind_context(root, protocol_path, 4)
    with pytest.raises(materializer.MatchedFoldsetAbort, match="sets differ"):
        materializer.materialize(protocol_path, output_dir)
    assert not output_dir.exists()


def test_validator_cpu_load_rejects_checkpoint_state_schema_drift(
    matched_tree: tuple[Path, Path, Path],
) -> None:
    root, protocol_path, output_dir = matched_tree
    protocol = json.loads(protocol_path.read_text()); entry = next(row for row in protocol["context_fold_inputs"] if row["outer_fold"] == 4)
    checkpoint_path, result_path, validation_path = (root / entry[name]["path"] for name in ("checkpoint", "train_result", "training_validation"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"].pop("pair_fc2.weight")
    checkpoint["final_state_sha256"] = state_sha(checkpoint["model_state_dict"]); torch.save(checkpoint, checkpoint_path)
    result = json.loads(result_path.read_text()); result["final_state_sha256"] = checkpoint["final_state_sha256"]; result["checkpoint_sha256"] = sha(checkpoint_path); write(result_path, result)
    source_validation = json.loads(validation_path.read_text()); source_validation["checkpoint_sha256"] = sha(checkpoint_path); source_validation["train_result_sha256"] = sha(result_path); write(validation_path, source_validation)
    rebind_context(root, protocol_path, 4)
    manifest = materializer.materialize(protocol_path, output_dir)
    with pytest.raises(validator.MatchedFoldsetValidationAbort, match="state cardinality"):
        validator.validate(protocol_path, manifest, output_dir / "matched_foldset_validation.json")
    assert {path.name for path in output_dir.iterdir()} == {"matched_foldset_manifest.json"}


def test_materializer_rejects_completed_job_mapping_drift(
    matched_tree: tuple[Path, Path, Path],
) -> None:
    root, protocol_path, output_dir = matched_tree
    protocol = json.loads(protocol_path.read_text()); entry = next(row for row in protocol["context_fold_inputs"] if row["outer_fold"] == 1)
    result_path, validation_path = root / entry["train_result"]["path"], root / entry["training_validation"]["path"]
    result = json.loads(result_path.read_text()); result["execution_chain"]["chunks"][0]["path"] = "chunk_job9999999/chunk.json"; write(result_path, result)
    source_validation = json.loads(validation_path.read_text()); source_validation["train_result_sha256"] = sha(result_path); write(validation_path, source_validation)
    rebind_context(root, protocol_path, 1)
    with pytest.raises(materializer.MatchedFoldsetAbort, match="JobIDRaw"):
        materializer.materialize(protocol_path, output_dir)


def test_fixed_production_scheduler_hashes_and_validator_independence() -> None:
    assert materializer.AMENDMENT_SHA256 == validator.AMENDMENT_SHA256 == "891a81b2ee0c5f286d9a53d0ae3a73b8273a0a9e1942a41003cd0483a28dae63"
    assert materializer.RECEIPT_SHA256 == validator.RECEIPT_SHA256 == "fbaf69f52c9aa6781f26b31c3a9daa22f1c8ce9bf030f2986e51da72c0c3eba4"
    source = (RC_ROOT / "programs/validate_dino_rcde_r1_matched_foldset_v1.py").read_text()
    assert "import materialize_dino_rcde_r1_matched_foldset_v1" not in source
    assert "import subprocess" not in source and "os.system(" not in source


def test_production_protocol_builds_when_present() -> None:
    protocol = RC_ROOT / "protocols/dino_rcde_r1_matched_foldset_v1_20260814.json"
    if not protocol.is_file():
        pytest.skip("production matched protocol/authority not materialized yet")
    manifest = materializer.build_manifest(protocol)
    assert manifest["status"] == materializer.MANIFEST_STATUS
    assert manifest["cross_fold_closure"]["optimization_reference_universe_count"] == 50
    assert manifest["cross_fold_closure"]["eligible_query_union_count"] == 594
