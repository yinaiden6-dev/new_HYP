from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))

import dino_rcde_track_r_v121_lineage_v1 as lineage  # noqa: E402
import freeze_dino_rcde_track_r_oof_prejoin_authority_v120 as freezer  # noqa: E402


RUNTIME_NAMES = (
    "producer",
    "validator",
    "finalizer",
    "lineage_runtime",
    "runtime_entry_validator",
    "package_initializer",
    "launcher",
    "post_vfit_controller",
    "post_oof_controller",
)


def test_runtime_import_closure_includes_program_and_model_dependencies() -> None:
    paths, edges = freezer.discover_runtime_import_closure()
    assert paths == sorted(set(paths))
    assert edges
    for required in (
        "programs/dino_rcde_sr0_mt_v_input_common_v1.py",
        "programs/dino_rcde_track_r_input_common_v1.py",
        "src/rc_aslo_xf/dino_rcde_sr0_mt_controls_v1.py",
        "src/rc_aslo_xf/dino_rcde_sr0_mt_v_runtime_v1.py",
        "src/rc_aslo_xf/dino_rcde_v1_2_resource_core.py",
    ):
        assert required in paths


def test_authority_runtime_binding_and_shard_source_fail_closed(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    bindings = {}
    for name in RUNTIME_NAMES:
        path = runtime_dir / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        bindings[name] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": lineage.file_sha256(path),
            "bytes": path.stat().st_size,
        }
    imported = runtime_dir / "imported.py"
    imported.write_text("# imported\n", encoding="utf-8")
    closure_rows = [
        {
            "path": imported.relative_to(tmp_path).as_posix(),
            "sha256": lineage.file_sha256(imported),
            "bytes": imported.stat().st_size,
        }
    ]
    bindings["runtime_import_closure"] = {
        "count": 1,
        "rows": closure_rows,
        "logical_sha256": lineage.canonical_sha256({"rows": closure_rows}),
    }
    authority = {
        "schema_version": "rc_current_authority_v121_20260821",
        "status": "DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED",
        "bindings": bindings,
    }
    authority["logical_sha256"] = lineage.logical_sha256(authority)
    authority_path = tmp_path / "registry/current_authority_v121_20260821.json"
    authority_path.parent.mkdir()
    authority_path.write_text(
        json.dumps(authority, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    authority_path.chmod(0o444)
    observed, authority_sha256 = lineage.read_authority(
        authority_path,
        root=tmp_path,
        expected_relative="registry/current_authority_v121_20260821.json",
        expected_schema="rc_current_authority_v121_20260821",
        expected_status="DINO_RCDE_TRACK_R_OOF_PREJOIN_SHARDS_AUTHORIZED",
    )
    lineage.validate_runtime_bindings(
        observed, root=tmp_path, required_names=RUNTIME_NAMES
    )
    lineage.require_shard_source_authority(
        {"source_bindings": {"authority_sha256": authority_sha256}},
        authority_sha256,
        0,
    )
    with pytest.raises(lineage.V121LineageError, match="source-authority SHA drift"):
        lineage.require_shard_source_authority(
            {"source_bindings": {"authority_sha256": "0" * 64}},
            authority_sha256,
            0,
        )

    imported.chmod(0o644)
    imported.write_text("# drift\n", encoding="utf-8")
    with pytest.raises(lineage.V121LineageError, match="runtime binding drift"):
        lineage.validate_runtime_bindings(
            observed, root=tmp_path, required_names=RUNTIME_NAMES
        )


def test_fit_to_oof_authority_sha_and_runtime_guards_are_wired() -> None:
    fit_validator = (
        ROOT / "programs/validate_dino_rcde_track_r_relative_v_fit_v1.py"
    ).read_text(encoding="utf-8")
    oof_freezer = (
        ROOT / "programs/freeze_dino_rcde_track_r_oof_prejoin_authority_v120.py"
    ).read_text(encoding="utf-8")
    producer = (
        ROOT / "programs/materialize_dino_rcde_track_r_oof_prejoin_shard_v1.py"
    ).read_text(encoding="utf-8")
    validator = (
        ROOT / "programs/validate_dino_rcde_track_r_oof_prejoin_shard_v1.py"
    ).read_text(encoding="utf-8")
    finalizer = (
        ROOT / "programs/finalize_dino_rcde_track_r_oof_prejoin_v120.py"
    ).read_text(encoding="utf-8")
    assert 'result.get("authority_sha256") == expected_authority_sha256' in fit_validator
    assert 'checkpoint.get("authority_sha256") == expected_authority_sha256' in fit_validator
    assert '"authority_sha256": authority_sha256' in fit_validator
    assert 'validation.get("authority_sha256") == parent_sha256' in oof_freezer
    assert 'result.get("authority_sha256") == parent_sha256' in oof_freezer
    assert 'checkpoint.get("authority_sha256") == parent_sha256' in oof_freezer
    assert '"parent_v120_authority_sha256": parent_sha256' in oof_freezer
    assert "validate_runtime_bindings(" in producer
    assert "require_shard_source_authority(value, authority_sha256" in validator
    assert "require_shard_source_authority(shard, authority_sha256" in finalizer


def test_oof_resource_and_scientific_null_contract_are_unchanged() -> None:
    freezer_source = (
        ROOT / "programs/freeze_dino_rcde_track_r_oof_prejoin_authority_v120.py"
    ).read_text(encoding="utf-8")
    assert '"array_spec": "0-49%20"' in freezer_source
    assert '"cpus_per_task": 8' in freezer_source
    assert '"memory_megabytes": 128000' in freezer_source
    assert '"gpus_per_node": 1' in freezer_source
    assert '"walltime_seconds": 7200' in freezer_source
    assert '"label_target_rival_read_authorized": False' in freezer_source
    assert '"scientific_reduction_authorized": False' in freezer_source
    assert '"scientific_GO_or_NO_GO": None' in freezer_source


@pytest.mark.parametrize(
    "relative,binding_name",
    (
        ("slurm/dino_rcde_track_r_oof_prejoin_shards_v1.sbatch", "launcher"),
        ("slurm/dino_rcde_track_r_post_vfit_controller_v1.sbatch", "post_vfit_controller"),
        ("slurm/dino_rcde_track_r_post_oof_controller_v1.sbatch", "post_oof_controller"),
    ),
)
def test_shell_entrypoints_check_their_frozen_v121_binding(
    relative: str, binding_name: str
) -> None:
    source = (ROOT / relative).read_text(encoding="utf-8")
    assert "validate_dino_rcde_track_r_v121_runtime_binding_v1.py" in source
    assert f"--binding-name {binding_name}" in source
    assert "cmp -s" in source
