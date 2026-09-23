from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
sys.path.insert(0, str(ROOT / "src"))

import freeze_dino_rcde_track_r_relative_v_fit_authority_v119 as freezer  # noqa: E402
import run_dino_rcde_track_r_relative_v_fit_v1 as runner  # noqa: E402
import validate_dino_rcde_track_r_relative_v_fit_v1 as validator  # noqa: E402


def _binding(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": runner.file_sha256(path),
    }


def _authority() -> tuple[dict[str, object], Path]:
    roots = (freezer.RUNNER, freezer.VALIDATOR, freezer.PACKAGE_INITIALIZER)
    paths, edges = freezer.discover_runtime_closure(roots)
    rows = [
        {
            "module": relative.removesuffix(".py").replace("/", "."),
            **_binding(relative),
        }
        for relative in paths
    ]
    active_node = Path(__file__).relative_to(ROOT).as_posix()
    node_rows = {"node0": _binding(active_node)}
    node_rows.update(
        {
            f"node{ordinal}": _binding(
                "protocols/dino_rcde_track_r_relative_v_fit_protocol_v1_20260821.json"
            )
            for ordinal in range(1, 12)
        }
    )
    authority: dict[str, object] = {
        "resource_contract": {
            "array_spec": "1-4%4",
            "cpus_per_task": 4,
            "memory_megabytes": 64000,
            "gpus_per_node": 1,
            "walltime_seconds": 10800,
        },
        "bindings": {
            "runner": _binding(runner.RUNNER_PATH),
            "validator": _binding(runner.VALIDATOR_PATH),
            "launcher": _binding(runner.LAUNCHER_PATH),
            "package_initializer": _binding(runner.PACKAGE_INITIALIZER_PATH),
            "runtime_import_closure": {
                "root_programs": list(roots),
                "module_count": len(rows),
                "modules": rows,
                "module_sequence_sha256": runner.logical_sha256({"rows": rows}),
                "import_edges": edges,
                "import_edge_sequence_sha256": runner.logical_sha256(
                    {"rows": edges}
                ),
            },
            "execution_nodes": {
                "count": 12,
                "rows": node_rows,
                "logical_sha256": runner.logical_sha256({"rows": node_rows}),
            },
        },
    }
    return authority, ROOT / active_node


def test_v120_runtime_closure_and_execution_node_are_fail_closed() -> None:
    authority, active_node = _authority()
    runner.validate_runtime_bindings(authority, active_node)
    validator.validate_runtime_bindings(authority)

    authority["bindings"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(runner.TrackRVFitError):
        runner.validate_runtime_bindings(authority, active_node)
    with pytest.raises(validator.TrackRVFitValidationError):
        validator.validate_runtime_bindings(authority)


def test_v120_active_manifest_must_be_one_of_twelve_authority_nodes() -> None:
    authority, _ = _authority()
    with pytest.raises(runner.TrackRVFitError):
        runner.validate_runtime_bindings(
            authority,
            ROOT / "protocols/dino_rcde_track_r_relative_v_fit_protocol_v1_20260821.json",
        )


def test_v120_freezer_discovers_complete_local_runtime_closure() -> None:
    paths, edges = freezer.discover_runtime_closure(
        (freezer.RUNNER, freezer.VALIDATOR, freezer.PACKAGE_INITIALIZER)
    )
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert len(paths) > 10
    assert len(edges) > 10
    assert "programs/dino_rcde_track_r_input_common_v1.py" in paths
    assert "programs/run_dino_rcde_sr0_mt_v_fit_v1.py" in paths
    assert "src/rc_aslo_xf/core.py" in paths


def test_v120_launcher_and_output_lineage_fields_are_frozen() -> None:
    launcher = (ROOT / freezer.LAUNCHER).read_text(encoding="utf-8")
    run_source = (ROOT / freezer.RUNNER).read_text(encoding="utf-8")
    validation_source = (ROOT / freezer.VALIDATOR).read_text(encoding="utf-8")
    assert "#SBATCH --array=1-4%4" in launcher
    assert "#SBATCH --cpus-per-task=4" in launcher
    assert "#SBATCH --mem=64000" in launcher
    assert "#SBATCH --time=03:00:00" in launcher
    assert run_source.count('"authority_logical_sha256": authority_logical_sha') >= 3
    assert 'checkpoint.get("authority_sha256") == expected_authority_sha256' in validation_source
    assert 'input_receipt.get("authority_sha256") == expected_authority_sha256' in validation_source
