from __future__ import annotations

import json
from pathlib import Path

import freeze_dino_rcde_gx_formal_r0_science_authority_v68 as freezer
import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as helper


ROOT = Path(__file__).resolve().parents[1]


def test_v68_authority_is_reproducible_and_science_scoped() -> None:
    authority = json.loads((ROOT / freezer.AUTHORITY).read_text(encoding="utf-8"))
    assert authority == freezer.build()
    assert authority["logical_sha256"] == helper.logical(authority)
    assert authority["scientific_reduction_authorized"] is True
    assert authority["scientific_validation_authorized"] is True
    assert authority["postjoin_cached_head_replay_authorized"] is True
    assert authority["postjoin_DINO_forward_authorized"] is False
    assert authority["model_backward_or_update_authorized"] is False
    assert authority["opened_or_sealed_access_authorized"] is False
    assert authority["automatic_stage_advance"] is False
    assert authority["resource_contract"]["walltime_seconds"] == 7200
    assert authority["resource_contract"]["partition"] == "accelerated"
    assert authority["resource_contract"]["gpus_per_task"] == 1
    assert authority["resource_contract"]["GPU_compute_authorized"] is False
    launcher = (ROOT / authority["bindings"]["launcher"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --time=02:00:00" in launcher
    assert "#SBATCH --partition=accelerated" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "export CUDA_VISIBLE_DEVICES=" in launcher


def test_v68_freezes_cpn_t_exclusion_and_registered_bootstrap() -> None:
    authority = json.loads((ROOT / freezer.AUTHORITY).read_text(encoding="utf-8"))
    contract = authority["science_contract"]
    assert contract["mandatory_nulls"] == ["C_BIND", "P_COORD", "N_REGION_RESAMPLE"]
    assert contract["T_ROOT_ASSIGNMENT_POPULATION_role"] == "AUXILIARY_NON_GATING_DIAGNOSTIC_ONLY"
    assert contract["bootstrap_seed"] == 17
    assert contract["bootstrap_repetitions"] == 10000
    assert contract["bootstrap_quantile_method"] == "linear"
    assert contract["query_count"] == 32


def test_v68_binds_fourfold_outputs_and_independent_execution_envelope() -> None:
    authority = json.loads((ROOT / freezer.AUTHORITY).read_text(encoding="utf-8"))
    for fold in range(1, 5):
        for suffix in ("result", "validation", "checkpoint", "trace", "init_checkpoint"):
            assert f"fold{fold}_{suffix}" in authority["bindings"]
    for name in ("reducer", "independent_validator", "freezer", "launcher", "test_reducer", "test_validator", "test_authority"):
        binding = authority["bindings"][name]
        path = Path(binding["path"])
        path = path if path.is_absolute() else ROOT / path
        assert path.is_file()
        assert helper.file_sha(path) == binding["sha256"]
