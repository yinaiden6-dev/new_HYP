from __future__ import annotations

import json
from pathlib import Path

import pytest

import freeze_dino_rcde_h0_c128_target_free_postjoin_pj2_authority_v1 as F


def _write(path: Path, value: dict) -> None:
    value["logical_sha256"] = F.logical(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    path.chmod(0o444)


def _lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    authority = tmp_path / "pj1_authority.json"
    result = tmp_path / "pj1_result.json"
    validation = tmp_path / "pj1_validation.json"
    _write(
        authority,
        {
            "namespace": F.NS,
            "status": "H0_C128_TARGET_FREE_POSTJOIN_PJ1_EXECUTION_AUTHORIZED",
            "scientific_reduction_authorized": False,
            "scientific_GO_or_NO_GO": None,
        },
    )
    _write(
        result,
        {
            "namespace": F.NS,
            "status": "H0_C128_TARGET_FREE_POSTJOIN_PJ1_READY",
            "authority_sha256": F.fsha(authority),
            "authority_logical_sha256": json.loads(authority.read_text())["logical_sha256"],
            "query_count": 594,
            "candidate_count": 76_032,
            "raw_directional_arm_logit_count": 304_128,
            "target_insertions": 0,
            "candidate_mutations": 0,
            "direction_reduction_count": 0,
            "winner_rank_margin_count": 0,
            "scientific_GO_or_NO_GO": None,
        },
    )
    _write(
        validation,
        {
            "namespace": F.NS,
            "status": "H0_C128_TARGET_FREE_POSTJOIN_PJ1_INDEPENDENT_VALIDATION_PASS",
            "validation_pass": True,
            "authority_sha256": F.fsha(authority),
            "authority_logical_sha256": json.loads(authority.read_text())["logical_sha256"],
            "result_sha256": F.fsha(result),
            "result_logical_sha256": json.loads(result.read_text())["logical_sha256"],
            "query_count": 594,
            "candidate_count": 76_032,
            "raw_directional_arm_logit_count": 304_128,
            "independent_full_shard_replay": True,
            "independent_label_join_replay": True,
            "scientific_metric_count": 0,
            "scientific_GO_or_NO_GO": None,
        },
    )
    monkeypatch.setattr(F, "PJ1AUTH", authority)
    monkeypatch.setattr(F, "PJ1", result)
    monkeypatch.setattr(F, "PJ1V", validation)
    original_bind = F.bind

    def bind(path: Path, logical_hash: bool = False) -> dict:
        if path in (authority, result, validation):
            value = {"path": str(path), "bytes": path.stat().st_size, "sha256": F.fsha(path)}
            if logical_hash:
                value["logical_sha256"] = json.loads(path.read_text())["logical_sha256"]
            return value
        if path.name == "h0_c128_target_free_postjoin_handoff_authority_v1_20260824.json":
            return {
                "path": str(path.relative_to(F.ROOT)),
                "bytes": 1,
                "sha256": "0" * 64,
                "logical_sha256": "1" * 64,
            }
        return original_bind(path, logical_hash)

    monkeypatch.setattr(F, "bind", bind)
    return authority, result, validation


def test_pj2_freezer_accepts_only_independently_passed_immutable_pj1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _lineage(tmp_path, monkeypatch)
    value = F.build()
    assert value["namespace"] == F.NS
    assert value["status"] == F.STATUS
    assert value["scientific_reduction_authorized"] is True
    assert value["automatic_stage_advance"] is False
    assert value["next_authorized_stage"] is None
    assert value["formula_contract"]["forbidden_cross_arm_margin_difference"] is True


def test_pj2_freezer_fails_closed_on_validation_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, validation = _lineage(tmp_path, monkeypatch)
    validation.chmod(0o644)
    payload = json.loads(validation.read_text(encoding="utf-8"))
    payload["validation_pass"] = False
    payload["logical_sha256"] = F.logical(payload)
    validation.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    validation.chmod(0o444)
    with pytest.raises(F.E, match="PJ1 validation drift"):
        F.build()
