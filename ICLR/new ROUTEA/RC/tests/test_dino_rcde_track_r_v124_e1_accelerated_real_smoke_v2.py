from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import freeze_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2 as freezer  # noqa: E402
import validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_authority_v2 as validator  # noqa: E402
import run_dino_rcde_track_r_v124_e1_accelerated_real_smoke_v2 as smoke_producer  # noqa: E402
import validate_dino_rcde_track_r_v124_e1_accelerated_real_smoke_independent_v2 as smoke_validator  # noqa: E402
from rc_aslo_xf import dino_rcde_track_r_v124_e1_real_smoke_authority_v2 as S  # noqa: E402


def relogical(value: dict) -> dict:
    value["logical_sha256"] = S.BASE.V2.logical_sha256(value)
    return value


def test_exact_overlay_path_map_and_launcher_population_pass() -> None:
    value = freezer.build_authority()
    result = validator.validate_candidate(value)
    assert result["validation_pass"] is True
    assert result["base_v3_binding_count"] == 115
    assert result["base_v3_import_closure_count"] == 65
    assert result["overlay_binding_count"] == len(S.OVERLAY_PATHS) == 11
    assert result["overlay_import_closure_count"] == 9
    assert result["launcher_test_path_count"] == 5
    assert result["launcher_executable_path_count"] == 3
    assert result["eligible_as_e1_result"] is False
    assert result["submission_authorized"] is False
    assert not freezer.OUTPUT.exists()


def test_replace_smoke_producer_row_with_smoke_tests_duplicate_poison() -> None:
    value = copy.deepcopy(freezer.build_authority())
    value["bindings"]["smoke_producer"] = copy.deepcopy(
        value["bindings"]["smoke_tests"]
    )
    relogical(value)
    with pytest.raises(Exception):
        S.validate_authority_envelope(value)


def test_duplicate_overlay_path_poison() -> None:
    value = copy.deepcopy(freezer.build_authority())
    value["bindings"]["smoke_validator_v1_parent"]["path"] = value[
        "bindings"
    ]["smoke_producer_v1_parent"]["path"]
    relogical(value)
    with pytest.raises(Exception):
        S.validate_authority_envelope(value)


def test_missing_launcher_executed_test_path_poison() -> None:
    value = freezer.build_authority()
    launcher = (
        ROOT / S.OVERLAY_PATHS["smoke_launcher"]
    ).read_text(encoding="utf-8")
    missing = launcher.replace(S.LAUNCHER_TEST_PATHS[-1], "", 1)
    with pytest.raises(Exception):
        S.validate_launcher_source(value, missing)


def test_swapped_overlay_keys_poison() -> None:
    value = copy.deepcopy(freezer.build_authority())
    first = copy.deepcopy(value["bindings"]["smoke_producer"])
    second = copy.deepcopy(value["bindings"]["smoke_independent_validator"])
    value["bindings"]["smoke_producer"] = second
    value["bindings"]["smoke_independent_validator"] = first
    relogical(value)
    with pytest.raises(Exception):
        S.validate_authority_envelope(value)


@pytest.mark.parametrize(
    "key",
    ["smoke_authority_freezer", "smoke_authority_independent_reviewer"],
)
def test_missing_authority_program_binding_poison(key: str) -> None:
    value = copy.deepcopy(freezer.build_authority())
    value["bindings"].pop(key)
    value["binding_modes"].pop(key)
    value["smoke_overlay_paths"].pop(key)
    relogical(value)
    with pytest.raises(Exception):
        S.validate_authority_envelope(value)


def test_launcher_actual_paths_each_exactly_once() -> None:
    value = freezer.build_authority()
    source = (
        ROOT / S.OVERLAY_PATHS["smoke_launcher"]
    ).read_text(encoding="utf-8")
    S.validate_launcher_source(value, source)
    for path in S.LAUNCHER_TEST_PATHS:
        assert source.count(path) == 1
    for path in S.LAUNCHER_EXECUTABLE_PATHS.values():
        assert source.count(path) == 1


def test_smoke_v2_is_separate_nonpromotable_and_v1_rejected() -> None:
    value = freezer.build_authority()
    assert value["eligible_as_e1_result"] is False
    assert value["output_contract"]["eligible_as_e1_result"] is False
    assert "real_smoke_v2_committed" in S.PRODUCER_NAMESPACE
    assert "real_smoke_validation_v2_committed" in S.VALIDATION_NAMESPACE
    assert (
        value["bindings"]["smoke_authority_v1_terminal_rejection"]["path"]
        == S.OVERLAY_PATHS["smoke_authority_v1_terminal_rejection"]
    )
    assert not (ROOT / S.PRODUCER_NAMESPACE).exists()
    assert not (ROOT / S.VALIDATION_NAMESPACE).exists()
    closure = set(S.discover_overlay_import_closure())
    assert S.OVERLAY_PATHS["smoke_authority_freezer"] in closure
    assert (
        S.OVERLAY_PATHS["smoke_authority_independent_reviewer"]
        in closure
    )


def test_smoke_v2_wrappers_configure_only_v2_nonpromotable_namespaces() -> None:
    smoke_producer._configure()
    smoke_validator._configure()
    assert smoke_producer.PUBLIC_DIR == ROOT / S.PRODUCER_NAMESPACE
    assert smoke_validator.PRODUCER_DIR == ROOT / S.PRODUCER_NAMESPACE
    assert smoke_validator.PUBLIC_DIR == ROOT / S.VALIDATION_NAMESPACE
    assert smoke_producer.STATUS.endswith("NONPROMOTABLE")
    assert smoke_validator.STATUS.endswith("NONPROMOTABLE")
    assert "real_smoke_v2_committed" in str(smoke_producer.PUBLIC_DIR)
    assert "real_smoke_validation_v2_committed" in str(
        smoke_validator.PUBLIC_DIR
    )
    assert "real_smoke_v1_committed" not in str(smoke_producer.PUBLIC_DIR)


def test_raw_authority_alias_symlink_rejected_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "authority.json"
    real.write_text("{}\n", encoding="ascii")
    real.chmod(0o444)
    alias = tmp_path / "authority-alias.json"
    alias.symlink_to(real)
    monkeypatch.setattr(S, "AUTHORITY_PATH", str(alias))
    with pytest.raises(Exception, match="symlink"):
        S.read_authority(
            alias, hashlib.sha256(real.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("key", sorted(S.required_bindings()))
def test_every_base_and_overlay_bound_symlink_poison(
    tmp_path: Path, key: str
) -> None:
    real = tmp_path / f"{key}.real"
    real.write_text("x", encoding="ascii")
    real.chmod(0o444)
    alias = tmp_path / f"{key}.alias"
    alias.symlink_to(real)
    with pytest.raises(Exception, match="symlink"):
        S._raw_regular_file(
            alias, expected_mode=0o444, expected_lexical=alias
        )


@pytest.mark.parametrize(
    "namespace_name",
    [
        "PRODUCER_NAMESPACE",
        "VALIDATION_NAMESPACE",
        "FORMAL_PRODUCER_NAMESPACE",
        "FORMAL_VALIDATION_NAMESPACE",
    ],
)
def test_dangling_smoke_and_formal_namespace_symlink_poison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    namespace_name: str,
) -> None:
    dangling = tmp_path / f"{namespace_name}.dangling"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    for name in (
        "PRODUCER_NAMESPACE",
        "VALIDATION_NAMESPACE",
        "FORMAL_PRODUCER_NAMESPACE",
        "FORMAL_VALIDATION_NAMESPACE",
    ):
        monkeypatch.setattr(
            S,
            name,
            str(dangling if name == namespace_name else tmp_path / name),
        )
    with pytest.raises(Exception, match="symlink"):
        S._reject_namespace_symlinks()


def test_launcher_guards_authority_executables_and_all_namespaces_symlinks() -> None:
    source = (
        ROOT / S.OVERLAY_PATHS["smoke_launcher"]
    ).read_text(encoding="utf-8")
    for variable in (
        "$rc_authority",
        "$rc_launcher",
        "$rc_producer",
        "$rc_validator",
        "$rc_producer_dir",
        "$rc_validation_dir",
    ):
        assert f'test ! -L "{variable}"' in source
    assert source.count("test ! -L") >= 10
    assert source.count("test ! -e") >= 6
