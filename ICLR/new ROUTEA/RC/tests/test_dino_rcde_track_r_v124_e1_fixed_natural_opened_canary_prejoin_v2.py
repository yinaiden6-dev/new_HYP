from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2 as producer  # noqa: E402
import validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v2 as independent  # noqa: E402
import freeze_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v2 as review_v2  # noqa: E402
from rc_aslo_xf.dino_rcde_sr0_mt_p_lock_v2 import canonical_sha256 as p_sha  # noqa: E402
from rc_aslo_xf.dino_rcde_track_r_c_col_p_phase_a_schema_v1 import candidate_logical_sha256  # noqa: E402
from rc_aslo_xf.dino_rcde_track_r_v124_e1_atomic_family_v2 import (  # noqa: E402
    CRASH_BOUNDARIES,
    E1AtomicFamilyV2Error,
    InjectedFamilyCrash,
    logical_sha256,
    publish_producer_family,
    publish_validation_family,
    validate_family,
)
from rc_aslo_xf import dino_rcde_track_r_v124_e1_authority_v2 as authority_v2  # noqa: E402


PHASE_A = ROOT / "results/dino_rcde_track_r_v124_c_col_p_phase_a_retry_v2/final/phase_a_artifact.pt"
GEOMETRY = ROOT / "cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"
REVIEW_V1 = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.json"
REJECTION_V1 = ROOT / "registry/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_review_authority_v1_20260826.terminal_rejection_v1.json"


@pytest.fixture(scope="module")
def phase_a():
    return torch.load(
        PHASE_A, map_location="cpu", weights_only=False, mmap=True
    )


def _json_artifact() -> dict:
    value = {
        "schema_version": "test_artifact",
        "status": "READY",
        "tensor": torch.arange(4),
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = authority_v2.canonical_sha256(
        {
            "schema_version": value["schema_version"],
            "status": value["status"],
            "tensor_sha256": "test",
            "scientific_GO_or_NO_GO": None,
            "automatic_stage_advance": False,
            "next_authorized_stage": None,
        }
    )
    return value


def _result_builder(_artifact_path: Path) -> dict:
    result = {
        "schema_version": "test_result",
        "status": "READY",
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    result["logical_sha256"] = logical_sha256(result)
    return result


def _authority_fixture() -> dict:
    bindings = {
        key: {"path": "dummy", "bytes": 0, "sha256": "0" * 64}
        for key in authority_v2.required_bindings()
    }
    value = {
        "schema_version": authority_v2.SCHEMA,
        "status": authority_v2.STATUS,
        "bindings": bindings,
        "independent_review_status": "PASS",
        "execution_ordinal": 0,
        "query_id": "DIFFICULT-0000",
        "source_fold": 2,
        "candidate_count": 128,
        "candidate_axis_sha256": authority_v2.CANDIDATE_AXIS_SHA256,
        "pair_sha256": authority_v2.PAIR_SHA256,
        "v_checkpoint_file_sha256": authority_v2.V_CHECKPOINT_FILE_SHA256,
        "v_checkpoint_state_sha256": authority_v2.V_CHECKPOINT_STATE_SHA256,
        "family_sequence": list(authority_v2.FAMILIES),
        "arm_sequence": list(authority_v2.ARMS),
        "expected_pair_arm_record_count": 9,
        "expected_directional_term_count": 36,
        "device_schedule": {
            "all_patch_device": "cpu",
            "regional_device": "cuda:0",
            "model_load_count": 1,
            "model_device_transition_count": 1,
        },
        "resource_contract": {
            "partition": "accelerated",
            "allowed_partitions": ["accelerated"],
            "gpu_count": 1,
            "cpus_per_task": 8,
            "memory_megabytes": 128000,
            "walltime_seconds": 7200,
        },
        "model_load_authorized": True,
        "model_forward_authorized": True,
        "p_training_authorized": False,
        "v_training_authorized": False,
        "model_backward_authorized": False,
        "model_update_authorized": False,
        "target_rival_join_authorized": False,
        "postjoin_authorized": False,
        "scientific_reduction_authorized": False,
        "automatic_submit_authorized": False,
        "scientific_GO_or_NO_GO": None,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
        "import_closure": list(authority_v2.discover_import_closure()),
        "import_closure_sha256": authority_v2.canonical_sha256(
            list(authority_v2.discover_import_closure())
        ),
    }
    value["logical_sha256"] = authority_v2.logical_sha256(value)
    return value


def _mutate_record(
    artifact: dict, *, field: str, value: object
) -> dict:
    changed = dict(artifact)
    wrappers = list(artifact["control_candidates"])
    wrapper = dict(wrappers[0])
    records = dict(wrapper["direction_records"])
    record = copy.deepcopy(records["a_to_b"])
    if field == "fold":
        record["query"]["outer_fold"] = value
    elif field == "checkpoint":
        record["p_checkpoint_sha256"] = value
    else:
        raise AssertionError(field)
    unsigned = {key: item for key, item in record.items() if key != "record_sha256"}
    record["record_sha256"] = p_sha(unsigned)
    records["a_to_b"] = record
    wrapper["direction_records"] = records
    sequence = list(wrapper["direction_record_sha256_sequence"])
    sequence[0] = record["record_sha256"]
    wrapper["direction_record_sha256_sequence"] = sequence
    wrapper["logical_sha256"] = candidate_logical_sha256(wrapper)
    wrappers[0] = wrapper
    changed["control_candidates"] = wrappers
    global_sequence = list(artifact["control_record_sha256_sequence"])
    global_sequence[0] = record["record_sha256"]
    changed["control_record_sha256_sequence"] = global_sequence
    return changed


def test_review_v1_is_immutable_and_terminally_rejected() -> None:
    review = json.loads(REVIEW_V1.read_text(encoding="ascii"))
    rejection = json.loads(REJECTION_V1.read_text(encoding="ascii"))
    assert REVIEW_V1.stat().st_mode & 0o777 == 0o444
    assert rejection["source_review_authority_sha256"] == (
        "0476e159c039d5bfb685e392984bc61d5607ce09a68ebf7df931b186d8e1f4e7"
    )
    assert rejection["terminal_review_decision"] == "FAIL"
    assert rejection["execution_authorized"] is False
    assert rejection["smoke_authorized"] is False
    assert rejection["submission_authorized"] is False
    assert rejection["logical_sha256"] == logical_sha256(rejection)
    assert review["execution_authorized"] is False


def test_review_v2_candidate_builds_but_remains_unfrozen_nonexecuting() -> None:
    value = review_v2.build_authority()
    assert value["status"] == review_v2.STATUS
    assert value["terminal_independent_review_status"] == "PENDING"
    assert value["execution_authorized"] is False
    assert value["smoke_authorized"] is False
    assert value["submission_authorized"] is False
    assert value["scientific_GO_or_NO_GO"] is None
    assert value["next_authorized_stage"] is None
    assert value["logical_sha256"] == review_v2.logical_sha256(value)
    assert not review_v2.OUTPUT.exists()


def test_deep_validates_all_128_wrappers_and_256_records(phase_a) -> None:
    receipt = producer.deep_validate_c_col_population(
        phase_a, canonical_geometry_path=GEOMETRY
    )
    replay = independent.deep_validate_c_col_independent(
        phase_a, canonical_geometry_path=GEOMETRY
    )
    assert receipt["wrapper_count"] == replay["wrapper_count"] == 128
    assert receipt["record_count"] == replay["record_count"] == 256
    assert receipt["record_sequence_sha256"] == replay[
        "record_sequence_sha256"
    ]


@pytest.mark.parametrize("kind", ["loss", "duplicate", "reorder"])
def test_128_256_loss_duplicate_reorder_poison(phase_a, kind: str) -> None:
    changed = dict(phase_a)
    wrappers = list(phase_a["control_candidates"])
    if kind == "loss":
        wrappers = wrappers[:-1]
    elif kind == "duplicate":
        wrappers[1] = wrappers[0]
    else:
        wrappers[0], wrappers[1] = wrappers[1], wrappers[0]
    changed["control_candidates"] = wrappers
    with pytest.raises(producer.E1V2ProducerError):
        producer.deep_validate_c_col_population(
            changed, canonical_geometry_path=GEOMETRY
        )


@pytest.mark.parametrize(
    ("field", "bad"),
    [("fold", 3), ("checkpoint", "f" * 64)],
)
def test_wrong_record_fold_and_checkpoint_poison(
    phase_a, field: str, bad: object
) -> None:
    changed = _mutate_record(phase_a, field=field, value=bad)
    with pytest.raises(producer.E1V2ProducerError):
        producer.deep_validate_c_col_population(
            changed, canonical_geometry_path=GEOMETRY
        )


def test_authority_exact_whitelist_rejects_arbitrary_binding() -> None:
    value = _authority_fixture()
    authority_v2.validate_authority_envelope(value)
    value = copy.deepcopy(value)
    value["bindings"]["target_roles"] = {
        "path": "valid.json",
        "bytes": 1,
        "sha256": "a" * 64,
    }
    value["logical_sha256"] = authority_v2.logical_sha256(value)
    with pytest.raises(authority_v2.E1AuthorityV2Error, match="whitelist"):
        authority_v2.validate_authority_envelope(value)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("source_fold", 3),
        ("v_checkpoint_file_sha256", "e" * 64),
        ("v_checkpoint_state_sha256", "d" * 64),
    ],
)
def test_authority_wrong_fold_checkpoint_poison(field: str, bad: object) -> None:
    value = _authority_fixture()
    value[field] = bad
    value["logical_sha256"] = authority_v2.logical_sha256(value)
    with pytest.raises(authority_v2.E1AuthorityV2Error):
        authority_v2.validate_authority_envelope(value)


def test_import_closure_contains_terminal_modules() -> None:
    closure = set(authority_v2.discover_import_closure())
    required = {
        "src/rc_aslo_xf/__init__.py",
        "src/rc_aslo_xf/dino_rcde_cw1_multitile_superregion_v2.py",
        "src/rc_aslo_xf/dino_rcde_cw1_multitile_vdecode_v1.py",
        "src/rc_aslo_xf/dino_rcde_sr0_mt_controls_v1.py",
        "src/rc_aslo_xf/dino_rcde_sr0_mt_p_lock_v2.py",
        "src/rc_aslo_xf/dino_rcde_sr0_mt_p_natural_adapter_v2.py",
        "src/rc_aslo_xf/dino_rcde_sr0_mt_p_runtime_v1.py",
        "programs/run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2.py",
        "programs/validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v2.py",
    }
    assert required.issubset(closure)
    assert len(authority_v2.required_bindings()) == len(
        authority_v2.BASE_BINDINGS
    ) + len(closure)


@pytest.mark.parametrize("boundary", CRASH_BOUNDARIES)
def test_producer_crash_boundaries_publish_no_partial_or_complete_family(
    tmp_path: Path, boundary: str
) -> None:
    public = tmp_path / f"family-{boundary.lower()}"
    with pytest.raises(InjectedFamilyCrash, match=boundary):
        publish_producer_family(
            public,
            authority_sha256="a" * 64,
            artifact=_json_artifact(),
            result_builder=_result_builder,
            crash_after=boundary,
        )
    if boundary == "AFTER_ATOMIC_RENAME":
        assert public.is_dir()
        validate_family(
            public,
            authority_sha256="a" * 64,
            family_role="E1_V2_PRODUCER_FAMILY",
            producer=True,
        )
    else:
        assert not public.exists()
        assert list(tmp_path.glob(f".{public.name}.stage-*"))


def test_exact_committed_reuse_skips_builder(tmp_path: Path) -> None:
    public = tmp_path / "committed"
    publish_producer_family(
        public,
        authority_sha256="b" * 64,
        artifact=_json_artifact(),
        result_builder=_result_builder,
    )

    def forbidden(_path: Path):
        raise AssertionError("result builder ran during reuse")

    _, result, reused = publish_producer_family(
        public,
        authority_sha256="b" * 64,
        artifact=_json_artifact(),
        result_builder=forbidden,
    )
    assert reused is True
    assert result["status"] == "READY"


def test_committed_tamper_and_extra_member_fail_closed(tmp_path: Path) -> None:
    public = tmp_path / "tamper"
    publish_producer_family(
        public,
        authority_sha256="c" * 64,
        artifact=_json_artifact(),
        result_builder=_result_builder,
    )
    result = public / "producer_result.json"
    result.chmod(0o644)
    result.write_text("{}\n", encoding="ascii")
    result.chmod(0o444)
    with pytest.raises(E1AtomicFamilyV2Error):
        validate_family(
            public,
            authority_sha256="c" * 64,
            family_role="E1_V2_PRODUCER_FAMILY",
            producer=True,
        )

    public2 = tmp_path / "extra"
    publish_producer_family(
        public2,
        authority_sha256="d" * 64,
        artifact=_json_artifact(),
        result_builder=_result_builder,
    )
    (public2 / "foreign.json").write_text("{}", encoding="ascii")
    with pytest.raises(E1AtomicFamilyV2Error, match="member set"):
        validate_family(
            public2,
            authority_sha256="d" * 64,
            family_role="E1_V2_PRODUCER_FAMILY",
            producer=True,
        )


@pytest.mark.parametrize(
    "boundary",
    (
        "AFTER_RESULT_PAYLOAD",
        "AFTER_RESULT_RECEIPT",
        "AFTER_RESULT_COMMIT",
        "AFTER_FAMILY_COMMIT",
        "AFTER_ATOMIC_RENAME",
    ),
)
def test_validation_family_crash_boundaries(
    tmp_path: Path, boundary: str
) -> None:
    public = tmp_path / f"validation-{boundary.lower()}"
    result = _result_builder(Path("unused"))
    with pytest.raises(InjectedFamilyCrash):
        publish_validation_family(
            public,
            authority_sha256="e" * 64,
            result=result,
            crash_after=boundary,
        )
    assert public.is_dir() if boundary == "AFTER_ATOMIC_RENAME" else not public.exists()


def test_v2_launcher_is_accelerated_only_and_no_submit() -> None:
    launcher = (
        ROOT
        / "slurm/dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=accelerated" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "dev_accelerated-h100" not in launcher
    assert "dev_cpuonly" not in launcher
    assert "\nsbatch " not in launcher
    assert "--dependency" not in launcher


def test_independent_v2_does_not_import_producer_v2() -> None:
    source = (
        ROOT
        / "programs/validate_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_independent_v2.py"
    ).read_text(encoding="utf-8")
    assert (
        "import run_dino_rcde_track_r_v124_e1_fixed_natural_opened_canary_prejoin_v2"
        not in source
    )
    assert "deep_validate_c_col_population" not in source
