from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch


RC_ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = RC_ROOT / "programs"
SRC = RC_ROOT / "src"
for import_root in (str(PROGRAMS), str(SRC)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

import run_dino_rcde_r1_main_train_v1_5 as v15  # noqa: E402
import run_dino_rcde_r1_main_train_v1_7_contract_repair as repair  # noqa: E402
import validate_dino_rcde_r1_main_train_v1_7_contract_repair as validator  # noqa: E402


ROLES_PATH = RC_ROOT / "protocols/dino_rcde_training_roles_600_v1_2_20260812.json"
DATA_ROOT = (
    RC_ROOT
    / "results/dino_rcde_p0_v1_2/finalize_repair_job5069473/data"
)
LEDGER_PATH = DATA_ROOT / "fold_access_and_episode_receipt.json"
SHAPE_PATH = DATA_ROOT / "resource_shape_ledger.json"


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        json.loads(ROLES_PATH.read_text(encoding="utf-8")),
        json.loads(LEDGER_PATH.read_text(encoding="utf-8")),
        json.loads(SHAPE_PATH.read_text(encoding="utf-8")),
    )


def _shape_query_maps(
    shape: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = [row for row in shape["rows"] if row["kind"] == "query"]
    return (
        {str(row["id"]): row for row in rows},
        {int(row["query_ordinal"]): row for row in rows},
    )


def test_known_job5070402_mapping_is_source_678_to_cache_slot_374(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, ledger, shape = frozen_inputs
    by_id, by_source = _shape_query_maps(shape)
    row = by_id["OUTCOME-0504"]
    assert int(row["query_ordinal"]) == 678
    assert int(row["execution_ordinal"]) == 374
    assert by_source[678] == row

    episodes = [
        item for item in ledger["episodes"] if item["query_id"] == "OUTCOME-0504"
    ]
    assert len(episodes) == 3
    assert {
        (int(item["query_ordinal"]), int(item["execution_ordinal"]))
        for item in episodes
    } == {(678, 374)}

    correct = RC_ROOT / "runtime/dino_rcde_p0_v1_2/stable_fp16_cache/query_0374.pt"
    wrong = RC_ROOT / "runtime/dino_rcde_p0_v1_2/stable_fp16_cache/query_0678.pt"
    assert correct.is_file()
    assert not wrong.exists()


def test_all_1800_episodes_close_query_id_source_and_execution_namespaces(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    roles, ledger, shape = frozen_inputs
    role_rows = roles.get("records", roles.get("rows", []))
    role_by_id = {str(row["query_id"]): row for row in role_rows}
    shape_by_id, _ = _shape_query_maps(shape)
    executions = {
        int(row["execution_ordinal"])
        for row in shape["rows"]
        if row["kind"] == "query"
    }
    references = {
        int(row["physical_row"])
        for row in shape["rows"]
        if row["kind"] == "reference"
    }
    assert len(role_by_id) == len(shape_by_id) == 600
    assert executions == set(range(600))
    assert len(ledger["episodes"]) == 1_800

    triples: set[tuple[str, int, int]] = set()
    for episode in ledger["episodes"]:
        query_id = str(episode["query_id"])
        shape_row = shape_by_id[query_id]
        role = role_by_id[query_id]
        source_ordinal = int(episode["query_ordinal"])
        execution_ordinal = int(episode["execution_ordinal"])
        assert source_ordinal == int(shape_row["query_ordinal"])
        assert source_ordinal == int(role["query_ordinal"])
        assert execution_ordinal == int(shape_row["execution_ordinal"])
        assert 0 <= execution_ordinal < 600
        assert int(episode["target_physical_row"]) in references
        assert all(
            int(item["physical_row"]) in references
            for item in episode["selected_negatives"]
        )
        triples.add((query_id, source_ordinal, execution_ordinal))
    assert len(triples) == 600


def test_input_closure_replays_all_episodes_without_loading_34gb_cache(
    monkeypatch: pytest.MonkeyPatch,
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, _, shape = frozen_inputs
    query_rows = [row for row in shape["rows"] if row["kind"] == "query"]
    reference_rows = [row for row in shape["rows"] if row["kind"] == "reference"]
    synthetic_index = {
        **{
            ("query", int(row["execution_ordinal"])): Path(
                f"query_{int(row['execution_ordinal']):04d}.pt"
            )
            for row in query_rows
        },
        **{
            ("reference", int(row["physical_row"])): Path(
                f"reference_{int(row['physical_row']):05d}.pt"
            )
            for row in reference_rows
        },
    }
    monkeypatch.setattr(repair.base, "cache_index", lambda _: synthetic_index)
    monkeypatch.setattr(repair, "_validate_payload", lambda *args, **kwargs: None)
    protocol = {
        "bindings": {
            "training_roles": {
                "path": str(ROLES_PATH.relative_to(RC_ROOT)),
            },
            "episode_ledger": {
                "path": str(LEDGER_PATH.relative_to(RC_ROOT)),
            },
            "resource_shape_ledger": {
                "path": str(SHAPE_PATH.relative_to(RC_ROOT)),
            },
        },
        "cache": {
            "model_checkpoint_logical_sha256": repair.EXPECTED_CACHE_MODEL_SHA256,
        },
    }
    frozen = repair.input_closure(protocol, Path("unused"), fold=1)
    assert frozen["preflight"]["known_regression_mapping"] == {
        "query_id": "OUTCOME-0504",
        "query_ordinal": 678,
        "execution_ordinal": 374,
    }
    assert frozen["preflight"]["all_episode_mapping_count"] == 1_800
    assert frozen["preflight"]["cache_execution_ordinal_range"] == [0, 599]
    assert frozen["manifest"]["query_cache_key_field"] == "execution_ordinal"
    assert frozen["manifest"]["historical_query_ordinal_model_visible"] is False
    assert frozen["manifest"]["reference_row_intersection_count"] == 0
    assert frozen["manifest"]["allowed_reference_count"] == 37


def test_frozen_pair_directions_are_derived_from_p0_candidate_order(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, ledger, _ = frozen_inputs
    old_direction_disagreements = 0
    pair_count = 0
    for episode in ledger["episodes"]:
        order = [int(item["physical_row"]) for item in episode["model_candidate_order"]]
        positions = {row: index for index, row in enumerate(order)}
        target = int(episode["target_physical_row"])
        rivals = [int(item["physical_row"]) for item in episode["selected_negatives"]]
        expected = [positions[target] < positions[rival] for rival in rivals]
        observed = repair.frozen_pair_directions(episode)
        assert observed == expected
        old = [
            v15.target_first(
                int(episode["heldout_fold"]), str(episode["query_id"]), rival
            )
            for rival in rivals
        ]
        old_direction_disagreements += sum(a != b for a, b in zip(observed, old))
        pair_count += len(rivals)
    assert pair_count == 14_400
    # This catches a regression back to the independent V1.5 direction hash.
    assert old_direction_disagreements == 7_228


def test_frozen_pair_directions_reject_incomplete_or_duplicate_order(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    _, ledger, _ = frozen_inputs
    duplicate = dict(ledger["episodes"][0])
    duplicate["model_candidate_order"] = list(duplicate["model_candidate_order"])
    duplicate["model_candidate_order"][1] = duplicate["model_candidate_order"][0]
    with pytest.raises(repair.ContractRepairAbort, match="not unique"):
        repair.frozen_pair_directions(duplicate)

    missing = dict(ledger["episodes"][0])
    missing["model_candidate_order"] = list(missing["model_candidate_order"][:-1])
    with pytest.raises(repair.ContractRepairAbort, match="not the episode row set"):
        repair.frozen_pair_directions(missing)


def test_query_pool_order_uses_frozen_source_hash_not_query_id(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    roles, ledger, shape = frozen_inputs
    correct, wrong = repair.ordered_pools(ledger, roles, shape, fold=1)
    assert (len(correct), len(wrong)) == (357, 88)
    for pool in (correct, wrong):
        keys = [
            v15.hash_parts(
                repair.QUERY_ORDER_NAMESPACE,
                1,
                row["_source_image_sha256"],
            )
            for row in pool
        ]
        assert keys == sorted(keys)
    assert [row["query_id"] for row in (correct[:2] + wrong[:2])] == [
        "OUTCOME-0070",
        "DIFFICULT-0055",
        "DIFFICULT-0061",
        "OUTCOME-0614",
    ]
    assert repair.QUERY_ORDER_NAMESPACE == "RCDE_QUERY_ORDER_V1_1"


def _tracker_fixture(monkeypatch: pytest.MonkeyPatch) -> repair.CacheAccessTracker:
    query_payload = {
        "source_image_sha256": "q" * 64,
        "model_checkpoint_logical_sha256": "m" * 64,
        "logical_sha256": "query-logical",
    }
    reference_payload = {
        "source_image_sha256": "r" * 64,
        "model_checkpoint_logical_sha256": "m" * 64,
        "logical_sha256": "reference-logical",
    }
    payloads = {
        "query_0001.pt": query_payload,
        "reference_00002.pt": reference_payload,
    }
    monkeypatch.setattr(
        repair.torch,
        "load",
        lambda path, **_: payloads[Path(path).name],
    )
    monkeypatch.setattr(
        repair.cache_base,
        "_cache_logical_hash",
        lambda payload: payload["logical_sha256"],
    )
    return repair.CacheAccessTracker(
        index={
            ("query", 1): Path("query_0001.pt"),
            ("reference", 2): Path("reference_00002.pt"),
        },
        query_by_execution={1: {"source_image_sha256": "q" * 64}},
        reference_by_row={2: {"source_image_sha256": "r" * 64}},
        allowed_queries=[1],
        allowed_references=[2],
        model_sha256="m" * 64,
    )


def test_cache_access_tracker_enforces_allowlist_and_opens_each_payload_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = _tracker_fixture(monkeypatch)
    first = tracker.load("query", 1)
    second = tracker.load("query", 1)
    tracker.load("reference", 2)
    assert first is second
    assert tracker.semantic_counts[("query", 1)] == 2
    assert tracker.file_open_counts[("query", 1)] == 1
    assert tracker.semantic_counts[("reference", 2)] == 1
    assert tracker.file_open_counts[("reference", 2)] == 1

    with pytest.raises(repair.ContractRepairAbort, match="outside eligible"):
        tracker.load("query", 678)
    with pytest.raises(repair.ContractRepairAbort, match="outside signed allowlist"):
        tracker.load("reference", 3)
    with pytest.raises(repair.ContractRepairAbort, match="unknown cache"):
        tracker.load("opened", 1)

    ledger = tracker.ledger(job_id="17", start_update=0, completed_updates=1)
    assert ledger["query_rows"] == [
        {
            "ordinal": 1,
            "semantic_read_count": 2,
            "file_open_count": 1,
            "cache_file": "query_0001.pt",
        }
    ]
    assert ledger["reference_rows"] == [
        {
            "ordinal": 2,
            "semantic_read_count": 1,
            "file_open_count": 1,
            "cache_file": "reference_00002.pt",
        }
    ]


def test_protected_access_schema_is_exact_nonempty_and_all_zero() -> None:
    observed = repair.protected_zeros()
    assert tuple(observed) == repair.PROTECTED_ZERO_KEYS
    assert set(observed) == {
        "C8_runtime_read_count",
        "S8_runtime_read_count",
        "opened_runtime_read_count",
        "sealed_runtime_read_count",
        "unauthorized_natural_result_read_count",
        "home_files_modified",
    }
    assert observed and all(value == 0 for value in observed.values())


def _write_chunk(
    root: Path,
    *,
    job_id: str,
    start: int,
    completed: int,
    status: str = "DINO_RCDE_R1_MAIN_DEV_CHUNK_COMPLETE",
    prior_job_id: str | None = None,
) -> Path:
    chunk_dir = root / f"chunk_job{job_id}"
    chunk_dir.mkdir()
    bound_files = {}
    for role, filename in (
        ("resume_state", "resume_state.pt"),
        ("runtime_read_ledger", "runtime_read_ledger.json"),
        ("input_preflight", "input_preflight.json"),
    ):
        path = chunk_dir / filename
        path.write_bytes(f"{role}:{job_id}:{start}:{completed}".encode("ascii"))
        bound_files[role] = (
            path.relative_to(root).as_posix(),
            v15.file_sha256(path),
        )
    (chunk_dir / "chunk.json").write_text(
        json.dumps(
            {
                "status": status,
                "job_id": job_id,
                "start_update": start,
                "completed_updates": completed,
                "restored_from_resume": prior_job_id is not None,
                "restored_resume_state": (
                    f"chunk_job{prior_job_id}/resume_state.pt"
                    if prior_job_id is not None
                    else None
                ),
                "restored_resume_state_sha256": (
                    v15.file_sha256(
                        root / f"chunk_job{prior_job_id}/resume_state.pt"
                    )
                    if prior_job_id is not None
                    else None
                ),
                "resume_state": bound_files["resume_state"][0],
                "resume_state_sha256": bound_files["resume_state"][1],
                "runtime_read_ledger": bound_files["runtime_read_ledger"][0],
                "runtime_read_ledger_sha256": bound_files["runtime_read_ledger"][1],
                "input_preflight": bound_files["input_preflight"][0],
                "input_preflight_sha256": bound_files["input_preflight"][1],
            }
        ),
        encoding="utf-8",
    )
    return chunk_dir


def test_chunk_history_requires_exact_contiguous_positive_coverage(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    _write_chunk(valid, job_id="10", start=0, completed=128)
    _write_chunk(valid, job_id="11", start=128, completed=512, prior_job_id="10")
    rows = repair._chunk_history(valid)
    assert [(row["start_update"], row["completed_updates"]) for row in rows] == [
        (0, 128),
        (128, 512),
    ]

    gap = tmp_path / "gap"
    gap.mkdir()
    _write_chunk(gap, job_id="20", start=0, completed=128)
    _write_chunk(gap, job_id="21", start=129, completed=512, prior_job_id="20")
    with pytest.raises(repair.ContractRepairAbort, match="non-contiguous"):
        repair._chunk_history(gap)

    backwards = tmp_path / "backwards"
    backwards.mkdir()
    _write_chunk(backwards, job_id="30", start=0, completed=128)
    _write_chunk(backwards, job_id="31", start=128, completed=128, prior_job_id="30")
    with pytest.raises(repair.ContractRepairAbort, match="non-contiguous"):
        repair._chunk_history(backwards)

    bad_status = tmp_path / "status"
    bad_status.mkdir()
    _write_chunk(
        bad_status,
        job_id="40",
        start=0,
        completed=128,
        status="PARTIAL_OR_FAILED",
    )
    with pytest.raises(repair.ContractRepairAbort, match="status drift"):
        repair._chunk_history(bad_status)


def test_atomic_chunk_is_invisible_before_directory_rename(tmp_path: Path) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    staging = tmp_path / ".formal.chunk_job50.staging.123"
    staging.mkdir()
    final_name = "chunk_job50"
    bound_files: dict[str, tuple[str, str]] = {}
    for role, filename in (
        ("resume_state", "resume_state.pt"),
        ("runtime_read_ledger", "runtime_read_ledger.json"),
        ("input_preflight", "input_preflight.json"),
    ):
        path = staging / filename
        path.write_bytes(f"{role}:50:0:128".encode("ascii"))
        bound_files[role] = (
            f"{final_name}/{filename}",
            v15.file_sha256(path),
        )
    (staging / "chunk.json").write_text(
        json.dumps(
            {
                "status": "DINO_RCDE_R1_MAIN_DEV_CHUNK_COMPLETE",
                "job_id": "50",
                "start_update": 0,
                "completed_updates": 128,
                "restored_from_resume": False,
                "restored_resume_state": None,
                "restored_resume_state_sha256": None,
                "resume_state": bound_files["resume_state"][0],
                "resume_state_sha256": bound_files["resume_state"][1],
                "runtime_read_ledger": bound_files["runtime_read_ledger"][0],
                "runtime_read_ledger_sha256": bound_files[
                    "runtime_read_ledger"
                ][1],
                "input_preflight": bound_files["input_preflight"][0],
                "input_preflight_sha256": bound_files["input_preflight"][1],
            }
        ),
        encoding="utf-8",
    )

    # A crash at this point leaves only a private sibling staging directory.
    assert repair._chunk_history(output) == []
    os.rename(staging, output / final_name)
    rows = repair._chunk_history(output)
    assert len(rows) == 1
    assert rows[0]["completed_updates"] == 128


def test_final_committed_state_can_finalize_twice_without_new_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "formal"
    output.mkdir()
    identity = output / "run_identity.json"
    identity.write_text("{}\n", encoding="utf-8")
    manifest_path = output / "signed_reference_access_manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")

    model = repair.DINO_RCDE_V1_2()
    initial_sha = v15.state_dict_sha256(model)
    state = {
        "schema_version": "rc_dino_rcde_r1_main_resume_state_v1_7",
        "arm": "RCDE_BAG",
        "outer_fold": 1,
        "completed_updates": v15.UPDATE_COUNT,
        "protocol_sha256": "p" * 64,
        "authority_sha256": "a" * 64,
        "initial_state_sha256": initial_sha,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "loss_trace": [],
        "pair_count": 0,
        "python_random_state": None,
        "numpy_random_state": None,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": [],
    }
    chunk = _write_chunk(output, job_id="60", start=0, completed=v15.UPDATE_COUNT)
    torch.save(state, chunk / "resume_state.pt")
    row = json.loads((chunk / "chunk.json").read_text(encoding="utf-8"))
    row["training_complete"] = True
    row["resume_state_sha256"] = v15.file_sha256(chunk / "resume_state.pt")
    (chunk / "runtime_read_ledger.json").write_text(
        json.dumps({"query_rows": [], "reference_rows": []}) + "\n",
        encoding="utf-8",
    )
    row["runtime_read_ledger_sha256"] = v15.file_sha256(
        chunk / "runtime_read_ledger.json"
    )
    (chunk / "chunk.json").write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        repair,
        "schedule_prefix",
        lambda *args, **kwargs: __import__("hashlib").sha256(b"schedule"),
    )
    kwargs = {
        "output_dir": output,
        "protocol": {"execution": {"maximum_jobs_submitted_now": 4}},
        "protocol_sha256": "p" * 64,
        "authority_sha256": "a" * 64,
        "identity_path": identity,
        "manifest_path": manifest_path,
        "manifest": {
            "allowed_reference_physical_rows": [],
            "heldout_reference_physical_rows": [],
        },
        "correct": [],
        "wrong": [],
        "initial_state_sha256": initial_sha,
    }
    repair.finalize_training(**kwargs)
    before = sorted(path.name for path in output.iterdir() if path.name.startswith("chunk_job"))
    # Model a crash after checkpoint but before result publication.
    (output / "train_result.json").unlink()
    repair.finalize_training(**kwargs)
    after = sorted(path.name for path in output.iterdir() if path.name.startswith("chunk_job"))
    assert before == after == ["chunk_job60"]
    assert (output / "checkpoint_update2048.pt").is_file()
    assert (output / "train_result.json").is_file()


def test_validator_replays_seed17_initial_state_after_incidental_rng_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RNGDependentModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(3))

    monkeypatch.setattr(validator, "DINO_RCDE_V1_2", RNGDependentModel)
    monkeypatch.setattr(validator, "EXPECTED_PARAMETERS", 3)
    output = tmp_path / "validated"
    chunk = output / "chunk_job70"
    chunk.mkdir(parents=True)
    checkpoint_path = output / "checkpoint_update2048.pt"
    resume_path = chunk / "resume_state.pt"
    protocol_sha = "p" * 64
    authority_sha = "a" * 64

    torch.manual_seed(validator.SEED)
    initial_model = RNGDependentModel()
    initial_state = {
        name: tensor.detach().clone()
        for name, tensor in initial_model.state_dict().items()
    }
    initial_sha = validator.state_dict_sha256(initial_state)
    final_state = {
        name: tensor.detach().clone() + 1.0
        for name, tensor in initial_state.items()
    }
    final_sha = validator.state_dict_sha256(final_state)
    checkpoint = {
        "schema_version": "rc_dino_rcde_r1_main_checkpoint_v1_7",
        "arm": "RCDE_BAG",
        "outer_fold": 1,
        "update": validator.UPDATES,
        "seed": validator.SEED,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": final_sha,
        "model_state_dict": final_state,
    }
    torch.save(checkpoint, checkpoint_path)
    optimizer = {
        "state": {
            0: {
                "step": torch.tensor(float(validator.UPDATES)),
                "exp_avg": torch.zeros(3),
                "exp_avg_sq": torch.zeros(3),
            }
        },
        "param_groups": [
            {
                "params": [0],
                "lr": 3.0e-5,
                "weight_decay": 1.0e-4,
            }
        ],
    }
    resume = {
        "schema_version": "rc_dino_rcde_r1_main_resume_state_v1_7",
        "arm": "RCDE_BAG",
        "outer_fold": 1,
        "completed_updates": validator.UPDATES,
        "protocol_sha256": protocol_sha,
        "authority_sha256": authority_sha,
        "initial_state_sha256": initial_sha,
        "model_state_dict": final_state,
        "optimizer_state_dict": optimizer,
        "loss_trace": [],
        "pair_count": 0,
        "python_random_state": None,
        "numpy_random_state": validator.np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": [],
    }
    torch.save(resume, resume_path)
    result = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": validator.file_sha256(checkpoint_path),
        "completed_resume_state": resume_path.relative_to(output).as_posix(),
        "completed_resume_state_sha256": validator.file_sha256(resume_path),
        "initial_state_sha256": initial_sha,
        "final_state_sha256": final_sha,
    }
    final_chunk = {
        "resume_state": resume_path.relative_to(output).as_posix(),
        "resume_state_sha256": validator.file_sha256(resume_path),
    }

    # Move the validator RNG far away from seed 17 immediately before replay.
    torch.manual_seed(991_337)
    torch.rand(10_000)
    observed_checkpoint, observed_resume = validator.validate_checkpoint_and_resume(
        output,
        result,
        protocol_sha,
        authority_sha,
        [],
        0,
        final_chunk,
    )
    assert observed_checkpoint == checkpoint_path
    assert observed_resume == resume_path


def test_schedule_event_binds_execution_slot_source_hash_and_frozen_order(
    frozen_inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    roles, ledger, shape = frozen_inputs
    correct, wrong = repair.ordered_pools(ledger, roles, shape, fold=1)
    episodes = v15.update_episodes(correct, wrong, 0)
    event = repair.schedule_event(1, 1, episodes)
    assert event["outer_fold"] == 1
    assert event["update"] == 1
    assert [row["query_id"] for row in event["queries"]] == [
        "OUTCOME-0070",
        "DIFFICULT-0055",
        "DIFFICULT-0061",
        "OUTCOME-0614",
    ]
    for episode, row in zip(episodes, event["queries"], strict=True):
        assert row["execution_ordinal"] == int(episode["execution_ordinal"])
        assert row["query_ordinal"] == int(episode["query_ordinal"])
        assert row["source_image_sha256"] == episode["_source_image_sha256"]
        assert row["target_first"] == repair.frozen_pair_directions(episode)
        expected_order = [
            int(item["physical_row"]) for item in episode["model_candidate_order"]
        ]
        assert row["frozen_model_candidate_rows"] == expected_order
        assert row["frozen_model_candidate_rows_sha256"] == repair.canonical_sha256(
            expected_order
        )
