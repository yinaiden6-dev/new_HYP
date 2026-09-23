from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "programs"))

import rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v2 as runtime_v2  # noqa: E402
import rc_aslo_xf.dino_rcde_track_r_control_decoder_v2 as decoder_v2  # noqa: E402
from materialize_dino_rcde_sr0_mt_p_features_v1 import (  # noqa: E402
    synthetic_source,
)
from rc_aslo_xf.dino_rcde_cw1_multitile_superregion_v2 import (  # noqa: E402
    ARM_ALL_PATCH,
)
from rc_aslo_xf.dino_rcde_sr0_mt_p_runtime_v1 import (  # noqa: E402
    initialize_training,
)
from rc_aslo_xf.dino_rcde_sr0_mt_v_runtime_v1 import (  # noqa: E402
    CandidatePLockV1,
    VQueryBundle,
    canonical_candidate_axis,
)
from rc_aslo_xf.dino_rcde_track_r_phase_b_all_patch_v1 import (  # noqa: E402
    RECEIPT_FIELDS,
    phase_b_all_patch_invariance,
)
from test_dino_rcde_sr0_mt_c_col_p_v1 import _real_bundle  # noqa: E402
from test_dino_rcde_sr0_mt_three_arm_v1 import _SharedSyntheticV  # noqa: E402
from test_dino_rcde_sr0_mt_three_arm_v1 import _locks  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _CountingV(_SharedSyntheticV):
    def __init__(self) -> None:
        super().__init__()
        self.decode_candidate_count = 0

    def decode_candidate(self, *args, **kwargs):
        self.decode_candidate_count += 1
        return super().decode_candidate(*args, **kwargs)


def test_phase_b_runs_only_all_patch_and_closes_real_c_col_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synthetic_source(query_count=1, candidate_count=128)
    p_model, _ = initialize_training()
    real = _real_bundle(source, p_model, candidate_count=128)

    # Synthetic C_COL changes every P-lock receipt while retaining the exact
    # query and all 128 DINO candidates.  The Phase-B path must not read these
    # changed receipts.
    control_locks = {}
    for position, key in enumerate(canonical_candidate_axis(real.locks)):
        lock = real.locks[key]
        directions = {
            direction: replace(
                sealed,
                p_lock_record_sha256=_sha(
                    f"synthetic-c-col-p-lock-{position}-{direction}"
                ),
            )
            for direction, sealed in lock.direction_locks.items()
        }
        control_locks[key] = CandidatePLockV1(
            candidate=lock.candidate,
            direction_locks=directions,
        )
    control = VQueryBundle(
        query_id=real.query_id,
        execution_ordinal=real.execution_ordinal,
        outer_fold=real.outer_fold,
        query=real.query,
        locks=control_locks,
        candidate_axis_sha256=real.candidate_axis_sha256,
    )
    pair = canonical_candidate_axis(real.locks)[:2]

    calls: list[str] = []
    direct = runtime_v2.decode_direct_arm_term

    def all_patch_only(*args, **kwargs):
        arm_name = kwargs["arm_name"]
        if arm_name != ARM_ALL_PATCH:
            raise AssertionError("FULL/LOCAL direct term was called")
        calls.append(arm_name)
        return direct(*args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("three-arm or C_DINO decoder was called")

    monkeypatch.setattr(runtime_v2, "decode_direct_arm_term", all_patch_only)
    monkeypatch.setattr(decoder_v2, "decode_pair_v2", forbidden)
    monkeypatch.setattr(decoder_v2, "decode_c_dino_pair_v2", forbidden)

    model = _CountingV().eval()
    model_sha = runtime_v2.state_dict_sha256(model)
    receipt = phase_b_all_patch_invariance(
        model=model,
        real_bundle=real,
        control_bundle=control,
        pair_member_keys=pair,
        expected_model_checkpoint_sha256=model_sha,
    )
    assert set(receipt) == RECEIPT_FIELDS
    assert receipt["real_request_sha256"] == receipt["control_request_sha256"]
    assert receipt["real_patch_tensor_sha256"] == receipt[
        "control_patch_tensor_sha256"
    ]
    assert receipt["real_direction_sequence_sha256"] == receipt[
        "control_direction_sequence_sha256"
    ]
    assert receipt["real_pair_score_sha256"] == receipt[
        "control_pair_score_sha256"
    ]
    assert calls == [ARM_ALL_PATCH] * 8
    assert model.decode_candidate_count == 16


def test_single_arm_decoder_cannot_read_p_or_regional_fields() -> None:
    query, locks = _locks()
    left, right = tuple(locks)

    class PRegionalPoison:
        def __init__(self, candidate) -> None:
            self.candidate = candidate

        @property
        def direction_locks(self):
            raise AssertionError("ALL_PATCH read a P/regional field")

    poison = {
        key: PRegionalPoison(lock.candidate) for key, lock in locks.items()
    }
    arm = decoder_v2.decode_all_patch_pair_v2(
        _SharedSyntheticV().eval(),
        query,
        poison,  # type: ignore[arg-type]
        left,
        right,
    )
    assert arm.arm_name == ARM_ALL_PATCH
    assert all(
        term.decoded_root_ordinals == (-1,)
        for term in (*arm.forward_by_direction, *arm.reverse_by_direction)
    )
