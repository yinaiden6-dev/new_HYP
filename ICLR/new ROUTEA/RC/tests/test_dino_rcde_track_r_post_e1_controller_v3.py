from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch"
FREEZER = (
    ROOT
    / "programs/freeze_dino_rcde_track_r_relative_e1_episode_reduction_authority_v119_r2.py"
)


def test_v3_is_new_fail_closed_controller_and_v2_is_untouched() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    v2 = ROOT / "slurm/dino_rcde_track_r_post_e1_controller_v2.sbatch"
    assert hashlib.sha256(v2.read_bytes()).hexdigest() == (
        "82f3ec11ea208be0716d04b8144b5fd1940faccf0e1548fe0df934bb37fa9336"
    )
    for literal in (
        "#SBATCH --job-name=rcde_tr_c1v3",
        "#SBATCH --partition=cpuonly",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=65536",
        "#SBATCH --time=01:00:00",
        'cmp -s "$0" "$rc_controller"',
        'case "${SLURM_JOB_ID:?}" in (*[!0-9]*|\'\') exit 72 ;; esac',
        'test ! -e "$rc_absent"',
        'test ! -L "$rc_absent"',
        "current_authority_v118_20260821.json",
        "DINO_RCDE_TRACK_R_RELATIVE_E1_PARALLEL_AUTHORITY_V118_REVISION2_INDEPENDENT_VALIDATION_PASS",
        "for ordinal in range(50):",
        "eligible == 594 and excluded == 6 and episodes == 1782",
        'sorted(producer_root.glob("*.pt")) == expected_producers',
        "DINO_RCDE_TRACK_R_RELATIVE_E1_EPISODE_LEDGERS_INDEPENDENT_VALIDATION_PASS",
        'b.get("path")=="slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch"',
        'b.get("sha256")==sha(cp)',
        'a.get("authority_revision")==2',
        "DINO_RCDE_TRACK_R_LEDGER_RUNTIME_EPISODE_ID_V2",
        's.get("failure_job_id")==5097842',
        "DINO_RCDE_TRACK_R_RELATIVE_V_FIT_AUTHORIZED",
        'sbatch --parsable "$rc_vfit_launcher"',
        'sbatch --parsable --dependency="afterok:$rc_vfit_job" "$rc_post_vfit_controller"',
        "TRACK_R_V120_VFIT_ARRAY_AND_POST_VFIT_CONTROLLER_SUBMITTED",
        "ENGINEERING_SCHEDULER_CONTINUATION_RECEIPT_ONLY",
        '  "$rc_receipt"',
        'test "$(stat -c \'%a\' "$rc_receipt")" = 444',
    ):
        assert literal in source
    subprocess.run(["bash", "-n", str(CONTROLLER)], check=True)


def test_v3_orders_validation_and_outputs_before_child_submission() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    absent = source.index("# A rerun may never adopt")
    preflight = source.index("# Independently close V118-r2")
    freeze_v119 = source.index('"$rc_python" "$rc_reduce_freezer"')
    reduce_ledgers = source.index('"$rc_python" "$rc_reducer"')
    validate_ledgers = source.index('"$rc_python" "$rc_ledger_validator"')
    freeze_v120 = source.index('"$rc_python" "$rc_vfit_freezer"')
    submit_vfit = source.index("rc_vfit_job=$(sbatch")
    submit_controller = source.index("rc_post_vfit_job=$(sbatch")
    write_receipt = source.index('"$rc_python" - "$rc_receipt"')
    assert (
        absent
        < preflight
        < freeze_v119
        < reduce_ledgers
        < validate_ledgers
        < freeze_v120
        < submit_vfit
        < submit_controller
        < write_receipt
    )
    assert 'mkdir -p "$(dirname "$rc_ledger_validation")"' in source
    assert 'case "$rc_vfit_job" in (*[!0-9]*|\'\') exit 73 ;; esac' in source
    assert 'case "$rc_post_vfit_job" in (*[!0-9]*|\'\') exit 74 ;; esac' in source


def test_v3_embedded_python_and_v119_binding_are_frozen() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(blocks) == 2
    for index, block in enumerate(blocks):
        compile(block, f"controller_v3_embedded_{index}.py", "exec")

    freezer = FREEZER.read_text(encoding="utf-8")
    assert (
        '"controller": H.bind("slurm/dino_rcde_track_r_post_e1_controller_v3.sbatch")'
        in freezer
    )
    assert (
        '"tests/test_dino_rcde_track_r_post_e1_controller_v3.py"' in freezer
    )
    assert (
        '"controller": H.bind("slurm/dino_rcde_track_r_post_e1_controller_v2.sbatch")'
        not in freezer
    )
