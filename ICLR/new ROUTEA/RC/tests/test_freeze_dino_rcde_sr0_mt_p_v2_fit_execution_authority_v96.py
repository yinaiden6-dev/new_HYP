from __future__ import annotations

import ast
import inspect

import freeze_dino_rcde_sr0_mt_p_v2_fit_execution_authority_v96 as F


def test_v96_authority_is_fit_only_and_fail_closed() -> None:
    value = F.build()
    assert value["status"] == "RCDE_SR0_MT_P_V2_FORMAL_FIT_EXECUTION_AUTHORIZED"
    assert value["authorized_scope"] == {
        "p_v2_exact_resume_validation": True,
        "p_v2_formal_fit_execution": True,
    }
    assert value["fit_count"] == 16
    assert value["inner_fit_count"] == 12
    assert value["outer_refit_count"] == 4
    assert value["training_authorized"] is True
    assert value["consumable_checkpoint_authorized"] is False
    assert value["p_lock_materialization_authorized"] is False
    assert value["v_training_authorized"] is False
    assert value["heldout_scoring_authorized"] is False
    assert value["scientific_GO_or_NO_GO"] is None
    assert value["next_authorized_stage"] is None
    assert value["automatic_stage_advance"] is False
    assert value["resource_contract"] == {
        "partition": "cpuonly",
        "array_spec": "0-15%4",
        "array_task_count": 16,
        "array_max_concurrency": 4,
        "cpus_per_task": 1,
        "memory_megabytes": 16384,
        "walltime_seconds": 14400,
    }


def test_freezer_is_declarative_and_does_not_run_training() -> None:
    calls: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(F))):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert "run_fit" not in calls
    assert "sbatch" not in calls
    assert "subprocess" not in calls


def test_fit_order_matches_the_frozen_array_order() -> None:
    assert F.expected_fit_ids() == [
        "P_OUTER1_INNER2_FIT",
        "P_OUTER1_INNER3_FIT",
        "P_OUTER1_INNER4_FIT",
        "P_OUTER1_OUTER_REFIT",
        "P_OUTER2_INNER1_FIT",
        "P_OUTER2_INNER3_FIT",
        "P_OUTER2_INNER4_FIT",
        "P_OUTER2_OUTER_REFIT",
        "P_OUTER3_INNER1_FIT",
        "P_OUTER3_INNER2_FIT",
        "P_OUTER3_INNER4_FIT",
        "P_OUTER3_OUTER_REFIT",
        "P_OUTER4_INNER1_FIT",
        "P_OUTER4_INNER2_FIT",
        "P_OUTER4_INNER3_FIT",
        "P_OUTER4_OUTER_REFIT",
    ]
