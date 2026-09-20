#!/usr/bin/env python3
"""Formal four-fold GX-CBNR R0 scientific reducer.

The expensive representation is replayed from the independently validated
fold checkpoint and target-free relational cache.  Target/rival roles are
opened only after all four target-free fold populations have been sealed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "programs"))

import run_dino_rcde_gx_formal_fold_train_v1 as FOLD
from rc_aslo_xf.dino_rcde_gx_cbnr_v3 import (
    geometric_exclusivity,
    gx_action,
    pair_centered_gx_residual,
)
from rc_aslo_xf.dino_rcde_gx_formal_head_train_v1 import (
    EXPECTED_BRANCHES,
    NULL_NAMES,
    RAW_CORRECT,
    episode_loss,
)
from rc_aslo_xf.dino_rcde_gx_relational_head_cache_v1 import (
    replay_candidate_energy,
)


SCHEMA_VERSION = "rc_dino_rcde_gx_formal_r0_science_v1_20260825"
AUTHORITY_STATUS = "GX_CBNR_R0_SCIENTIFIC_REDUCTION_EXECUTION_AUTHORIZED"
GO = "GX_CBNR_R0_REPRESENTATION_GO"
NO_GO = "GX_CBNR_R0_NO_DEPLOYABLE_GEOMETRIC_EXCLUSIVITY"
FOLDS = (1, 2, 3, 4)
CONTROLS = ("C_BIND", "P_COORD", "N_REGION_RESAMPLE")
BOOTSTRAP_ENDPOINTS = (
    "correctness_increment",
    "target_z_minus_rival_z",
    "target_real_minus_C_BIND",
    "target_real_minus_P_COORD",
    "target_real_minus_N_REGION_RESAMPLE",
)
BOOTSTRAP_SEED = 17
BOOTSTRAP_REPLICATES = 10_000


class R0ScienceError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise R0ScienceError(message)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    require(
        resolved.is_file()
        and not resolved.is_symlink()
        and not {"opened", "sealed"}.intersection(
            component.lower() for component in resolved.parts
        ),
        f"unsafe input path: {resolved}",
    )
    value = json.loads(resolved.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object absent: {resolved}")
    return value


def bound_path(authority: Mapping[str, Any], name: str) -> Path:
    binding = authority.get("bindings", {}).get(name)
    require(isinstance(binding, Mapping), f"authority binding absent: {name}")
    raw = Path(str(binding.get("path", "")))
    path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    require(
        path.is_file()
        and not path.is_symlink()
        and not {"opened", "sealed"}.intersection(
            component.lower() for component in path.parts
        )
        and path.stat().st_size == int(binding.get("bytes", -1))
        and file_sha256(path) == binding.get("sha256"),
        f"authority binding drift: {name}",
    )
    return path


def validate_authority(path: Path, output: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    authority = read_json(path)
    require(
        authority.get("status") == AUTHORITY_STATUS
        and authority.get("logical_sha256") == logical_sha256(authority)
        and authority.get("scientific_reduction_authorized") is True
        and authority.get("opened_or_sealed_access_authorized") is False
        and authority.get("automatic_stage_advance") is False
        and output.resolve(strict=False)
        == (ROOT / str(authority.get("output", ""))).resolve(strict=False),
        "scientific-reduction authority envelope drift",
    )
    names = [
        "parent_training_authority_v54",
        "address_manifest",
        "loss_role_manifest",
        "cache_index",
    ]
    for fold in FOLDS:
        names.extend(
            (
                f"fold{fold}_result",
                f"fold{fold}_validation",
                f"fold{fold}_checkpoint",
                f"fold{fold}_init_checkpoint",
            )
        )
    paths = {name: bound_path(authority, name) for name in names}
    receipt = {
        "authority_path": str(path.resolve()),
        "authority_bytes": path.stat().st_size,
        "authority_sha256": file_sha256(path.resolve()),
        "authority_logical_sha256": authority["logical_sha256"],
        "binding_count": len(paths),
        "binding_population_sha256": canonical_sha256(
            [
                {"name": name, "path": str(paths[name]), "sha256": file_sha256(paths[name])}
                for name in sorted(paths)
            ]
        ),
    }
    receipt["logical_sha256"] = logical_sha256(receipt)
    authority["_execution_receipt"] = receipt
    return authority, paths


def candidate_energies(model: torch.nn.Module, branches: Mapping[str, Any]) -> dict[str, float]:
    require(
        set(branches) == set(EXPECTED_BRANCHES)
        and tuple(EXPECTED_BRANCHES) == ("REAL", *CONTROLS)
        and tuple(NULL_NAMES) == CONTROLS,
        "candidate branch population is not exactly REAL+C/P/N",
    )
    values = {
        branch: replay_candidate_energy(model, branches[branch], for_training=True).energy
        for branch in EXPECTED_BRANCHES
    }
    gx = geometric_exclusivity(
        values["REAL"], tuple(values[name] for name in NULL_NAMES)
    )
    output = {branch: float(values[branch].detach().cpu()) for branch in EXPECTED_BRANCHES}
    output["null_logmeanexp_CPN"] = float(gx.null_logmeanexp.detach().cpu())
    output["Z_CPN"] = float(gx.z.detach().cpu())
    require(all(math.isfinite(value) for value in output.values()), "nonfinite candidate energy")
    return output


def _exact_float(left: float, right: float) -> bool:
    return torch.equal(torch.tensor(left, dtype=torch.float32), torch.tensor(right, dtype=torch.float32))


def replay_record(
    model: torch.nn.Module,
    episode: Any,
    role: Mapping[str, Any],
    runner_record: Mapping[str, Any],
) -> dict[str, Any]:
    with torch.no_grad():
        target = candidate_energies(model, episode.target_branches)
        rival = candidate_energies(model, episode.rival_branches)
        # A second replay in reversed candidate order is the actual reorder check.
        rival_reordered = candidate_energies(model, episode.rival_branches)
        target_reordered = candidate_energies(model, episode.target_branches)
        loss = episode_loss(model, episode)

    raw = float(episode.raw_target_margin)
    target_z = target["Z_CPN"]
    rival_z = rival["Z_CPN"]
    geometric = float(loss.geometric_margin)
    final = float(loss.final_margin)
    raw_correct = episode.stratum == RAW_CORRECT
    action = (
        gx_action(
            raw_challenger_minus_winner=torch.tensor(-raw, dtype=torch.float32),
            challenger_z=torch.tensor(rival_z, dtype=torch.float32),
            winner_z=torch.tensor(target_z, dtype=torch.float32),
            all_nulls_eligible=True,
        )
        if raw_correct
        else gx_action(
            raw_challenger_minus_winner=torch.tensor(raw, dtype=torch.float32),
            challenger_z=torch.tensor(target_z, dtype=torch.float32),
            winner_z=torch.tensor(rival_z, dtype=torch.float32),
            all_nulls_eligible=True,
        )
    )
    forced_hold = gx_action(
        raw_challenger_minus_winner=torch.tensor(raw if not raw_correct else -raw, dtype=torch.float32),
        challenger_z=torch.tensor(target_z if not raw_correct else rival_z, dtype=torch.float32),
        winner_z=torch.tensor(rival_z if not raw_correct else target_z, dtype=torch.float32),
        all_nulls_eligible=False,
    )
    centered = pair_centered_gx_residual(
        torch.tensor(target_z, dtype=torch.float32),
        torch.tensor(rival_z, dtype=torch.float32),
    )
    reverse = torch.tensor(rival_z, dtype=torch.float32) - torch.tensor(target_z, dtype=torch.float32)
    natural_hold = action.action == "RELATIVE_NULL_HOLD"
    deployed_margin = raw if natural_hold else final
    final_correct = deployed_margin > 0.0
    invariants = {
        "candidate_swap_exact": torch.equal(centered.target_minus_rival, -reverse),
        "candidate_reorder_exact": target == target_reordered and rival == rival_reordered,
        "pair_zero_sum_exact": torch.equal(centered.target + centered.rival, torch.zeros_like(centered.target)),
        "forced_ineligible_hold_exact": forced_hold.action == "RELATIVE_NULL_HOLD",
        "natural_hold_preserves_raw_exact": (not natural_hold) or deployed_margin == raw,
    }
    require(
        _exact_float(target_z, float(loss.target_z))
        and _exact_float(rival_z, float(loss.rival_z))
        and _exact_float(geometric, target_z - rival_z)
        and _exact_float(final, raw + geometric),
        "checkpoint replay/loss closure drift",
    )
    for field, expected in (
        ("raw_target_margin", raw),
        ("target_z", target_z),
        ("rival_z", rival_z),
        ("geometric_margin", geometric),
        ("final_margin", final),
    ):
        require(_exact_float(float(runner_record[field]), expected), f"runner replay drift: {field}")
    require(runner_record["action"] == action.action, "runner action replay drift")

    target_real_minus_null = {
        name: target["REAL"] - target[name] for name in CONTROLS
    }
    rival_real_minus_null = {
        name: rival["REAL"] - rival[name] for name in CONTROLS
    }
    value: dict[str, Any] = {
        "outer_fold": int(role["outer_fold"]),
        "execution_ordinal": int(role["execution_ordinal"]),
        "query_id": str(role["query_id"]),
        "episode_id": str(episode.episode_id),
        "group_sha256": str(role["group_sha256"]),
        "track": str(role["track"]),
        "raw_correct": raw_correct,
        "raw_target_margin": raw,
        "candidate_energies": {"target": target, "rival": rival},
        "target_real_minus_null": target_real_minus_null,
        "rival_real_minus_null": rival_real_minus_null,
        "target_z_minus_rival_z": geometric,
        "final_target_margin": final,
        # The registered action path preserves RAW exactly on HOLD.  An
        # overlay-only positive margin therefore cannot count as a deployed
        # final direction or rescue.
        "raw_wrong_final_direction": (not raw_correct) and final_correct,
        "action": action.action,
        "deployed_target_margin": deployed_margin,
        "final_correct": final_correct,
        "rescue": (not raw_correct) and final_correct,
        "break": raw_correct and not final_correct,
        "correctness_increment": float(final_correct) - float(raw_correct),
        "null_eligibility": {name: True for name in CONTROLS},
        "invariants": invariants,
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def registered_bootstrap(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if field.startswith("target_real_minus_"):
            control = field.removeprefix("target_real_minus_")
            value = float(row["target_real_minus_null"][control])
        else:
            value = float(row[field])
        require(math.isfinite(value), f"nonfinite bootstrap endpoint: {field}")
        grouped[str(row["group_sha256"])].append(value)
    require(len(grouped) > 1, f"insufficient bootstrap clusters: {field}")
    cluster_means = np.asarray(
        [np.mean(grouped[group], dtype=np.float64) for group in sorted(grouped)],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(
        0,
        cluster_means.size,
        size=(BOOTSTRAP_REPLICATES, cluster_means.size),
    )
    samples = cluster_means[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "cluster_field": "group_sha256",
        "cluster_count": int(cluster_means.size),
        "cluster_means_equal_weighted": True,
        "observed_group_balanced_mean": float(cluster_means.mean()),
        "rng": "numpy.PCG64",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "quantile_method": "linear",
        "two_sided_95_ci": [float(lower), float(upper)],
        "lower_strictly_positive": bool(lower > 0.0),
    }


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows], dtype=np.float64))


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    auxiliary_t_by_record: Mapping[tuple[int, int], tuple[bool, bool]] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in records]
    require(
        len(rows) == 32
        and len({(int(row["outer_fold"]), int(row["execution_ordinal"])) for row in rows}) == 32
        and sum(bool(row["raw_correct"]) for row in rows) == 16
        and sum(not bool(row["raw_correct"]) for row in rows) == 16,
        "Balanced-32 population drift",
    )
    require(
        all(row["record_sha256"] == canonical_sha256({k: v for k, v in row.items() if k != "record_sha256"}) for row in rows),
        "evaluation record SHA drift",
    )
    required_invariants = {
        "candidate_swap_exact",
        "candidate_reorder_exact",
        "pair_zero_sum_exact",
        "forced_ineligible_hold_exact",
        "natural_hold_preserves_raw_exact",
    }
    for row in rows:
        raw_correct = bool(row["raw_correct"])
        final_correct = bool(row["final_correct"])
        expected_increment = float(final_correct) - float(raw_correct)
        require(
            set(row["null_eligibility"]) == set(CONTROLS)
            and all(bool(row["null_eligibility"][name]) for name in CONTROLS)
            and set(row["invariants"]) == required_invariants
            and bool(row["raw_wrong_final_direction"])
            is ((not raw_correct) and final_correct)
            and bool(row["rescue"]) is ((not raw_correct) and final_correct)
            and bool(row["break"]) is (raw_correct and not final_correct)
            and float(row["correctness_increment"]) == expected_increment
            and (
                row["action"] != "RELATIVE_NULL_HOLD"
                or float(row["deployed_target_margin"])
                == float(row["raw_target_margin"])
            ),
            "evaluation record action/outcome/null/invariant drift",
        )
    fold_summaries: dict[str, Any] = {}
    for fold in FOLDS:
        selected = [row for row in rows if int(row["outer_fold"]) == fold]
        require(len(selected) == 8, f"fold {fold} population drift")
        correct = [row for row in selected if bool(row["raw_correct"])]
        wrong = [row for row in selected if not bool(row["raw_correct"])]
        require(len(correct) == len(wrong) == 4, f"fold {fold} stratum drift")
        rescue = sum(bool(row["rescue"]) for row in selected)
        breaks = sum(bool(row["break"]) for row in selected)
        mean_null = {
            control: float(np.mean([float(row["target_real_minus_null"][control]) for row in selected], dtype=np.float64))
            for control in CONTROLS
        }
        mean_z = _mean(selected, "target_z_minus_rival_z")
        fold_summaries[str(fold)] = {
            "query_count": 8,
            "raw_correct_count": 4,
            "raw_wrong_count": 4,
            "raw_wrong_final_direction_count": sum(bool(row["raw_wrong_final_direction"]) for row in wrong),
            "raw_correct_retained_count": sum(bool(row["final_correct"]) for row in correct),
            "rescue_count": rescue,
            "break_count": breaks,
            "net_rescue": rescue - breaks,
            "mean_target_z_minus_rival_z": mean_z,
            "mean_target_real_minus_null": mean_null,
            "gates": {
                "rescue_strictly_exceeds_break": rescue > breaks,
                "net_rescue_positive": rescue - breaks > 0,
                "mean_target_z_minus_rival_z_positive": mean_z > 0.0,
                "target_real_minus_each_null_positive": all(value > 0.0 for value in mean_null.values()),
            },
        }

    track_summaries: dict[str, Any] = {}
    for track in sorted({str(row["track"]) for row in rows}):
        selected = [row for row in rows if str(row["track"]) == track]
        rescue = sum(bool(row["rescue"]) for row in selected)
        breaks = sum(bool(row["break"]) for row in selected)
        track_summaries[track] = {
            "query_count": len(selected),
            "raw_correct_count": sum(bool(row["raw_correct"]) for row in selected),
            "raw_wrong_count": sum(not bool(row["raw_correct"]) for row in selected),
            "rescue_count": rescue,
            "break_count": breaks,
            "net_rescue": rescue - breaks,
            "nonnegative_gate": rescue >= breaks,
        }

    def eligibility_slice(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        denominator = 2 * len(selected)
        return {
            control: {
                "candidate_denominator_count": denominator,
                "eligible_candidate_count": denominator,
                "ineligible_candidate_count": 0,
                "reason_counts": {"ELIGIBLE": denominator},
            }
            for control in CONTROLS
        }

    null_summary = {
        "controls": list(CONTROLS),
        "overall": eligibility_slice(rows),
        "by_fold": {
            str(fold): eligibility_slice([row for row in rows if int(row["outer_fold"]) == fold])
            for fold in FOLDS
        },
        "by_track": {
            track: eligibility_slice([row for row in rows if str(row["track"]) == track])
            for track in track_summaries
        },
        "silent_denominator_drop_count": 0,
    }

    t_map = auxiliary_t_by_record or {}
    t_rows = []
    for row in rows:
        pair = t_map.get((int(row["outer_fold"]), int(row["execution_ordinal"])), (False, False))
        require(len(pair) == 2, "auxiliary T candidate cardinality drift")
        t_rows.extend(
            {"fold": int(row["outer_fold"]), "track": str(row["track"]), "present": bool(value)}
            for value in pair
        )
    auxiliary_t = {
        "name": "T_ROOT_ASSIGNMENT_POPULATION",
        "selected_candidate_denominator_count": len(t_rows),
        "present_count": sum(row["present"] for row in t_rows),
        "absent_count": sum(not row["present"] for row in t_rows),
        "by_fold": {
            str(fold): {
                "candidate_denominator_count": sum(row["fold"] == fold for row in t_rows),
                "present_count": sum(row["fold"] == fold and row["present"] for row in t_rows),
            }
            for fold in FOLDS
        },
        "by_track": {
            track: {
                "candidate_denominator_count": sum(row["track"] == track for row in t_rows),
                "present_count": sum(row["track"] == track and row["present"] for row in t_rows),
            }
            for track in track_summaries
        },
        "enters_calibration": False,
        "enters_loss": False,
        "enters_action": False,
        "enters_gate": False,
        "enters_bootstrap": False,
    }

    invariant_names = tuple(sorted(required_invariants))
    invariants = {
        "record_count": len(rows),
        **{
            f"{name}_count": sum(bool(row["invariants"][name]) for row in rows)
            for name in invariant_names
        },
        "all_pass": all(all(bool(value) for value in row["invariants"].values()) for row in rows),
    }
    bootstrap = {endpoint: registered_bootstrap(rows, endpoint) for endpoint in BOOTSTRAP_ENDPOINTS}
    raw_wrong_direction = sum(bool(row["raw_wrong_final_direction"]) for row in rows)
    retained = sum(bool(row["raw_correct"]) and bool(row["final_correct"]) for row in rows)
    rescues = sum(bool(row["rescue"]) for row in rows)
    breaks = sum(bool(row["break"]) for row in rows)
    totals = {
        "query_count": 32,
        "raw_correct_count": 16,
        "raw_wrong_count": 16,
        "raw_wrong_final_direction_count": raw_wrong_direction,
        "raw_correct_retained_count": retained,
        "rescue_count": rescues,
        "break_count": breaks,
        "net_rescue": rescues - breaks,
    }
    gates = {
        "raw_wrong_direction_at_least_11_of_16": raw_wrong_direction >= 11,
        "raw_correct_retention_exactly_16_of_16": retained == 16,
        "every_fold_rescue_strictly_exceeds_break": all(value["gates"]["rescue_strictly_exceeds_break"] for value in fold_summaries.values()),
        "every_fold_net_rescue_positive": all(value["gates"]["net_rescue_positive"] for value in fold_summaries.values()),
        "every_fold_mean_target_z_minus_rival_z_positive": all(value["gates"]["mean_target_z_minus_rival_z_positive"] for value in fold_summaries.values()),
        "every_fold_target_real_minus_each_null_positive": all(value["gates"]["target_real_minus_each_null_positive"] for value in fold_summaries.values()),
        "difficult_track_rescue_at_least_break": "difficult" in track_summaries and track_summaries["difficult"]["nonnegative_gate"],
        "every_available_track_nonnegative": all(value["nonnegative_gate"] for value in track_summaries.values()),
        "all_invariants_pass": invariants["all_pass"],
        "all_nulls_reported_without_drop": null_summary["silent_denominator_drop_count"] == 0,
        "all_required_bootstrap_lower_strictly_positive": all(value["lower_strictly_positive"] for value in bootstrap.values()),
    }
    gates["all_gates_pass"] = all(gates.values())
    return {
        "fold_summaries": fold_summaries,
        "track_summaries": track_summaries,
        "null_eligibility_summary": null_summary,
        "auxiliary_t_diagnostic": auxiliary_t,
        "invariants": invariants,
        "bootstrap": bootstrap,
        "gates": gates,
        "totals": totals,
    }


def _validate_fold_result(result: Mapping[str, Any], validation: Mapping[str, Any], fold: int, result_path: Path) -> None:
    require(
        result.get("schema_version") == FOLD.SCHEMA_VERSION
        and result.get("status") == FOLD.RESULT_STATUS
        and result.get("outer_fold") == fold
        and result.get("completed_updates") == 128
        and result.get("evaluation_record_count") == 8
        and result.get("logical_sha256") == FOLD.logical_sha256(result),
        f"fold {fold} result envelope drift",
    )
    require(
        validation.get("status") == "GX_CBNR_FORMAL_FOLD_TRAIN_VALIDATION_PASS"
        and validation.get("validation_pass") is True
        and validation.get("outer_fold") == fold
        and validation.get("runner_result_sha256") == file_sha256(result_path)
        and validation.get("evaluation_record_population_sha256") == result["evaluation_record_population_sha256"]
        and validation.get("logical_sha256") == logical_sha256(validation),
        f"fold {fold} independent validation drift",
    )


def reduce_from_authority(authority_path: Path, output: Path) -> dict[str, Any]:
    authority, paths = validate_authority(authority_path, output)
    training_authority = read_json(paths["parent_training_authority_v54"])
    require(
        training_authority.get("status") == "GX_CBNR_FORMAL_FOURFOLD_TRAINING_EXECUTION_AUTHORIZED"
        and training_authority.get("logical_sha256") == logical_sha256(training_authority)
        and training_authority.get("scientific_reduction_authorized") is False,
        "parent training authority drift",
    )

    # Seal every target-free fold before opening the role manifest at all.
    target_free: dict[int, Mapping[str, Any]] = {}
    models: dict[int, torch.nn.Module] = {}
    fold_results: dict[int, Mapping[str, Any]] = {}
    fold_validations: dict[int, Mapping[str, Any]] = {}
    for fold in FOLDS:
        init = paths[f"fold{fold}_init_checkpoint"]
        target_free[fold] = FOLD.load_target_free_fold(
            address_manifest_path=paths["address_manifest"],
            cache_index_path=paths["cache_index"],
            outer_fold=fold,
            expected_init_checkpoint_sha256=file_sha256(init),
        )
        model, receipt = FOLD.default_model_loader(init, fold)
        require(receipt["sha256"] == file_sha256(init), "init model receipt drift")
        checkpoint = torch.load(paths[f"fold{fold}_checkpoint"], map_location="cpu", weights_only=True)
        require(
            checkpoint.get("schema_version") == FOLD.CHECKPOINT_SCHEMA
            and checkpoint.get("status") == FOLD.RESULT_STATUS
            and checkpoint.get("outer_fold") == fold
            and checkpoint.get("update") == 128,
            f"fold {fold} checkpoint drift",
        )
        state = FOLD.formal_state_from_payload(checkpoint["training_state"])
        model.load_state_dict(state.model_state_dict, strict=True)
        model.eval()
        models[fold] = model
        result = read_json(paths[f"fold{fold}_result"])
        validation = read_json(paths[f"fold{fold}_validation"])
        _validate_fold_result(result, validation, fold, paths[f"fold{fold}_result"])
        require(result["checkpoint_sha256"] == file_sha256(paths[f"fold{fold}_checkpoint"]), "fold result/checkpoint drift")
        fold_results[fold] = result
        fold_validations[fold] = validation

    target_free_seal = canonical_sha256(
        [
            {
                "outer_fold": fold,
                "context_count": target_free[fold]["target_free_context_count"],
                "cache_count": target_free[fold]["target_free_cache_count"],
                "seal_sha256": target_free[fold]["target_free_seal_sha256"],
            }
            for fold in FOLDS
        ]
    )

    records: list[dict[str, Any]] = []
    selected_addresses: set[tuple[int, int]] = set()
    for fold in FOLDS:
        joined = FOLD.open_loss_roles_and_build_fold(
            loss_role_manifest_path=paths["loss_role_manifest"],
            target_free=target_free[fold],
            outer_fold=fold,
        )
        runner_by_execution = {
            int(row["execution_ordinal"]): row
            for row in fold_results[fold]["evaluation_records"]
        }
        for episode, role in joined["evaluations"]:
            execution = int(role["execution_ordinal"])
            require(execution in runner_by_execution, "runner evaluation address absent")
            records.append(replay_record(models[fold], episode, role, runner_by_execution[execution]))
            selected_addresses.add((fold, execution))

    cache_index = read_json(paths["cache_index"])
    auxiliary_t_by_record: dict[tuple[int, int], tuple[bool, bool]] = {}
    for row in cache_index["records"]:
        address = (int(row["outer_fold"]), int(row["execution_ordinal"]))
        if address not in selected_addresses:
            continue
        candidates = row.get("candidate_caches")
        require(isinstance(candidates, list) and len(candidates) == 2, "cache-index candidate summary drift")
        auxiliary_t_by_record[address] = tuple(bool(item["auxiliary_t_present"]) for item in candidates)  # type: ignore[assignment]
    require(set(auxiliary_t_by_record) == selected_addresses, "auxiliary T evaluation population drift")

    records.sort(key=lambda row: (int(row["outer_fold"]), int(row["execution_ordinal"])))
    summaries = summarize_records(records, auxiliary_t_by_record=auxiliary_t_by_record)
    status = GO if summaries["gates"]["all_gates_pass"] else NO_GO
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "scientific_GO_or_NO_GO": status,
        "claim_level": "INTERNAL_BALANCED32_MATCHED_NULL_CALIBRATED_CONNECTED_DINO_RELATIONAL_ENERGY_SCREEN",
        "claim_boundary": {
            "matched_null_calibrated_connected_dino_relational_energy_only": True,
            "full_gallery_claim_authorized": False,
            "ownership_claim_authorized": False,
            "target_presence_or_absence_claim_authorized": False,
            "opened_or_sealed_claim_authorized": False,
            "downstream_action_experiment_requires_separate_authority": True,
        },
        "input_receipts": {
            "authority_execution_receipt": authority["_execution_receipt"],
            "parent_training_authority_sha256": file_sha256(paths["parent_training_authority_v54"]),
            "address_manifest_sha256": file_sha256(paths["address_manifest"]),
            "loss_role_manifest_sha256": file_sha256(paths["loss_role_manifest"]),
            "cache_index_sha256": file_sha256(paths["cache_index"]),
            "fold_result_sha256": {str(fold): file_sha256(paths[f"fold{fold}_result"]) for fold in FOLDS},
            "fold_validation_sha256": {str(fold): file_sha256(paths[f"fold{fold}_validation"]) for fold in FOLDS},
            "fold_checkpoint_sha256": {str(fold): file_sha256(paths[f"fold{fold}_checkpoint"]) for fold in FOLDS},
        },
        "access_audit": {
            "all_four_target_free_folds_sealed_before_role_join": True,
            "target_free_fourfold_seal_sha256": target_free_seal,
            "target_free_context_count": sum(int(target_free[fold]["target_free_context_count"]) for fold in FOLDS),
            "target_free_cache_count": sum(int(target_free[fold]["target_free_cache_count"]) for fold in FOLDS),
            "loss_role_manifest_open_count_after_seal": 4,
            "label_target_rival_role_model_input_count": 0,
            "opened_sealed_read_count": 0,
        },
        "evaluation_records": records,
        "evaluation_record_population_sha256": canonical_sha256(records),
        **summaries,
        "automatic_stage_advance": False,
        # Even a representation GO does not itself authorize or advance the
        # separately frozen action/full-C128 experiment.
        "next_authorized_stage": None,
    }
    value["logical_sha256"] = logical_sha256(value)
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"immutable reducer output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = reduce_from_authority(args.authority, args.output)
        atomic_json(args.output.resolve(strict=False), value)
        print(json.dumps({"status": value["status"], "logical_sha256": value["logical_sha256"]}, sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps({"status": "GX_CBNR_R0_SCIENTIFIC_REDUCTION_ABORT", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
