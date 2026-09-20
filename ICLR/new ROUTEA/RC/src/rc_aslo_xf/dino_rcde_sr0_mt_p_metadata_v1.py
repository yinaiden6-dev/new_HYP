"""Pure metadata builders for the formal SR0-MT P execution DAG.

The functions in this module never load natural token payloads.  They consume
only independently validated feature ledgers, the later-authorized I0
metadata-only postjoin, and independently validated lock ledgers.  CLI callers
must validate natural authority before opening any of those inputs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .dino_rcde_cw1_multitile_sr0_p_v1 import (
    FEATURE_SCHEMA_SHA256,
    P_CHECKPOINT_INTERVAL_UPDATES,
    P_EPISODES_PER_UPDATE,
    P_FINAL_LEARNING_RATE,
    P_GRADIENT_CLIP_L2,
    P_LEARNING_RATE,
    P_SEED,
    P_UPDATES_TOTAL,
    P_WARMUP_UPDATES,
    P_WEIGHT_DECAY,
)
from .dino_rcde_sr0_mt_p_runtime_v1 import (
    FEATURE_LEDGER_SCHEMA,
    FEATURE_VALIDATION_SCHEMA,
    FIT_VALIDATION_SCHEMA,
    FIT_MANIFEST_SCHEMA,
    CHECKPOINT_SCHEMA,
    LOCK_MANIFEST_SCHEMA,
    LOCK_LEDGER_SCHEMA,
    PAIR_JOIN_SCHEMA,
    P_DIRECTIONS,
    FormalPContractError,
    canonical_sha256,
    feature_ledger_index,
    feature_ledger_logical_sha256,
    file_sha256,
    load_json_mapping,
    load_torch_mapping,
)


FEATURE_SHARD_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_feature_shard_manifest_v1_20260815"
)
FEATURE_AGGREGATION_RESULT_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_feature_aggregation_result_v1_20260815"
)
FEATURE_AGGREGATE_VALIDATION_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_feature_aggregate_validation_v1_20260815"
)
FEATURE_AGGREGATE_VALIDATION_PASS = (
    "RCDE_SR0_MT_P_FEATURE_AGGREGATE_INDEPENDENT_VALIDATION_PASS"
)
P_METADATA_PLAN_SCHEMA = "rc_dino_rcde_sr0_mt_p_metadata_plan_v1_20260815"
LOCK_MEMBERSHIP_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_lock_membership_manifest_v1_20260815"
)
OOF_LOCK_INPUT_MANIFEST_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_oof_lock_input_manifest_v1_20260815"
)
OOF_LOCK_CONCAT_SCHEMA = "rc_dino_rcde_sr0_mt_p_oof_lock_concat_v1_20260815"
OOF_LOCK_CONCAT_VALIDATION_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_oof_lock_concat_validation_v1_20260815"
)
LOCK_MANIFEST_FINALIZATION_RECEIPT_SCHEMA = (
    "rc_dino_rcde_sr0_mt_p_lock_manifest_finalization_receipt_v1_20260815"
)
I0_POSTJOIN_SCHEMA = "rc_dino_rcde_sr0_mt_i0_postjoin_ledger_v1_20260815"
I0_POSTJOIN_VALIDATION_SCHEMA = (
    "rc_dino_rcde_sr0_mt_i0_postjoin_validation_v1_20260815"
)
I0_POSTJOIN_PASS = "RCDE_SR0_MT_I0_POSTJOIN_INDEPENDENT_VALIDATION_PASS"
FEATURE_VALIDATION_PASS = "RCDE_SR0_MT_P_FEATURE_INDEPENDENT_VALIDATION_PASS"
LOCK_VALIDATION_PASS = "RCDE_SR0_MT_P_LOCK_INDEPENDENT_VALIDATION_PASS"
FIT_VALIDATION_PASS = "RCDE_SR0_MT_P_FIT_INDEPENDENT_VALIDATION_PASS"


def _logical(value: Mapping[str, object]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalPContractError(message)


def _safe_path(value: object, *, name: str) -> Path:
    path = Path(str(value)).resolve()
    _require(path.is_file() and not path.is_symlink(), f"{name} is absent or unsafe")
    return path


def invocation_execution_manifest_sha256(argv: Sequence[str]) -> str | None:
    """Return the physical manifest SHA for exclusive manifest-mode invocation."""

    values = list(argv)
    if "--execution-manifest" not in values:
        return None
    _require(
        len(values) == 2 and values[0] == "--execution-manifest",
        "execution-manifest mode cannot be mixed with explicit arguments",
    )
    path = _safe_path(values[1], name="execution manifest")
    return file_sha256(path)


def _validated_feature(
    ledger_path: Path,
    ledger_sha256: str,
    validation_path: Path,
    validation_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    _require(file_sha256(ledger_path) == ledger_sha256, "feature ledger file hash drift")
    _require(
        file_sha256(validation_path) == validation_sha256,
        "feature validation file hash drift",
    )
    validation = load_json_mapping(validation_path, name="P feature validation")
    _require(
        validation.get("schema_version") == FEATURE_VALIDATION_SCHEMA
        and validation.get("status") == FEATURE_VALIDATION_PASS
        and validation.get("validation_pass") is True
        and validation.get("ledger_file_sha256") == ledger_sha256
        and validation.get("logical_sha256") == _logical(validation),
        "independent feature validation/hash closure missing",
    )
    ledger = load_torch_mapping(ledger_path, name="validated P feature ledger")
    feature_ledger_index(ledger)
    return ledger, validation


def aggregate_validated_feature_shards(
    manifest: Mapping[str, object],
    *,
    expected_query_count: int,
    expected_candidate_count: int,
) -> dict[str, object]:
    """Merge disjoint validated shards into one canonical feature ledger."""

    _require(
        manifest.get("schema_version") == FEATURE_SHARD_MANIFEST_SCHEMA
        and manifest.get("logical_sha256") == _logical(manifest),
        "feature shard manifest schema/logical hash drift",
    )
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and bool(shards), "feature shard list is empty")
    records: list[dict[str, object]] = []
    axes: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    header: dict[str, object] | None = None
    executions: set[int] = set()
    query_ids: set[str] = set()
    audit_totals: dict[str, int] = defaultdict(int)
    for ordinal, raw in enumerate(shards):
        _require(isinstance(raw, Mapping), "feature shard entry is not a mapping")
        ledger_path = _safe_path(raw.get("ledger_path"), name="feature shard")
        validation_path = _safe_path(
            raw.get("validation_path"), name="feature shard validation"
        )
        ledger_sha = str(raw.get("ledger_file_sha256"))
        validation_sha = str(raw.get("validation_file_sha256"))
        ledger, validation = _validated_feature(
            ledger_path, ledger_sha, validation_path, validation_sha
        )
        current = {
            "feature_schema": ledger.get("feature_schema"),
            "feature_schema_sha256": ledger.get("feature_schema_sha256"),
            "feature_dtype": ledger.get("feature_dtype"),
            "vector_scalar_max_ulps": ledger.get("vector_scalar_max_ulps"),
            "candidate_count_per_query": ledger.get("candidate_count_per_query"),
            "direction_count": ledger.get("direction_count"),
            "target_free": ledger.get("target_free"),
        }
        if header is None:
            header = current
        _require(current == header, "feature shard header drift")
        _require(
            ledger.get("candidate_count_per_query") == expected_candidate_count,
            "feature shard candidate arity drift",
        )
        raw_axes = ledger.get("query_candidate_axes")
        raw_records = ledger.get("records")
        _require(
            isinstance(raw_axes, list) and isinstance(raw_records, list),
            "feature shard axes/records absent",
        )
        shard_executions: list[int] = []
        for axis in raw_axes:
            _require(isinstance(axis, Mapping), "feature shard axis is not a mapping")
            execution = int(axis.get("execution_ordinal", -1))
            query_id = str(axis.get("query_id", ""))
            _require(
                execution >= 0 and query_id and execution not in executions
                and query_id not in query_ids,
                "feature shard query overlap/address drift",
            )
            executions.add(execution)
            query_ids.add(query_id)
            shard_executions.append(execution)
            axes.append(dict(axis))
        _require(
            shard_executions == sorted(shard_executions),
            "feature shard execution order is not canonical",
        )
        records.extend(dict(item) for item in raw_records if isinstance(item, Mapping))
        audit = ledger.get("protected_access_audit")
        _require(
            isinstance(audit, Mapping) and all(int(value) == 0 for value in audit.values()),
            "feature shard protected access is nonzero",
        )
        for key, value in audit.items():
            audit_totals[str(key)] += int(value)
        provenance.append(
            {
                "shard_ordinal": ordinal,
                "ledger_file_sha256": ledger_sha,
                "ledger_logical_sha256": ledger["logical_sha256"],
                "validation_file_sha256": validation_sha,
                "validation_logical_sha256": validation["logical_sha256"],
                "query_execution_ordinals": shard_executions,
                "query_execution_ordinals_sha256": canonical_sha256(shard_executions),
            }
        )
    _require(
        executions == set(range(expected_query_count))
        and len(query_ids) == expected_query_count,
        "aggregate feature query population is not exact/contiguous",
    )
    _require(header is not None, "feature shard header absent")
    axes.sort(key=lambda item: int(item["execution_ordinal"]))
    records.sort(
        key=lambda item: (
            int(item["execution_ordinal"]),
            int(item["candidate_position"]),
            P_DIRECTIONS.index(str(item["direction"])),
        )
    )
    expected_records = expected_query_count * expected_candidate_count * len(P_DIRECTIONS)
    _require(len(records) == expected_records, "aggregate feature record count drift")
    result: dict[str, object] = {
        "schema_version": FEATURE_LEDGER_SCHEMA,
        "claim_level": "ENGINEERING_VALIDATED_P_FEATURE_AGGREGATION_ONLY",
        "target_free": True,
        "source_role": "VALIDATED_FEATURE_SHARD_AGGREGATE",
        "source_file_sha256": str(manifest["logical_sha256"]),
        "source_logical_sha256": str(manifest["logical_sha256"]),
        "feature_schema": header["feature_schema"],
        "feature_schema_sha256": header["feature_schema_sha256"],
        "feature_dtype": header["feature_dtype"],
        "vector_scalar_max_ulps": header["vector_scalar_max_ulps"],
        "query_count": expected_query_count,
        "candidate_count_per_query": expected_candidate_count,
        "direction_count": len(P_DIRECTIONS),
        "record_count": expected_records,
        "query_candidate_axes": axes,
        "validated_feature_shards": provenance,
        "records": records,
        "protected_access_audit": dict(sorted(audit_totals.items())),
    }
    result["logical_sha256"] = feature_ledger_logical_sha256(result)
    feature_ledger_index(result)
    return result


def _feature_candidate_maps(
    ledger: Mapping[str, object],
) -> tuple[dict[str, dict[int, str]], dict[str, object]]:
    index = feature_ledger_index(ledger)
    candidates: dict[str, dict[int, str]] = defaultdict(dict)
    anchors: dict[str, object] = {}
    positions: dict[str, dict[int, tuple[int, str]]] = defaultdict(dict)
    for record in index.values():
        anchors.setdefault(record.query_id, record)
        prior = candidates[record.query_id].setdefault(
            record.candidate_physical_row, record.candidate_key
        )
        _require(prior == record.candidate_key, "physical row maps to two candidate keys")
        address = (record.candidate_physical_row, record.candidate_key)
        prior_address = positions[record.query_id].setdefault(
            record.candidate_position, address
        )
        _require(prior_address == address, "candidate position address drift")
    for query_id, mapping in candidates.items():
        _require(
            len(mapping) == int(ledger["candidate_count_per_query"]),
            f"feature candidate physical rows are not unique: {query_id}",
        )
    return dict(candidates), anchors


def _validate_postjoin(
    postjoin: Mapping[str, object],
    validation: Mapping[str, object],
    *,
    postjoin_file_sha256: str,
    expected_query_count: int,
) -> list[dict[str, object]]:
    records = postjoin.get("records")
    _require(
        postjoin.get("schema_version") == I0_POSTJOIN_SCHEMA
        and postjoin.get("status") == "RCDE_SR0_MT_I0_POSTJOIN_LEDGER_SEALED"
        and isinstance(records, list)
        and len(records) == expected_query_count
        and postjoin.get("record_sequence_sha256") == canonical_sha256(records)
        and postjoin.get("logical_sha256") == _logical(postjoin),
        "I0 postjoin ledger closure drift",
    )
    _require(
        validation.get("schema_version") == I0_POSTJOIN_VALIDATION_SCHEMA
        and validation.get("status") == I0_POSTJOIN_PASS
        and validation.get("errors") in (None, [])
        and validation.get("postjoin_ledger_sha256") == postjoin_file_sha256
        and validation.get("logical_sha256") == _logical(validation),
        "I0 postjoin independent validation/hash closure missing",
    )
    output: list[dict[str, object]] = []
    for expected_execution, raw in enumerate(records):
        _require(isinstance(raw, Mapping), "I0 postjoin record is not a mapping")
        item = dict(raw)
        _require(
            item.get("execution_ordinal") == expected_execution
            and item.get("record_sha256")
            == canonical_sha256({k: v for k, v in item.items() if k != "record_sha256"}),
            "I0 postjoin record address/hash drift",
        )
        output.append(item)
    return output


def _membership(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda item: int(item["execution_ordinal"]))
    query_ids = [str(item["query_id"]) for item in ordered]
    identities = sorted({str(item["identity"]) for item in ordered})
    supergroups = sorted({str(item["supergroup"]) for item in ordered})
    return {
        "query_count": len(ordered),
        "query_ids": query_ids,
        "query_ids_sha256": canonical_sha256(query_ids),
        "identity_count": len(identities),
        "identities_sha256": canonical_sha256(identities),
        "supergroup_count": len(supergroups),
        "supergroups_sha256": canonical_sha256(supergroups),
    }


def _assert_disjoint(
    train: Sequence[Mapping[str, object]], heldout: Sequence[Mapping[str, object]]
) -> None:
    for field in ("identity", "supergroup"):
        left = {str(item[field]) for item in train}
        right = {str(item[field]) for item in heldout}
        _require(not left & right, f"recomputed {field} leakage")


def _recipe() -> dict[str, object]:
    return {
        "seed": P_SEED,
        "optimizer": "AdamW",
        "learning_rate": P_LEARNING_RATE,
        "weight_decay": P_WEIGHT_DECAY,
        "gradient_clip_l2": P_GRADIENT_CLIP_L2,
        "updates_total": P_UPDATES_TOTAL,
        "episodes_per_update": P_EPISODES_PER_UPDATE,
        "warmup_updates": P_WARMUP_UPDATES,
        "final_learning_rate": P_FINAL_LEARNING_RATE,
        "checkpoint_interval_updates": P_CHECKPOINT_INTERVAL_UPDATES,
        "early_stop": False,
        "auxiliary_loss": False,
        "direction_fusion_before_one_pair_loss": True,
    }


def build_p_metadata_plan(
    *,
    postjoin: Mapping[str, object],
    postjoin_validation: Mapping[str, object],
    postjoin_file_sha256: str,
    postjoin_validation_file_sha256: str,
    feature_ledger: Mapping[str, object],
    feature_ledger_file_sha256: str,
    feature_validation_file_sha256: str,
    execution_protocol_sha256: str,
    p_implementation_sha256: str,
    expected_query_count: int,
    expected_candidate_count: int,
    execution_scope: str,
) -> dict[str, object]:
    """Build 12 inner and four outer pair/fit/lock-membership artifacts."""

    rows = _validate_postjoin(
        postjoin,
        postjoin_validation,
        postjoin_file_sha256=postjoin_file_sha256,
        expected_query_count=expected_query_count,
    )
    _require(
        feature_ledger.get("query_count") == expected_query_count
        and feature_ledger.get("candidate_count_per_query") == expected_candidate_count,
        "feature ledger population does not match I0 postjoin",
    )
    candidate_maps, anchors = _feature_candidate_maps(feature_ledger)
    _require(set(candidate_maps) == {str(item["query_id"]) for item in rows}, "I0/feature query population drift")
    by_identity: dict[str, set[int]] = defaultdict(set)
    by_supergroup: dict[str, set[int]] = defaultdict(set)
    for item in rows:
        query_id = str(item["query_id"])
        fold = int(item["outer_fold"])
        _require(fold in (1, 2, 3, 4), "I0 source fold drift")
        anchor = anchors[query_id]
        _require(
            anchor.execution_ordinal == item["execution_ordinal"]
            and anchor.query_source_image_sha256 == item["source_image_sha256"]
            and anchor.source_fold == fold,
            "I0/feature query addressing drift",
        )
        by_identity[str(item["identity"])].add(fold)
        by_supergroup[str(item["supergroup"])].add(fold)
        hit = item.get("target_hit")
        _require(isinstance(hit, bool), "I0 target-hit flag drift")
        if hit:
            target_row = int(item["target_physical_row"])
            rival_row = int(item["strongest_rival_physical_row"])
            _require(target_row != rival_row, "target and rival physical rows collide")
            _require(
                target_row in candidate_maps[query_id]
                and rival_row in candidate_maps[query_id],
                "I0 target/rival row absent from feature candidate axis; insertion forbidden",
            )
            item["target_candidate_key"] = candidate_maps[query_id][target_row]
            item["rival_candidate_key"] = candidate_maps[query_id][rival_row]
        else:
            _require(
                item.get("target_physical_row") is None
                and item.get("strongest_rival_physical_row") is None,
                "target miss exposes a physical pair",
            )
    _require(
        all(len(folds) == 1 for folds in by_identity.values())
        and all(len(folds) == 1 for folds in by_supergroup.values()),
        "identity/supergroup leaks across source folds",
    )
    common = {
        "i0_postjoin_ledger_file_sha256": postjoin_file_sha256,
        "i0_postjoin_validation_file_sha256": postjoin_validation_file_sha256,
        "i0_postjoin_record_sequence_sha256": postjoin["record_sequence_sha256"],
        "feature_ledger_file_sha256": feature_ledger_file_sha256,
        "feature_validation_file_sha256": feature_validation_file_sha256,
    }
    scopes: list[dict[str, object]] = []
    for outer in (1, 2, 3, 4):
        outer_train = [item for item in rows if int(item["outer_fold"]) != outer]
        outer_heldout = [item for item in rows if int(item["outer_fold"]) == outer]
        _assert_disjoint(outer_train, outer_heldout)
        definitions = [
            ("INNER_TRAIN_FIT", inner, [x for x in outer_train if int(x["outer_fold"]) != inner], [x for x in outer_train if int(x["outer_fold"]) == inner])
            for inner in sorted({1, 2, 3, 4} - {outer})
        ]
        definitions.append(("OUTER_TRAIN_REFIT", None, outer_train, outer_heldout))
        for fit_role, inner, train, lock_rows in definitions:
            _assert_disjoint(train, lock_rows)
            fit_id = f"P_OUTER{outer}_" + (f"INNER{inner}_FIT" if inner is not None else "OUTER_REFIT")
            eligible = [item for item in train if item["target_hit"] is True]
            episodes = [
                {
                    "query_id": item["query_id"],
                    "query_source_image_sha256": item["source_image_sha256"],
                    "execution_ordinal": item["execution_ordinal"],
                    "target_candidate_key": item["target_candidate_key"],
                    "rival_candidate_key": item["rival_candidate_key"],
                }
                for item in eligible
            ]
            train_membership = _membership(train)
            eligible_membership = _membership(eligible)
            lock_membership = _membership(lock_rows)
            membership_binding = canonical_sha256(
                {
                    "fit_id": fit_id,
                    "train": train_membership,
                    "eligible": eligible_membership,
                    "lock": lock_membership,
                    **common,
                }
            )
            pair_join: dict[str, object] = {
                "schema_version": PAIR_JOIN_SCHEMA,
                "fit_id": fit_id,
                "fit_role": fit_role,
                "outer_fold": outer,
                "inner_heldout_fold": inner,
                **common,
                "membership_binding_sha256": membership_binding,
                "pair_loss_excludes_target_misses_only": True,
                "target_insertion_count": 0,
                "candidate_mutation_count": 0,
                "episode_count": len(episodes),
                "episode_query_ids_sha256": canonical_sha256([item["query_id"] for item in episodes]),
                "episodes": episodes,
            }
            pair_join["logical_sha256"] = _logical(pair_join)
            scopes.append(
                {
                    "fit_id": fit_id,
                    "fit_role": fit_role,
                    "outer_fold": outer,
                    "inner_heldout_fold": inner,
                    "train_membership": train_membership,
                    "eligible_membership": eligible_membership,
                    "lock_membership": lock_membership,
                    "membership_binding_sha256": membership_binding,
                    "pair_join": pair_join,
                }
            )
    _require(
        sum(item["fit_role"] == "INNER_TRAIN_FIT" for item in scopes) == 12
        and sum(item["fit_role"] == "OUTER_TRAIN_REFIT" for item in scopes) == 4,
        "P metadata topology is not 4x3 plus four refits",
    )
    plan: dict[str, object] = {
        "schema_version": P_METADATA_PLAN_SCHEMA,
        "claim_level": "ENGINEERING_P_METADATA_DAG_ONLY",
        "query_count": expected_query_count,
        "candidate_count_per_query": expected_candidate_count,
        "execution_scope": execution_scope,
        **common,
        "source_membership_sha256": canonical_sha256(
            [
                {
                    "query_id": item["query_id"],
                    "execution_ordinal": item["execution_ordinal"],
                    "outer_fold": item["outer_fold"],
                    "identity": item["identity"],
                    "supergroup": item["supergroup"],
                    "target_hit": item["target_hit"],
                }
                for item in rows
            ]
        ),
        "target_hit_count": sum(item["target_hit"] is True for item in rows),
        "target_miss_count": sum(item["target_hit"] is False for item in rows),
        "inner_fit_count": 12,
        "outer_refit_count": 4,
        "scopes": scopes,
        "automatic_stage_advance": False,
        "next_authorized_stage": None,
    }
    plan["logical_sha256"] = _logical(plan)
    return plan


def finalize_scope_documents(
    scope: Mapping[str, object],
    *,
    pair_join_file_sha256: str,
    feature_ledger_file_sha256: str,
    execution_protocol_sha256: str,
    p_implementation_sha256: str,
    execution_scope: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Create a ready FIT manifest and pre-fit lock membership manifest."""

    eligible_ids = list(scope["eligible_membership"]["query_ids"])  # type: ignore[index]
    # Source hashes/ordinals are carried by the pair join in the exact runner order.
    episodes = scope["pair_join"]["episodes"]  # type: ignore[index]
    _require(
        eligible_ids == [str(item["query_id"]) for item in episodes],
        "eligible membership/pair order drift",
    )
    fit: dict[str, object] = {
        "schema_version": FIT_MANIFEST_SCHEMA,
        "fit_id": scope["fit_id"],
        "fit_role": scope["fit_role"],
        "outer_fold": scope["outer_fold"],
        "inner_heldout_fold": scope["inner_heldout_fold"],
        "execution_scope": execution_scope,
        "identity_disjoint": True,
        "supergroup_disjoint": True,
        "membership_binding_sha256": scope["membership_binding_sha256"],
        "train_membership": scope["train_membership"],
        "eligible_membership": scope["eligible_membership"],
        "recipe": _recipe(),
        "bindings": {
            "feature_ledger_file_sha256": feature_ledger_file_sha256,
            "pair_join_file_sha256": pair_join_file_sha256,
            "execution_protocol_sha256": execution_protocol_sha256,
            "p_implementation_sha256": p_implementation_sha256,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        },
        "eligible_population_sha256": canonical_sha256(
            [
                {
                    "query_id": item["query_id"],
                    "source_image_sha256": item["query_source_image_sha256"],
                    "execution_ordinal": item["execution_ordinal"],
                }
                for item in episodes
            ]
        ),
    }
    fit["logical_sha256"] = _logical(fit)
    lock: dict[str, object] = {
        "schema_version": LOCK_MEMBERSHIP_MANIFEST_SCHEMA,
        "fit_id": scope["fit_id"],
        "crossfit_role": (
            "INNER_HELDOUT_OOF"
            if scope["fit_role"] == "INNER_TRAIN_FIT"
            else "OUTER_HELDOUT_DEPLOYMENT"
        ),
        "outer_fold": scope["outer_fold"],
        "inner_heldout_fold": scope["inner_heldout_fold"],
        "query_ids": scope["lock_membership"]["query_ids"],  # type: ignore[index]
        "query_ids_sha256": scope["lock_membership"]["query_ids_sha256"],  # type: ignore[index]
        "membership": scope["lock_membership"],
        "membership_binding_sha256": scope["membership_binding_sha256"],
        "feature_ledger_file_sha256": feature_ledger_file_sha256,
        "p_training_manifest_logical_sha256": fit["logical_sha256"],
        "checkpoint_binding_state": "DEFERRED_UNTIL_FIT_COMPLETE",
        "p_checkpoint_file_sha256": None,
        "target_free_complete_query_population": True,
        "automatic_stage_advance": False,
    }
    lock["logical_sha256"] = _logical(lock)
    return fit, lock


def finalize_lock_manifest(
    *,
    membership: Mapping[str, object],
    membership_file_sha256: str,
    fit_manifest: Mapping[str, object],
    fit_manifest_file_sha256: str,
    checkpoint: Mapping[str, object],
    checkpoint_file_sha256: str,
    fit_validation: Mapping[str, object],
    fit_validation_file_sha256: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Checkpoint-bind a result-blind membership after independent fit PASS."""

    _require(
        membership.get("schema_version") == LOCK_MEMBERSHIP_MANIFEST_SCHEMA
        and membership.get("logical_sha256") == _logical(membership)
        and membership.get("checkpoint_binding_state")
        == "DEFERRED_UNTIL_FIT_COMPLETE"
        and membership.get("p_checkpoint_file_sha256") is None,
        "lock membership manifest is not a valid result-blind artifact",
    )
    _require(
        fit_manifest.get("schema_version") == FIT_MANIFEST_SCHEMA
        and fit_manifest.get("logical_sha256") == _logical(fit_manifest)
        and membership.get("fit_id") == fit_manifest.get("fit_id")
        and membership.get("p_training_manifest_logical_sha256")
        == fit_manifest.get("logical_sha256"),
        "lock membership/fit manifest closure drift",
    )
    expected_fit_role = (
        "INNER_TRAIN_FIT"
        if membership.get("crossfit_role") == "INNER_HELDOUT_OOF"
        else "OUTER_TRAIN_REFIT"
    )
    _require(
        fit_manifest.get("fit_role") == expected_fit_role
        and fit_manifest.get("outer_fold") == membership.get("outer_fold")
        and fit_manifest.get("inner_heldout_fold")
        == membership.get("inner_heldout_fold"),
        "lock membership/fit fold role drift",
    )
    _require(
        checkpoint.get("schema_version") == CHECKPOINT_SCHEMA
        and checkpoint.get("completed_updates") == P_UPDATES_TOTAL
        and checkpoint.get("fit_manifest_file_sha256") == fit_manifest_file_sha256
        and checkpoint.get("fit_role") == expected_fit_role
        and checkpoint.get("outer_fold") == membership.get("outer_fold")
        and checkpoint.get("inner_heldout_fold")
        == membership.get("inner_heldout_fold"),
        "P checkpoint does not close the requested lock membership",
    )
    _require(
        fit_validation.get("schema_version") == FIT_VALIDATION_SCHEMA
        and fit_validation.get("status") == FIT_VALIDATION_PASS
        and fit_validation.get("validation_pass") is True
        and fit_validation.get("fresh_resume_exact") is True
        and fit_validation.get("checkpoint_sha256") == checkpoint_file_sha256
        and fit_validation.get("outer_fold") == membership.get("outer_fold")
        and fit_validation.get("inner_heldout_fold")
        == membership.get("inner_heldout_fold")
        and fit_validation.get("logical_sha256") == _logical(fit_validation),
        "independent P fit PASS/hash/fold closure missing",
    )
    final: dict[str, object] = {
        "schema_version": LOCK_MANIFEST_SCHEMA,
        "crossfit_role": membership["crossfit_role"],
        "outer_fold": membership["outer_fold"],
        "inner_heldout_fold": membership["inner_heldout_fold"],
        "bindings": {
            "feature_ledger_file_sha256": membership[
                "feature_ledger_file_sha256"
            ],
            "p_checkpoint_file_sha256": checkpoint_file_sha256,
            "p_training_manifest_file_sha256": fit_manifest_file_sha256,
        },
        "query_ids": membership["query_ids"],
        "query_ids_sha256": membership["query_ids_sha256"],
        "membership_binding_sha256": membership["membership_binding_sha256"],
        "lock_membership_manifest_file_sha256": membership_file_sha256,
        "fit_validation_file_sha256": fit_validation_file_sha256,
        "checkpoint_binding_state": "INDEPENDENT_FIT_PASS_BOUND",
        "automatic_stage_advance": False,
    }
    final["logical_sha256"] = _logical(final)
    receipt: dict[str, object] = {
        "schema_version": LOCK_MANIFEST_FINALIZATION_RECEIPT_SCHEMA,
        "status": "RCDE_SR0_MT_P_LOCK_MANIFEST_FINALIZED_AFTER_FIT_PASS",
        "fit_id": membership["fit_id"],
        "lock_membership_manifest_file_sha256": membership_file_sha256,
        "fit_manifest_file_sha256": fit_manifest_file_sha256,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "fit_validation_file_sha256": fit_validation_file_sha256,
        "final_lock_manifest_logical_sha256": final["logical_sha256"],
        "next_authorized_stage": None,
        "automatic_stage_advance": False,
    }
    receipt["logical_sha256"] = _logical(receipt)
    return final, receipt


def build_oof_concat_manifests(
    *,
    metadata_plan: Mapping[str, object],
    input_manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    """Validate 12 inner lock ledgers and build four exact OOF concat manifests."""

    _require(
        metadata_plan.get("schema_version") == P_METADATA_PLAN_SCHEMA
        and metadata_plan.get("logical_sha256") == _logical(metadata_plan),
        "P metadata plan closure drift",
    )
    _require(
        input_manifest.get("schema_version") == OOF_LOCK_INPUT_MANIFEST_SCHEMA
        and input_manifest.get("logical_sha256") == _logical(input_manifest),
        "OOF lock input manifest closure drift",
    )
    entries = input_manifest.get("locks")
    _require(isinstance(entries, list) and len(entries) == 12, "OOF lock input count drift")
    observed: dict[tuple[int, int], dict[str, object]] = {}
    for raw in entries:
        _require(isinstance(raw, Mapping), "OOF lock input is not a mapping")
        ledger_path = _safe_path(raw.get("lock_ledger_path"), name="OOF lock ledger")
        validation_path = _safe_path(raw.get("validation_path"), name="OOF lock validation")
        ledger_sha = str(raw.get("lock_ledger_file_sha256"))
        validation_sha = str(raw.get("validation_file_sha256"))
        _require(file_sha256(ledger_path) == ledger_sha, "OOF lock ledger hash drift")
        _require(file_sha256(validation_path) == validation_sha, "OOF lock validation hash drift")
        ledger = load_json_mapping(ledger_path, name="inner OOF P lock ledger")
        validation = load_json_mapping(validation_path, name="inner OOF lock validation")
        _require(
            ledger.get("schema_version") == LOCK_LEDGER_SCHEMA
            and ledger.get("crossfit_role") == "INNER_HELDOUT_OOF"
            and ledger.get("target_free") is True
            and validation.get("status") == LOCK_VALIDATION_PASS
            and validation.get("validation_pass") is True
            and validation.get("lock_ledger_file_sha256") == ledger_sha
            and validation.get("logical_sha256") == _logical(validation),
            "OOF independent lock validation/hash closure missing",
        )
        outer = int(ledger.get("outer_fold", -1))
        inner = int(ledger.get("inner_heldout_fold", -1))
        key = (outer, inner)
        _require(
            outer in (1, 2, 3, 4) and inner in ({1, 2, 3, 4} - {outer})
            and key not in observed,
            "OOF fold topology duplicate/drift",
        )
        metadata = ledger.get("query_metadata")
        _require(isinstance(metadata, list), "OOF lock query metadata absent")
        query_ids = [str(item["query_id"]) for item in metadata if isinstance(item, Mapping)]
        _require(
            len(query_ids) == ledger.get("query_count") == len(set(query_ids)),
            "OOF lock query population duplicate/count drift",
        )
        observed[key] = {
            "outer_fold": outer,
            "inner_heldout_fold": inner,
            "lock_ledger_file_sha256": ledger_sha,
            "lock_validation_file_sha256": validation_sha,
            "lock_manifest_file_sha256": ledger.get("p_lock_manifest_file_sha256"),
            "p_checkpoint_file_sha256": ledger.get("p_checkpoint_file_sha256"),
            "query_ids": query_ids,
            "query_ids_sha256": canonical_sha256(query_ids),
        }
    scopes = metadata_plan.get("scopes")
    _require(isinstance(scopes, list), "P metadata plan scopes absent")
    expected = {
        (int(item["outer_fold"]), int(item["inner_heldout_fold"])): list(item["lock_membership"]["query_ids"])
        for item in scopes
        if isinstance(item, Mapping) and item.get("fit_role") == "INNER_TRAIN_FIT"
    }
    _require(set(observed) == set(expected), "OOF inputs do not close the 4x3 topology")
    outputs = []
    for outer in (1, 2, 3, 4):
        source_entries = [observed[(outer, inner)] for inner in sorted({1, 2, 3, 4} - {outer})]
        concatenated: list[str] = []
        for item in source_entries:
            key = (outer, int(item["inner_heldout_fold"]))
            _require(item["query_ids"] == expected[key], "OOF lock membership differs from recomputed I0 scope")
            concatenated.extend(item["query_ids"])
        _require(len(concatenated) == len(set(concatenated)), "OOF concatenation duplicates queries")
        outer_expected = sorted(
            (item for item in scopes if isinstance(item, Mapping) and item.get("fit_role") == "OUTER_TRAIN_REFIT" and item.get("outer_fold") == outer),
            key=lambda item: str(item["fit_id"]),
        )
        _require(len(outer_expected) == 1, "outer refit membership absent")
        expected_train = list(outer_expected[0]["train_membership"]["query_ids"])
        _require(set(concatenated) == set(expected_train), "OOF concat is not exact outer-train population")
        ordered = sorted(concatenated, key=lambda query_id: expected_train.index(query_id))
        value: dict[str, object] = {
            "schema_version": OOF_LOCK_CONCAT_SCHEMA,
            "outer_fold": outer,
            "crossfit_role": "CONCATENATED_INNER_OOF_P_LOCKS_ONLY",
            "source_lock_count": 3,
            "source_locks": [{k: v for k, v in item.items() if k != "query_ids"} for item in source_entries],
            "query_count": len(ordered),
            "query_ids": ordered,
            "query_ids_sha256": canonical_sha256(ordered),
            "metadata_plan_logical_sha256": metadata_plan["logical_sha256"],
            "i0_postjoin_ledger_file_sha256": metadata_plan["i0_postjoin_ledger_file_sha256"],
            "duplicate_query_count": 0,
            "missing_query_count": 0,
            "automatic_stage_advance": False,
        }
        value["logical_sha256"] = _logical(value)
        outputs.append(value)
    return outputs


__all__ = [
    "FEATURE_SHARD_MANIFEST_SCHEMA",
    "FEATURE_AGGREGATION_RESULT_SCHEMA",
    "FEATURE_AGGREGATE_VALIDATION_SCHEMA",
    "FEATURE_AGGREGATE_VALIDATION_PASS",
    "P_METADATA_PLAN_SCHEMA",
    "LOCK_MEMBERSHIP_MANIFEST_SCHEMA",
    "OOF_LOCK_INPUT_MANIFEST_SCHEMA",
    "OOF_LOCK_CONCAT_SCHEMA",
    "OOF_LOCK_CONCAT_VALIDATION_SCHEMA",
    "LOCK_MANIFEST_FINALIZATION_RECEIPT_SCHEMA",
    "aggregate_validated_feature_shards",
    "invocation_execution_manifest_sha256",
    "build_p_metadata_plan",
    "finalize_scope_documents",
    "finalize_lock_manifest",
    "build_oof_concat_manifests",
]
