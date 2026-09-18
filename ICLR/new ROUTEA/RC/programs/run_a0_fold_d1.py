"""Formal A0 prerequisite: fresh fold-local D1 substrate materialization.

The historical D1 source is imported read-only.  This wrapper changes only the
runtime fold selector before using its frozen recipe; it does not mutate that
source, historical outputs, HOME, or any opened/sealed data.  The result is a
train-only receipt, not an ownership claim and not a local-model endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[3]
ROUTEA = ROOT.parent
UPSTREAMS = ROOT / "registry" / "upstream_inputs.json"
OUT_ROOT = ROOT / "results" / "a0_optimization987_fivefold_oof_v1"
HISTORICAL_RUNNER = ROUTEA / "programs" / "run_routea_v3_1_c6direct_m1.py"


class A0D1Error(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_runner() -> Any:
    core = ROUTEA / "route_a_core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    spec = importlib.util.spec_from_file_location("rc_a0_historical_d1_runner", HISTORICAL_RUNNER)
    if spec is None or spec.loader is None:
        raise A0D1Error("cannot import frozen D1 support runner")
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolves postponed annotations through sys.modules while the
    # imported historical runner is being defined.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_fold(module: Any, fold: int) -> None:
    if fold not in range(5):
        raise A0D1Error("fold must be in 0..4")
    # The source expressly routes train/heldout data through module.FOLD and
    # builds its schedule through d1.FOLD.  This local runtime binding is the
    # only permitted fold variation; all other recipe constants remain frozen.
    module.FOLD = fold
    module.d1.FOLD = fold
    # `run()` performs a preflight dry-run before it configures the live
    # training module.  `route_a.o1_c6direct_m1_d1` is process-global, so a
    # second import sees that same module.  Preserve the pristine historical
    # function exactly once; wrapping an earlier wrapper would recurse.
    original_schedule = getattr(module.d1, "_rc_a0_pristine_schedule", None)
    if original_schedule is None:
        original_schedule = module.d1.build_three_track_schedule
        setattr(module.d1, "_rc_a0_pristine_schedule", original_schedule)

    def fold_effective_seed(*, seed: int = 17, fold: int | None = None) -> int:
        if seed != 17 or (fold is not None and fold != module.FOLD):
            raise ValueError("A0 D1 only permits base seed 17 and its assigned fold")
        return 17 + module.d1.FOLD_SEED_STRIDE * module.FOLD

    def fold_schedule(rows_by_track: Any, *, seed: int = 17, fold: int | None = None) -> Any:
        if seed != 17 or (fold is not None and fold != module.FOLD):
            raise ValueError("A0 D1 schedule may only use base seed 17 and assigned fold")
        return original_schedule(rows_by_track, seed=17, fold=module.FOLD)

    # Defaults in the old screen source were lexically bound to fold 0.  The
    # wrapper makes the one authorized A0 fold dimension explicit while keeping
    # every training constant and all schedule semantics unchanged.
    module.d1.effective_seed = fold_effective_seed
    module.d1.build_three_track_schedule = fold_schedule
    if module.d1.SEED != 17 or module.d1.effective_seed() != 17 + 1009 * fold:
        raise A0D1Error("frozen D1 fold seed contract failed")


def preflight_for_fold(fold: int) -> dict[str, Any]:
    upstream = json.loads(UPSTREAMS.read_text())
    path = Path(upstream["primary_line"]["identity_supergroup_preflight"]["path"])
    value = json.loads(path.read_text())
    if value.get("seed") != 17 or len(value.get("folds", [])) != 5:
        raise A0D1Error("fivefold preflight schema drift")
    entry = value["folds"][fold]
    if entry.get("fold") != fold or not entry.get("train_identities") or not entry.get("heldout_identities"):
        raise A0D1Error("requested fold has no legal identity split")
    return value


def dry_run(fold: int) -> dict[str, Any]:
    module = import_runner()
    configure_fold(module, fold)
    preflight = preflight_for_fold(fold)
    rows = module.load_query_rows(preflight)
    train = [row for row in rows if not row.heldout]
    heldout = [row for row in rows if row.heldout]
    expected = preflight["folds"][fold]
    if len(train) != expected["train_query_count"] or len(heldout) != expected["heldout_query_count"]:
        raise A0D1Error("fold query routing mismatch")
    tracks = {track: sum(row.track == track for row in train) for track in ("outcome", "difficult", "new_difficult_train")}
    if any(count == 0 for count in tracks.values()):
        raise A0D1Error("D1 schedule lacks a required training track")
    return {
        "status": "A0_D1_FOLD_PREFLIGHT_READY",
        "fold": fold,
        "effective_seed": module.d1.effective_seed(),
        "train_queries": len(train),
        "heldout_queries": len(heldout),
        "train_track_counts": tracks,
        "history_source_sha256": sha256(HISTORICAL_RUNNER),
        "opened_runtime_read_count": 0,
        "sealed_runtime_read_count": 0,
        "home_files_modified": 0,
    }


def run(fold: int) -> dict[str, Any]:
    check = dry_run(fold)
    module = import_runner()
    configure_fold(module, fold)
    preflight = preflight_for_fold(fold)
    fold_dir = OUT_ROOT / f"fold_{fold}"
    work_dir = ROOT / "tmp" / "a0_d1" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / "result.json").exists():
        prior = json.loads((fold_dir / "result.json").read_text())
        if prior.get("status") == "A0_D1_FOLD_TRAINING_COMPLETE":
            return prior
        raise A0D1Error("existing non-complete fold output blocks overwrite")
    module._runtime_contract()
    rows = module.load_query_rows(preflight)
    query_cache = module.cache_optimization_queries(rows)
    gallery = module.load_gallery(preflight)
    legal = module._legal_mask(preflight, gallery)
    device = torch.device("cuda")
    references = gallery.reference_tokens_cpu.float().to(device)
    reference_mask = gallery.reference_mask_cpu.to(device)
    materialization = json.loads(Path(json.loads(UPSTREAMS.read_text())["primary_line"]["optimization_materialization"]["path"]).read_text())
    receipt = {
        "file_sha256": sha256(Path(json.loads(UPSTREAMS.read_text())["primary_line"]["optimization_materialization"]["path"])),
        "logical_sha256": materialization["logical_sha256"],
    }
    adapter, d1_run, schedule = module.train_fresh_d1(
        rows, query_cache, gallery, references, reference_mask, legal, preflight, receipt, fold_dir, work_dir=work_dir,
    )
    result = {
        **check,
        "status": "A0_D1_FOLD_TRAINING_COMPLETE",
        "claim_level": "train_only_fold_local_D1_substrate_not_local_ownership_endpoint",
        "d1_checkpoint": str((fold_dir / "d1_checkpoint.pt").resolve()),
        "d1_checkpoint_sha256": sha256(fold_dir / "d1_checkpoint.pt"),
        "d1_history_sha256": sha256(fold_dir / "d1_history.json"),
        "schedule_sha256": d1_run["schedule_receipt"]["logical_sha256"],
        "d1_recipe_sha256": module.d1.canonical_sha256(module.d1.frozen_recipe()),
        "d1_parameter_count": sum(parameter.numel() for parameter in adapter.parameters()),
        "next_authorized_stage": "A0_LOCAL_HYPOTHESIS_TRAINING_ONLY_AFTER_ALL_FIVE_D1_FOLDS_COMPLETE",
    }
    (fold_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outcome = dry_run(args.fold) if args.dry_run else run(args.fold)
    print(json.dumps(outcome, sort_keys=True))


if __name__ == "__main__":
    main()
