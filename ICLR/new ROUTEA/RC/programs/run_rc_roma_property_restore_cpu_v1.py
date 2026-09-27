#!/usr/bin/env python3
"""Frozen CPU-only 18-arm property-restoration acquisition from retained A/J/P.

``prepare`` freezes the already sealed target/fixed-wrong diagnostic positions.
It does not select new candidates using outcomes. ``worker`` requires
an explicit execution gate written after the separate CPU pilot is reviewed.
Every scientific arm, including historical controls, is recomputed on CPU.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "programs"), str(ROOT / "src")]
OUT = ROOT / "results/rc_m_property_restoration_factorial_v1"
PLAN = ROOT / "plan/RC_M_PROPERTY_RESTORATION_FACTORIAL_V1_20260927.md"
SOURCE = ROOT / "results/rc_roma_position_factor_v1"
PROTOCOL = OUT / "protocol.json"
GATE = OUT / "execution_gate.json"
HISTORICAL_ANALYSIS = ROOT / "results/rc_mass_discrimination_source_v1/result.json"


def configure(threads):
    assert threads == 4, "FROZEN_FOUR_THREAD_BACKEND"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch


def modules():
    import probe_rc_roma_property_restore_cpu_v3 as helpers
    import rc_roma_property_restore_v1 as transform
    return helpers, transform


def cpu_environment(torch):
    return dict(torch=str(torch.__version__), python=sys.version,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                cpu_capability=torch.backends.cpu.get_cpu_capability(),
                mkldnn_enabled=torch.backends.mkldnn.enabled,
                autocast="Original DPT CPU BF16; final output_conv2 FP32",
                input_dtypes="A/P FP32; losslessly stored J restored to original FP32",
                pooling_dtype="sigmoid FP32 then original integer-cell FP64 mean",
                torch_config=torch.__config__.show(),
                omp_num_threads=os.environ["OMP_NUM_THREADS"],
                mkl_num_threads=os.environ["MKL_NUM_THREADS"])


def prepare(args, torch):
    h, t = modules()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = SOURCE / "manifest.json"
    manifest = h.read(manifest_path)
    old_pilot = h.read(SOURCE / "pilot_validation.json")
    h.checked(old_pilot["head_state"])
    authority = old_pilot["authority"]
    old_authority = h.read(h.checked(authority))
    assert old_authority["manifest"] == h.binding(manifest_path)
    historical = h.read(HISTORICAL_ANALYSIS)
    fixed_rows = {int(r["index"]): r for r in historical["phase_rows"]}
    assert set(fixed_rows) == set(t.PANEL)
    workers = []
    for index in t.PANEL:
        w = next(w for w in manifest["workers"] if w["execution_ordinal"] == index)
        path = SOURCE / f"query{index:03d}/gpu_validation.json"
        value = h.read(path)
        assert value["status"] == "POSITION_FACTOR_ALL128_PASS" and value["authority"] == authority
        axis = value["candidate_physical_rows"]
        assert len(axis) == len(set(axis)) == len(value["pairs"]) == 128
        assert value["index"] == index and value["query_id"] == w["query_id"]
        frozen = fixed_rows[index]
        assert frozen["query_id"] == w["query_id"] and frozen["axis"] == axis
        if frozen["target_present"]:
            target = int(frozen["target_position"])
            wrong = int(frozen["arm_metrics"]["NATIVE"]["strongest_position"])
            assert 0 <= target < 128 and 0 <= wrong < 128 and target != wrong
            positions = sorted([target, wrong])
        else:
            positions = []
        workers.append(dict(index=index, query_id=w["query_id"], candidate_physical_rows=axis,
                            positions=positions, target_present=bool(frozen["target_present"]),
                            gpu_validation=h.binding(path), input_worker=w))
    assert sum(len(w["positions"]) for w in workers) == 14
    assert [w["index"] for w in workers if not w["positions"]] == [7]
    code = [Path(__file__), Path(h.__file__), Path(t.__file__),
            ROOT / "programs/join_rc_roma_property_restore_v1.py",
            ROOT / "slurm/rc_m_property_restore_cpu_v1.sbatch",
            ROOT / "slurm/rc_m_property_restore_join_v1.sbatch",
            ROOT / "programs/rc_roma_position_factor_v1.py",
            ROOT / "programs/rc_roma_shared_native_cache_v1.py",
            ROOT / "programs/rc_roma_exact_feature_cache_v2.py",
            ROOT / "src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py",
            h.ROMA / "dpt.py", h.ROMA / "matcher.py"]
    algebra = t.self_test()
    h.write_once(OUT / "algebra.json", algebra)
    protocol = dict(
        schema="M_PROPERTY_RESTORATION_FACTORIAL_CPU_V1",
        scope="PRIMARY_FIXED_PAIRS",
        evaluation_scope="PRIMARY_FIXED_PAIRS",
        indices=list(t.PANEL), arms=list(t.ARMS), phase_arms=list(t.PHASE_ARMS),
        seed=t.SEED, delta=t.DELTA, candidates_per_query=128,
        threads=args.threads, interop_threads=1, radius=manifest["radius"],
        code_sources=[h.binding(p) for p in code], plan=h.binding(PLAN),
        sources=dict(source_authority=authority, source_manifest=h.binding(manifest_path),
                     source_head=old_pilot["head_state"],
                     historical_analysis=h.binding(HISTORICAL_ANALYSIS)),
        algebra=h.binding(OUT / "algebra.json"), workers=workers,
        runtime={k: v for k, v in cpu_environment(torch).items() if k != "cpu_capability"},
        backend_policy="All 18 scientific arms use one CPU backend. Historical GPU outputs are descriptive references only.",
        qualification_policy="execution_gate.json must explicitly authorize the reviewed separate pilot; no automatic numerical tolerance relaxation.",
        label_policy="Preparation uses only the historical sealed phase_rows target_position/target_present and NATIVE strongest_position plus query/axis identifiers. This is a retrospective mechanism panel, not a label-blind candidate selection or accuracy evaluation. Workers do not reopen target labels.",
        scope_boundary="Only 14 preselected pairs are recomputed. Complete original C128 axes are validated; no all-candidate ranking, action accuracy or 128-candidate mean-scale claim is made.",
        endpoint="COARSE confidence -> exact token pooling -> M. No matcher/refiner/encoder/LLM forward.",
        numerical_policy="Finite outputs and deterministic same-CPU native replay are required. CPU/GPU differences are retained without approximate PASS thresholds.",
        scientific_boundary="Conditional restoration of amplitude assignment and phase coherence; not ground-truth geometry recovery or unique causality.")
    h.write_once(PROTOCOL, protocol)
    h.write_once(OUT / "prepare_validation.json", dict(
        status="PROPERTY_RESTORE_CPU_PREPARATION_PASS", protocol=h.binding(PROTOCOL),
        algebra=h.binding(OUT / "algebra.json"), workers=len(workers), arms=len(t.ARMS),
        candidate_pairs=14, original_candidate_axes=len(workers) * 128,
        historical_fixed_diagnostic_positions_read=True, new_outcome_selection=False,
        execution_released=False))
    h.emit(stage="PREPARED", protocol=h.binding(PROTOCOL), arms=len(t.ARMS), queries=len(workers))


def guard(index, threads, torch):
    h, t = modules()
    p = h.read(PROTOCOL)
    assert p["schema"] == "M_PROPERTY_RESTORATION_FACTORIAL_CPU_V1"
    assert p["arms"] == list(t.ARMS) and p["indices"] == list(t.PANEL)
    assert index in p["indices"] and threads == p["threads"] == 4
    for b in p["code_sources"] + [p["plan"], p["algebra"], p["sources"]["source_authority"],
                                   p["sources"]["source_manifest"], p["sources"]["source_head"]]:
        h.checked(b)
    assert h.read(OUT / "prepare_validation.json")["protocol"] == h.binding(PROTOCOL)
    gate = h.read(GATE)
    assert gate["status"] == "CPU_PROPERTY_RESTORE_EXECUTION_AUTHORIZED"
    assert gate["protocol"] == h.binding(PROTOCOL)
    assert gate["threads"] == threads and gate["science_backend"] == "CPU_ALL18"
    qualification = h.read(h.checked(gate["qualification"]))
    assert qualification["status"].startswith("CPU_DPT_PILOT_MEASURED_")
    assert qualification["native_repeat_bit_exact"] and qualification["cpu_pool_integer_bounds_bit_exact"]
    current = cpu_environment(torch)
    # Numerical runtime and instruction capability are frozen across resumptions.
    # Hostname/job ID are deliberately absent: they are scheduling metadata.
    assert {k: v for k, v in current.items() if k != "cpu_capability"} == p["runtime"], ("CPU_RUNTIME_DRIFT", current, p["runtime"])
    qualified_environment = h.read(h.checked(qualification["environment"]))
    assert current["cpu_capability"] == qualified_environment["cpu_capability"], "QUALIFIED_CPU_CAPABILITY_DRIFT"
    w = next(w for w in p["workers"] if w["index"] == index)
    old = h.read(h.checked(w["gpu_validation"]))
    assert old["candidate_physical_rows"] == w["candidate_physical_rows"]
    return h, t, p, w, old, gate


def load_geometries(worker, torch, h):
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2

    w = worker["input_worker"]
    source = w["roma"]["payload"]
    value = torch.load(h.checked(source), map_location="cpu", weights_only=True, mmap=True)
    q = value["records"][w["source_index"]]
    assert q["query_id"] == w["source_query_id"]
    assert q["query_source_sha256"] == w["source_image_sha256"]
    assert q["candidate_physical_rows"] == worker["candidate_physical_rows"]

    def one(meta, sha, key):
        geom = build_colnomic_canonical_geometry_v2(
            source_image_sha256=sha, source_key=key,
            processor_config_sha256=meta["processor_config_sha256"],
            raw_size_hw=tuple(meta["raw_size_hw"]), exif_orientation=int(meta["exif_orientation"]),
            merged_grid_shape=tuple(meta["grid_shape"]), processor_input_frame=meta["processor_input_frame"])
        assert tuple(geom.oriented_size_hw) == tuple(meta["oriented_size_hw"])
        assert len(geom.cell_boxes_xyxy) == math.prod(meta["grid_shape"])
        return geom

    index = worker["index"]
    qmeta = q["query_geometry"]
    qsha = q["query_source_sha256"]
    qgeom = one(qmeta, qsha, f"property-restore:q{index:03d}")
    rows = []
    for position, candidate in enumerate(q["candidates"]):
        assert candidate["physical_row"] == worker["candidate_physical_rows"][position]
        meta, sha = candidate["reference_geometry"], candidate["reference_image_sha256"]
        rows.append(dict(query_sha=qsha, reference_sha=sha, metadata=[qmeta, meta],
                         geometries=[qgeom, one(meta, sha, f"property-restore:q{index:03d}:r{position:03d}")]))
    assert len(rows) == 128
    return rows, source


def worker(args, torch):
    started = time.monotonic()
    h, t, protocol, work, old_validation, gate = guard(args.index, args.threads, torch)
    folder = OUT / f"query{args.index:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_binding = h.binding(PROTOCOL)
        gate_binding = h.binding(GATE)
        if (folder / "validation.json").exists():
            validation = h.read(folder / "validation.json")
            assert validation["protocol"] == protocol_binding and validation["status"] == "PROPERTY_RESTORE_CPU_QUERY_PASS"
            for b in validation["pairs"]:
                h.checked(b)
            h.emit(stage="QUERY_ALREADY_COMPLETE", index=args.index)
            return 0
        h.write_once(folder / "backend.json", cpu_environment(torch))
        backend_binding = h.binding(folder / "backend.json")
        if not work["positions"]:
            assert not work["target_present"]
            h.write_once(folder / "validation.json", dict(
                status="PROPERTY_RESTORE_CPU_QUERY_PASS", protocol=protocol_binding,
                backend_binding=backend_binding, index=args.index, query_id=work["query_id"],
                candidate_physical_rows=work["candidate_physical_rows"], positions=[], arms=list(t.ARMS),
                pairs=[], scope="PRIMARY_FIXED_PAIRS", target_absent_from_natural_axis=True,
                reason="No sealed target/fixed-wrong contrast; no new pair computed and no candidate inserted.",
                all_arms_same_cpu_backend=True, native_repeat_receipts=[],
                original_gpu_bit_exact_not_claimed=True, source_validation=work["gpu_validation"]))
            h.emit(stage="TARGET_ABSENT_AXIS_RECORDED", index=args.index)
            return 0
        head_record = torch.load(h.checked(protocol["sources"]["source_head"]), map_location="cpu", weights_only=True, mmap=True)
        DPTHead = h.load_cpu_dpt(torch)
        head = DPTHead(**head_record["config"]).cpu().eval()
        head.load_state_dict(head_record["head"], strict=True)
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        omega, scale = head_record["omega"], head_record["scale"]
        from rc_roma_shared_native_cache_v1 import ExactPool
        pool = ExactPool()
        geometry_rows, geometry_source = load_geometries(work, torch, h)
        descriptor_cache = OrderedDict()
        measured_arms = 0
        completed_pairs = []
        native_repeat_done = False
        run_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.environ.get('SLURM_RESTART_COUNT', '0')}_{os.getpid()}"
        run_record = dict(protocol=protocol_binding, backend=backend_binding, gate=gate_binding,
                          index=args.index, budget_seconds=args.budget, run_id=run_id)

        def descriptor(record):
            key = record["sha256"]
            if key not in descriptor_cache:
                value = torch.load(h.checked(record), map_location="cpu", weights_only=True, mmap=True)
                descriptor_cache[key] = [x.clone() for x in value["features"]]
                if len(descriptor_cache) > 4:
                    descriptor_cache.popitem(last=False)
            descriptor_cache.move_to_end(key)
            return descriptor_cache[key]

        def finish(status, exit_code):
            receipt = dict(run_record, status=status, newly_measured_arms=measured_arms,
                           completed_pairs_this_scan=len(completed_pairs),
                           native_repeat_done=native_repeat_done,
                           elapsed_seconds=time.monotonic()-started,
                           max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            h.write_once(folder / "runs" / f"{run_id}.json", receipt)
            h.emit(stage=status, **{k: v for k, v in receipt.items() if k not in ("protocol", "gate", "backend")})
            return exit_code

        with torch.inference_mode():
            for position in work["positions"]:
                pair_dir = folder / f"pair{position:03d}"
                receipt_path = pair_dir / "validation.json"
                if receipt_path.exists():
                    done = h.read(receipt_path)
                    assert done["protocol"] == protocol_binding and done["position"] == position
                    assert done["status"] == "PROPERTY_RESTORE_CPU_PAIR_PASS"
                    assert [r["arm"] for r in done["arms"]] == list(t.ARMS)
                    completed_pairs.append(h.binding(receipt_path))
                    continue
                if time.monotonic()-started >= args.budget:
                    return finish("NORMAL_BUDGET_CHECKPOINT", 75)
                pair_dir.mkdir(parents=True, exist_ok=True)
                old_pair_binding = old_validation["pairs"][position]
                old_pair = torch.load(h.checked(old_pair_binding), map_location="cpu", weights_only=True, mmap=True)
                capture_binding = old_pair["capture"]
                capture = torch.load(h.checked(capture_binding), map_location="cpu", weights_only=True, mmap=True)
                physical = work["candidate_physical_rows"][position]
                assert old_pair["physical_row"] == physical and old_pair["position"] == position
                assert old_pair["index"] == args.index
                geom = geometry_rows[position]
                assert capture["query_sha"] == geom["query_sha"] and capture["reference_sha"] == geom["reference_sha"]
                assert capture["authority"] == protocol["sources"]["source_authority"]
                sources = dict(source_pair=old_pair_binding, source_capture=capture_binding,
                               descriptors=capture["descriptor_sources"], geometry=geometry_source,
                               source_head=protocol["sources"]["source_head"])
                appearances = [descriptor(b) for b in capture["descriptor_sources"]]
                contexts, positions = [], []
                for side in (0, 1):
                    values = capture["sides"][side]
                    contexts.append(values["J"]["tensor"].to(dtype=getattr(torch, values["J"]["original_dtype"])))
                    positions.append(values["P"]["tensor"].to(dtype=getattr(torch, values["P"]["original_dtype"])))

                def forward(side, changed, label):
                    fs = [x.clone() for x in appearances[side]]
                    fs[-1] = fs[-1] + contexts[side] + changed
                    h.emit(stage="SIDE_FORWARD_START", index=args.index, position=position,
                           arm=label, side=side)
                    tick = time.perf_counter()
                    prediction = head(fs, img_A=None, img_B=None)
                    elapsed = time.perf_counter()-tick
                    h.emit(stage="SIDE_FORWARD_DONE", index=args.index, position=position,
                           arm=label, side=side, seconds=elapsed)
                    assert prediction.device.type == "cpu" and prediction.dtype == torch.float32
                    assert bool(torch.isfinite(prediction).all()), "NONFINITE_DPT_OUTPUT"
                    return prediction[..., :2].clone(), prediction[..., 2:].clone(), elapsed

                # Recheck this launch's first pending natural pair on the same
                # CPU backend, even when its NATIVE arm was saved by an earlier
                # job.  An old GPU result is never the determinism reference.
                precomputed_native = None
                if not native_repeat_done:
                    repeat_rows = []
                    precomputed_native = []
                    existing_native = None
                    if (pair_dir / "NATIVE.pt").exists():
                        existing_native = torch.load(pair_dir / "NATIVE.pt", map_location="cpu", weights_only=True)
                        assert existing_native["protocol"] == protocol_binding
                    for side in (0, 1):
                        w1, c1, seconds1 = forward(side, positions[side], "NATIVE_REPEAT_FIRST")
                        w2, c2, seconds2 = forward(side, positions[side], "NATIVE_REPEAT_SECOND")
                        precomputed_native.append((w1, c1, seconds1))
                        checks = dict(confidence=h.differences(c2, c1, torch), warp=h.differences(w2, w1, torch))
                        if existing_native is not None:
                            checks["prior_cpu_confidence"] = h.differences(c1, existing_native["sides"][side]["confidence"], torch)
                        assert all(v["bit_exact"] for v in checks.values()), "CPU_NATIVE_REPEAT_DRIFT"
                        repeat_rows.append(dict(side=side, checks=checks, seconds=[seconds1, seconds2]))
                    repeat = dict(protocol=protocol_binding, backend=backend_binding, index=args.index,
                                  position=position, physical_row=physical, run_id=run_id,
                                  status="SAME_CPU_NATIVE_REPEAT_PASS", sides=repeat_rows)
                    h.write_once(folder / "native_repeat" / f"{run_id}.json", repeat)
                    native_repeat_done = True
                    h.emit(stage="SAME_CPU_NATIVE_REPEAT_PASS", index=args.index, position=position)

                arms = []
                for arm in t.ARMS:
                    arm_path = pair_dir / f"{arm}.pt"
                    if arm_path.exists():
                        row = torch.load(arm_path, map_location="cpu", weights_only=True, mmap=True)
                        assert row["protocol"] == protocol_binding and row["backend_binding"] == backend_binding
                        assert row["arm"] == arm and row["position"] == position and row["physical_row"] == physical
                        assert row["sources"] == sources
                        arms.append(dict(arm=arm, payload=h.binding(arm_path), M=float(row["M"])))
                        continue
                    if time.monotonic()-started >= args.budget:
                        return finish("NORMAL_BUDGET_CHECKPOINT", 75)
                    sides, metrics = [], []
                    for side in (0, 1):
                        changed, invariants = t.transform(positions[side], omega, scale, arm, protocol["radius"][side], side)
                        assert bool(torch.isfinite(changed).all()), "NONFINITE_PROPERTY_INPUT"
                        if arm == "NATIVE" and precomputed_native is not None:
                            assert torch.equal(changed, positions[side])
                            warp, confidence, elapsed = precomputed_native[side]
                        else:
                            warp, confidence, elapsed = forward(side, changed, arm)
                        overlap = confidence[0, ..., 0].sigmoid()
                        weights = pool(overlap, geom["geometries"][side])
                        assert weights.dtype == torch.float64 and bool(torch.isfinite(weights).all())
                        assert bool(((weights >= 0) & (weights <= 1)).all())
                        entry = dict(weights=weights, confidence=confidence, invariants=invariants)
                        if arm == "NATIVE":
                            entry["warp"] = warp
                        comparison = None
                        if arm in old_pair["branches"]:
                            old_side = old_pair["branches"][arm][side]
                            comparison = dict(confidence=h.differences(confidence, old_side["confidence"], torch),
                                              weights=h.differences(weights, old_side["weights"], torch),
                                              meaning="Descriptive CPU/GPU drift only; GPU values do not enter scientific factorial contrasts.")
                        metrics.append(dict(side=side, head_seconds=elapsed, gpu_reference=comparison))
                        sides.append(entry)
                    mass = torch.sqrt(sides[0]["weights"].mean() * sides[1]["weights"].mean())
                    assert bool(torch.isfinite(mass)) and float(mass) > 0, "INVALID_MASS"
                    gpu_mass = None
                    if arm in old_pair["branches"]:
                        old_sides = old_pair["branches"][arm]
                        reference = torch.sqrt(old_sides[0]["weights"].mean() * old_sides[1]["weights"].mean())
                        gpu_mass = dict(M=float(reference), difference=float(mass-reference),
                                        log_difference=float(mass.log()-reference.log()),
                                        bit_exact=(mass.dtype == reference.dtype and torch.equal(mass, reference)))
                    row = dict(protocol=protocol_binding, backend_binding=backend_binding, index=args.index,
                               query_id=work["query_id"], position=position, physical_row=physical,
                               arm=arm, M=float(mass), sides=sides, metrics=metrics, sources=sources,
                               source_pair=old_pair_binding, source_capture=capture_binding,
                               geometry_metadata=geom["metadata"], gpu_reference=gpu_mass,
                               cpu_only_scientific_arm=True)
                    h.save_once(arm_path, row, torch)
                    arms.append(dict(arm=arm, payload=h.binding(arm_path), M=float(mass)))
                    measured_arms += 1
                    h.emit(stage="ARM_SAVED", index=args.index, position=position, arm=arm,
                           M=float(mass), head_seconds=sum(x["head_seconds"] for x in metrics))
                assert len(arms) == 18 and [x["arm"] for x in arms] == list(t.ARMS)
                pair_receipt = dict(status="PROPERTY_RESTORE_CPU_PAIR_PASS", protocol=protocol_binding,
                                    backend_binding=backend_binding, index=args.index, query_id=work["query_id"],
                                    position=position, physical_row=physical, arms=arms, sources=sources,
                                    all_arms_same_cpu_backend=True, original_gpu_bit_exact_not_claimed=True)
                h.write_once(receipt_path, pair_receipt)
                completed_pairs.append(h.binding(receipt_path))
                h.emit(stage="PAIR_COMPLETE", index=args.index, position=position, completed_pairs=len(completed_pairs))
        assert len(completed_pairs) == len(work["positions"]) == 2
        validation = dict(status="PROPERTY_RESTORE_CPU_QUERY_PASS", protocol=protocol_binding,
                          backend_binding=backend_binding, index=args.index, query_id=work["query_id"],
                          candidate_physical_rows=work["candidate_physical_rows"], arms=list(t.ARMS),
                          positions=work["positions"], scope="PRIMARY_FIXED_PAIRS",
                          pairs=completed_pairs, all_arms_same_cpu_backend=True,
                          native_repeat_receipts=[h.binding(p) for p in sorted((folder / "native_repeat").glob("*.json"))],
                          original_gpu_bit_exact_not_claimed=True, source_validation=work["gpu_validation"])
        h.write_once(folder / "validation.json", validation)
        return finish("QUERY_COMPLETE", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budget", type=float, default=430.0)
    args = parser.parse_args()
    assert 0 < args.budget <= 430
    torch = configure(args.threads)
    if args.mode == "prepare":
        assert args.index is None
        prepare(args, torch)
        return 0
    assert args.index is not None
    return worker(args, torch)


if __name__ == "__main__":
    raise SystemExit(main())
