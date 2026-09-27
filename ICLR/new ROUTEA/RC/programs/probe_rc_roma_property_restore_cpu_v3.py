#!/usr/bin/env python3
"""Measure CPU DPT parity and cost before any new property-restoration experiment.

This probe does not train, collect new image features, choose candidates using
labels, or approve an approximate replay.  It compares four already sealed GPU
arms at the first natural pair by default.  Each arm is saved independently so
a bounded CPU job can resume without overwriting historical experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
SOURCE = ROOT / "results/rc_roma_position_factor_v1"
OUT = ROOT / "results/rc_roma_property_restore_cpu_v3"
ROMA = WORKSPACE / "third_party/RoMaV2/src/romav2"
ARMS = ("NATIVE", "ZERO", "GLOBAL_X_PLUS", "LOCAL_X_PLUS")


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).absolute()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def checked(record):
    assert binding(record["path"]) == record, ("SOURCE_SHA_DRIFT", record["path"])
    return Path(record["path"])


def write_once(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, ("IMMUTABLE_JSON_DRIFT", str(path))
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text)
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_once(path, value, torch):
    path = Path(path)
    assert not path.exists(), ("IMMUTABLE_TENSOR", str(path))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def differences(actual, expected, torch):
    assert actual.shape == expected.shape
    a, b = actual.detach().double(), expected.detach().double()
    assert bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    delta = a - b
    absolute = delta.abs().flatten()
    bit_exact = (actual.dtype == expected.dtype and torch.equal(
        actual.reshape(-1).contiguous().view(torch.uint8), expected.reshape(-1).contiguous().view(torch.uint8)))
    return dict(shape=list(actual.shape), actual_dtype=str(actual.dtype),
                expected_dtype=str(expected.dtype), bit_exact=bit_exact,
                different_values=int(torch.count_nonzero(delta)),
                max_abs=float(absolute.max()), mean_abs=float(absolute.mean()),
                rms=float(delta.square().mean().sqrt()),
                p99_abs=float(torch.quantile(absolute, 0.99)),
                relative_l2=float(delta.norm() / b.norm().clamp_min(1e-12)))


def load_cpu_dpt(torch):
    # Import only the pinned DPT source, without romav2.__init__ importing all
    # encoders or discovering/downloading pretrained models.  The original DPT
    # itself retains its BF16 autocast block and FP32 final output_conv2.
    assert "romav2" not in sys.modules, "FRESH_PROCESS_REQUIRED"
    package = types.ModuleType("romav2")
    package.__path__ = [str(ROMA)]
    package.__package__ = "romav2"
    sys.modules["romav2"] = package
    device_module = types.ModuleType("romav2.device")
    device_module.device = torch.device("cpu")
    sys.modules["romav2.device"] = device_module
    module = importlib.import_module("romav2.dpt")
    assert module.device.type == "cpu"
    assert Path(module.__file__).resolve() == (ROMA / "dpt.py").resolve()
    return module.DPTHead


def pooling_diagnostic(confidence, sealed_weights, geometry, torch):
    """Separate CPU sigmoid rounding from integer-box pooling verification.

    Historical confidence.sigmoid() ran on CUDA before transfer to CPU.
    Its rounded probability map was not retained.  Thus the saved logits do
    not license a claim of exact GPU pooling reconstruction with CPU sigmoid.
    We compare CPU FP32 and FP64 sigmoid routes to the historical weights,
    while validating the pooling structure against independent integer boxes.
    """
    from rc_roma_shared_native_cache_v1 import ExactPool

    logits = confidence[0, ..., 0]
    probability32 = logits.sigmoid()
    probability64 = logits.double().sigmoid()
    actual32 = ExactPool()(probability32, geometry)
    actual64 = ExactPool()(probability64, geometry)
    height, width = logits.shape
    bounds, expected_values = [], []
    # Independent literal integer-bound reconstruction, intentionally separate
    # from ExactPool's cached lists.  The sigmoid result is held identical.
    for box, valid in zip(geometry.cell_boxes_xyxy, geometry.valid_patch_mask, strict=True):
        if not bool(valid):
            bounds.append(None)
            expected_values.append(probability32.new_zeros((), dtype=torch.float64))
            continue
        x0, y0, x1, y1 = (float(value) for value in box)
        left = max(0, min(width - 1, math.floor(x0 * width)))
        top = max(0, min(height - 1, math.floor(y0 * height)))
        right = max(left + 1, min(width, math.ceil(x1 * width)))
        bottom = max(top + 1, min(height, math.ceil(y1 * height)))
        bounds.append([top, bottom, left, right])
        expected_values.append(probability32.double()[top:bottom, left:right].mean())
    independently_pooled = torch.stack(expected_values)
    assert torch.equal(independently_pooled, actual32), "CPU_POOL_INTEGER_BOUNDS_MISMATCH"
    assert actual32.shape == actual64.shape == sealed_weights.shape
    diagnostic = dict(
        cpu_sigmoid_fp32_then_pool_vs_sealed=differences(actual32, sealed_weights, torch),
        cpu_sigmoid_fp64_then_pool_vs_sealed=differences(actual64, sealed_weights, torch),
        cpu_sigmoid_fp32_vs_fp64=differences(probability32.double(), probability64, torch),
        cpu_pool_independent_integer_bounds_bit_exact=True,
        historical_gpu_probability_map_saved=False,
        interpretation="Cross-backend sigmoid rounding is measured, not asserted away; historical logits alone cannot establish bit-exact CUDA probability pooling.")
    tensors = dict(cpu_fp32_sigmoid_repool=actual32, cpu_fp64_sigmoid_repool=actual64,
                   integer_pool_bounds=bounds)
    return diagnostic, tensors


def run(args):
    # Configure before importing torch, and pin actual intra/inter-op counts.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(ROOT / "programs"), str(ROOT / "src")]
    import rc_roma_position_factor_v1 as phase
    from rc_roma_shared_native_cache_v1 import ExactPool
    from rc_aslo_xf.colnomic_dino_canonical_geometry_v2 import build_colnomic_canonical_geometry_v2

    DPTHead = load_cpu_dpt(torch)
    manifest_path = SOURCE / "manifest.json"
    manifest = read(manifest_path)
    assert args.index in manifest["indices"] and 0 <= args.position < 128
    worker = next(w for w in manifest["workers"] if w["execution_ordinal"] == args.index)
    query_dir = SOURCE / f"query{args.index:03d}"
    source_validation = read(query_dir / "gpu_validation.json")
    assert source_validation["status"] == "POSITION_FACTOR_ALL128_PASS"
    pair_binding = source_validation["pairs"][args.position]
    pair = torch.load(checked(pair_binding), map_location="cpu", weights_only=True, mmap=True)
    assert pair["index"] == args.index and pair["position"] == args.position
    cap_binding = pair["capture"]
    capture = torch.load(checked(cap_binding), map_location="cpu", weights_only=True, mmap=True)
    authority = pair["authority"]
    checked(authority)
    assert read(authority["path"])["manifest"] == binding(manifest_path)
    assert capture["authority"] == authority == source_validation["authority"]
    head_path = SOURCE / "head_replay_state.pt"
    assert checked(read(SOURCE / "pilot_validation.json")["head_state"]) == head_path
    head_state = torch.load(head_path, map_location="cpu", weights_only=True, mmap=True)
    assert head_state["authority"] == authority

    # Historical RoMa records already contain the geometry metadata; no original
    # image, retrieval token bank, image encoder, matcher, or target label is read.
    geom_binding = worker["roma"]["payload"]
    geometry_payload = torch.load(checked(geom_binding), map_location="cpu", weights_only=True, mmap=True)
    query = geometry_payload["records"][worker["source_index"]]
    assert query["query_id"] == worker["source_query_id"]
    assert query["query_source_sha256"] == capture["query_sha"] == worker["source_image_sha256"]
    candidate = query["candidates"][args.position]
    assert candidate["physical_row"] == pair["physical_row"] == query["candidate_physical_rows"][args.position]
    assert candidate["reference_image_sha256"] == capture["reference_sha"]
    geom_metadata = [query["query_geometry"], candidate["reference_geometry"]]
    geometries = []
    for side, (meta, image_sha) in enumerate(zip(geom_metadata, (capture["query_sha"], capture["reference_sha"]))):
        geometries.append(build_colnomic_canonical_geometry_v2(
            source_image_sha256=image_sha,
            source_key=f"cpu-property-probe:{args.index}:{args.position}:{side}",
            processor_config_sha256=meta["processor_config_sha256"],
            raw_size_hw=tuple(meta["raw_size_hw"]),
            exif_orientation=int(meta["exif_orientation"]),
            merged_grid_shape=tuple(meta["grid_shape"]),
            processor_input_frame=meta["processor_input_frame"]))
        assert tuple(geometries[-1].oriented_size_hw) == tuple(meta["oriented_size_hw"])
        assert len(geometries[-1].cell_boxes_xyxy) == math.prod(meta["grid_shape"])
    del geometry_payload, query, candidate

    features = []
    for descriptor in capture["descriptor_sources"]:
        payload = torch.load(checked(descriptor), map_location="cpu", weights_only=True, mmap=True)
        features.append([t.clone() for t in payload["features"]])
    assert len(features) == len(capture["sides"]) == 2
    head = DPTHead(**head_state["config"]).cpu().eval()
    head.load_state_dict(head_state["head"], strict=True)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    omega, scale = head_state["omega"], head_state["scale"]
    pool = ExactPool()

    folder = OUT / f"query{args.index:03d}_pair{args.position:03d}_threads{args.threads:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    source_code = [Path(__file__), Path(phase.__file__), ROMA / "dpt.py", ROMA / "matcher.py",
                   ROOT / "programs/rc_roma_shared_native_cache_v1.py",
                   ROOT / "programs/rc_roma_exact_feature_cache_v2.py",
                   ROOT / "src/rc_aslo_xf/colnomic_dino_canonical_geometry_v2.py"]
    protocol = dict(schema="CPU_DPT_PROPERTY_RESTORE_PILOT_V2", index=args.index,
                    position=args.position, physical_row=pair["physical_row"],
                    query_sha=capture["query_sha"], reference_sha=capture["reference_sha"],
                    arms=list(ARMS), threads=args.threads, interop_threads=1,
                    source_authority=authority, source_manifest=binding(manifest_path),
                    source_gpu_validation=binding(query_dir / "gpu_validation.json"),
                    source_pair=pair_binding, source_capture=cap_binding,
                    source_head=binding(head_path), geometry_source=geom_binding,
                    descriptors=capture["descriptor_sources"], code=[binding(p) for p in source_code],
                    geometry_metadata=geom_metadata,
                    outcome_blind_selection="Execution ordinal and natural candidate position only",
                    device="cpu", autocast="Original DPT BF16 block; final output_conv2 FP32",
                    numerical_policy="Measure all errors. No approximate acceptance threshold is chosen or relaxed.",
                    source_pooling_policy="Historical sigmoid executed on CUDA; recomputing sigmoid on CPU is only a numerical diagnostic, not an exact historical pool gate. CPU pool structure is independently verified with common probability input and explicit integer bounds.",
                    backend_policy="Any later non-bit-exact CPU experiment must recompute every scientific arm on the same frozen CPU backend. Sealed GPU outputs are numerical references only and cannot be mixed into CPU causal contrasts.",
                    next_stage_policy="Pilot measurement never automatically launches a larger experiment.")
    write_once(folder / "protocol.json", protocol)
    protocol_binding = binding(folder / "protocol.json")
    environment = dict(python=sys.version, torch=torch.__version__, platform=platform.platform(),
                       torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                       mkldnn_enabled=torch.backends.mkldnn.enabled,
                       cpu_capability=torch.backends.cpu.get_cpu_capability(),
                       omp_num_threads=os.environ["OMP_NUM_THREADS"], mkl_num_threads=os.environ["MKL_NUM_THREADS"],
                       torch_config=torch.__config__.show(), torch_parallel_info=torch.__config__.parallel_info())
    write_once(folder / "environment.json", environment)
    environment_binding = binding(folder / "environment.json")
    if (folder / "result.json").exists():
        complete = read(folder / "result.json")
        assert complete["protocol"] == protocol_binding and complete["environment"] == environment_binding
        for row in complete["arms"]:
            checked(row["payload"])
        emit(stage="PILOT_ALREADY_MEASURED", status=complete["status"], path=str(folder / "result.json"))
        return

    def head_forward(side, p):
        j = capture["sides"][side]["J"]
        j = j["tensor"].to(dtype=getattr(torch, j["original_dtype"]), device="cpu")
        fs = [t.clone() for t in features[side]]
        # Literal original _compute_head_preds operation; clones prevent input
        # list mutation from contaminating later arms or the second run.
        fs[-1] = fs[-1] + j + p
        start = time.perf_counter()
        output = head(fs, img_A=None, img_B=None)
        elapsed = time.perf_counter() - start
        assert output.device.type == "cpu" and output.dtype == torch.float32
        return output[..., :2].clone(), output[..., 2:].clone(), elapsed

    collected = []
    with torch.inference_mode():
        for arm in ARMS:
            path = folder / f"{arm}.pt"
            if path.exists():
                result = torch.load(path, map_location="cpu", weights_only=True)
                assert result["protocol"] == protocol_binding and result["environment"] == environment_binding
                assert result["arm"] == arm
                collected.append(result)
                emit(stage="REUSED", arm=arm, path=str(path))
                continue
            sides, row_metrics = [], []
            for side in (0, 1):
                p_info = capture["sides"][side]["P"]
                p = p_info["tensor"].to(dtype=getattr(torch, p_info["original_dtype"]), device="cpu")
                changed, invariants = phase.transform(p, omega, scale, arm, manifest["radius"][side], side)
                gpu = pair["branches"][arm][side]
                pooling_report, pooling_tensors = pooling_diagnostic(gpu["confidence"], gpu["weights"], geometries[side], torch)
                warp, confidence, seconds = head_forward(side, changed)
                weights = pool(confidence[0, ..., 0].sigmoid(), geometries[side])
                comparison = dict(confidence=differences(confidence, gpu["confidence"], torch),
                                  weights=differences(weights, gpu["weights"], torch),
                                  overlap=differences(confidence.sigmoid(), gpu["confidence"].sigmoid(), torch))
                native_repeat = None
                if arm == "NATIVE":
                    comparison["warp"] = differences(warp, gpu["native_warp"], torch)
                    w2, c2, t2 = head_forward(side, changed)
                    repeat_metrics = dict(warp=differences(w2, warp, torch),
                                          confidence=differences(c2, confidence, torch))
                    native_repeat = dict(warp=w2, confidence=c2, head_seconds=t2, comparison=repeat_metrics)
                    assert all(v["bit_exact"] for v in repeat_metrics.values()), "CPU_NATIVE_NOT_DETERMINISTIC"
                sides.append(dict(cpu_warp=warp, cpu_confidence=confidence, cpu_weights=weights,
                                  sealed_gpu_confidence=gpu["confidence"].clone(),
                                  sealed_gpu_weights=gpu["weights"].clone(),
                                  sealed_gpu_warp=gpu.get("native_warp"),
                                  source_pooling_diagnostic=pooling_tensors,
                                  invariants=invariants, native_repeat=native_repeat))
                row_metrics.append(dict(side=side, head_seconds=seconds, errors=comparison,
                                        source_pooling_diagnostic=pooling_report))
                emit(stage="SIDE_MEASURED", arm=arm, **row_metrics[-1])
            cpu_mass = torch.sqrt(sides[0]["cpu_weights"].mean() * sides[1]["cpu_weights"].mean())
            gpu_mass = torch.sqrt(sides[0]["sealed_gpu_weights"].mean() * sides[1]["sealed_gpu_weights"].mean())
            assert float(cpu_mass) > 0 and float(gpu_mass) > 0
            mass = dict(cpu=float(cpu_mass), sealed_gpu=float(gpu_mass),
                        error=differences(cpu_mass.reshape(1), gpu_mass.reshape(1), torch),
                        log_error=float(cpu_mass.log() - gpu_mass.log()))
            result = dict(protocol=protocol_binding, environment=environment_binding,
                          arm=arm, sides=sides, metrics=row_metrics, mass=mass)
            save_once(path, result, torch)
            collected.append(result)
            emit(stage="ARM_SAVED", arm=arm, mass=mass, path=str(path))

    exact = all(metric["bit_exact"] for r in collected for side in r["metrics"] for metric in side["errors"].values())
    baseline = collected[0]["mass"]
    arms = []
    for r in collected:
        gpu_delta = float(torch.tensor(r["mass"]["sealed_gpu"], dtype=torch.float64).log()
                          - torch.tensor(baseline["sealed_gpu"], dtype=torch.float64).log())
        cpu_delta = float(torch.tensor(r["mass"]["cpu"], dtype=torch.float64).log()
                          - torch.tensor(baseline["cpu"], dtype=torch.float64).log())
        arms.append(dict(arm=r["arm"], metrics=r["metrics"], mass=r["mass"],
                         native_relative_logM=dict(cpu=cpu_delta, sealed_gpu=gpu_delta,
                                                   difference=cpu_delta-gpu_delta),
                         payload=binding(folder / f"{r['arm']}.pt")))
    by_arm = {r["arm"]: r for r in collected}
    global_mass, local_mass = (by_arm[n]["mass"] for n in ("GLOBAL_X_PLUS", "LOCAL_X_PLUS"))
    cpu_contrast = math.log(local_mass["cpu"]) - math.log(global_mass["cpu"])
    gpu_contrast = math.log(local_mass["sealed_gpu"]) - math.log(global_mass["sealed_gpu"])
    summary = dict(status="CPU_DPT_PILOT_MEASURED_BIT_EXACT" if exact else "CPU_DPT_PILOT_MEASURED_WITH_DRIFT_REVIEW_REQUIRED",
                   protocol=protocol_binding, environment=environment_binding, arms=arms,
                   native_repeat_bit_exact=True, cpu_pool_integer_bounds_bit_exact=True,
                   sealed_gpu_repool_bit_exact=all(m["source_pooling_diagnostic"]["cpu_sigmoid_fp32_then_pool_vs_sealed"]["bit_exact"] for r in collected for m in r["metrics"]),
                   local_minus_global_logM=dict(cpu=cpu_contrast, sealed_gpu=gpu_contrast,
                                                difference=cpu_contrast-gpu_contrast,
                                                note="One sign, one axis, one natural pair: backend diagnostic only; not the predeclared antithetic scientific contrast."),
                   total_measured_head_seconds=sum(m["head_seconds"] for r in collected for m in r["metrics"]),
                   native_repeat_head_seconds=sum(s["native_repeat"]["head_seconds"] for s in collected[0]["sides"]),
                   max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   approximate_acceptance_threshold=None, expanded_experiment_authorized_by_probe=False,
                   interpretation="One natural candidate pair is an engineering pilot, not an identity or causal result.",
                   suggested_next_gate="Inspect confidence, weight and logM errors; freeze any numerical tolerance before additional pairs, then validate on additional outcome-blind pairs and all planned arms. Do not approve solely from aggregate M or loosen thresholds after observing effects.")
    write_once(folder / "result.json", summary)
    emit(stage="PILOT_MEASURED", status=summary["status"], path=str(folder / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    assert args.threads > 0
    run(args)


if __name__ == "__main__":
    main()
