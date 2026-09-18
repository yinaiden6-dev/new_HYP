#!/usr/bin/env python3
"""Cache original RAW local evidence for a shared query prior; no model fitting.

FULL64 retains original RAW C128. PAIR64 retains its two frozen training
candidates and complete RAW axis. Labels are copied only for the PAIR records.
No original evaluation outcomes or target identities are read.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/rc_shared_query_target_prior_cache_v1"
AUTH = ROOT / "registry/rc_shared_query_target_prior_cache_authority_v1_20260909.json"
PLAN = ROOT / "plan/RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_20260909.md"
LAUNCH = ROOT / "slurm/rc_shared_query_target_prior_cache_v1_dev_cpuonly_59m.sbatch"
FULL = ROOT / "results/rc_original_raw_visibility_pv_inputs_v1"
PAIR = ROOT / "results/routea_matched_three_arm_pair64_training_features_v2"
GEOMETRY = ROOT / "cache/dino_rcde_colnomic_sr_full600_geometry_v2/full600_geometry_payload_v2.pt"
SCALARS = ("real_score", "visibility_mass", "query_control_score", "reference_control_score")
PINNED = {
    "full_manifest": (FULL / "manifest.json", "2777f72c1eacc3dd46f1ab4ba7f18c0dda23ddcb95a51f4b87c9fecd67dec2d4"),
    "pair_validation": (PAIR / "independent_validation.json", "657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b"),
    "pair_lineage": (PAIR / "post_full_lineage.json", "76be3c0cfc839cbe2029e8f7b25f030be15fdb5c37c752ccf9e21e248da33d4e"),
    "geometry": (GEOMETRY, "fff5b980ffa88997ed9bb2a509686800a3448d18a93a714c4ef89007fb24329b"),
    "old_score_source": (ROOT / "programs/run_romav2_colnomic_visibility_xf_six_case_v1.py", "fb73bdd6cc2b585405a9fcb1e411535f487e83d29d6021f8c1f632af78ecfd3c"),
}


def need(value, message):
    if not bool(value):
        raise RuntimeError(message)


def safe(path):
    path = Path(path)
    need(path.resolve().is_relative_to(ROOT) and not path.is_symlink(), "UNSAFE_PATH:" + str(path))
    need(not any(x in str(path).lower() for x in ("d1_mi", "d1-mi", "d1_minimal_intervention", "grozi", "gisc_prerecall_universe")), "PROTECTED_PATH")
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(safe(path).read_text())


def bind(path):
    return {"path": str(safe(path).relative_to(ROOT)), "sha256": sha(path)}


def checked(item):
    path = safe(ROOT / item["path"])
    need(sha(path) == item["sha256"], "SOURCE_HASH_DRIFT:" + str(path))
    return path


def save(path, value):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return bind(path)


def raw_tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode("ascii"))
    h.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def source_pins():
    result = {}
    for name, (path, expected) in PINNED.items():
        need(sha(path) == expected, "PIN_DRIFT:" + name)
        result[name] = bind(path)
    validation = read(PINNED["pair_validation"][0])
    need(validation["status"] == "ROUTEA_MATCHED_THREE_ARM_PAIR64_V2_VALIDATED" and all(x is True for x in validation["checks"].values()), "PAIR_VALIDATION")
    payload = bind(PAIR / "payload.pt")
    need(payload["sha256"] == validation["v2_payload_sha256"], "PAIR_PAYLOAD_PIN")
    result["pair_payload"] = payload
    for name, path in (("program", Path(__file__)), ("plan", PLAN), ("launcher", LAUNCH)):
        result[name] = bind(path)
    return result


class InputBundle:
    """Read only the original RAW query tokens, visibility maps and pair labels."""

    def __init__(self, sources):
        import torch
        self.torch = torch
        self.full = read(checked(sources["full_manifest"]))
        need(self.full["status"] == "RC_ORIGINAL_RAW_VISIBILITY_PV_INPUTS_FULL64_EXACT_EXPORT_PASS", "FULL_MANIFEST")
        self.pair = torch.load(checked(sources["pair_payload"]), map_location="cpu", mmap=True, weights_only=True)
        self.geometry = torch.load(checked(sources["geometry"]), map_location="cpu", mmap=True, weights_only=False)
        self.queries = {r["query_id"]: r for r in self.geometry["query_records"]}
        self.references = {r["physical_row"]: r for r in self.geometry["reference_records"]}
        self.cache = {}
        self.hashes = {}

    def leakage(self):
        axes = {}
        for kind, records in (("PAIR64", self.pair["records"]), ("FULL_EVAL32", [r for r in self.full["records"] if r["role"] == "EVAL"]), ("FULL_TRAIN32", [r for r in self.full["records"] if r["role"] == "TRAIN"])):
            axes[kind] = {"query_id": {r["query_id"] for r in records}, "execution_ordinal": {r["execution_ordinal"] for r in records}, "source_image_sha256": {self.queries[r["query_id"]]["source_image_sha256"] for r in records}}
        overlaps = {kind: {key: sorted(axes[kind][key] & axes["FULL_EVAL32"][key]) for key in axes[kind]} for kind in ("PAIR64", "FULL_TRAIN32")}
        need(all(not values for row in overlaps.values() for values in row.values()), "TRAIN_TO_EVAL_QUERY_LEAKAGE")
        need(len(self.pair["records"]) == 64 and len(self.full["records"]) == 64, "POPULATION")
        return {"status": "TRAIN_EVAL_QUERY_AND_EXACT_IMAGE_DISJOINT_PASS", "overlap": overlaps, "identity_disjointness_claimed": False, "source_metadata_only": True}

    def _cache_entry(self, record):
        pointer = record["colnomic_cache"]
        path = safe(ROOT / pointer["cache_path"])
        if pointer["storage_kind"] == "existing_spatial_cache":
            if str(path) not in self.cache:
                need(sha(path) == pointer["cache_file_sha256"], "TOKEN_ARCHIVE_HASH")
                self.hashes[str(path)] = bind(path)
                self.cache[str(path)] = self.torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            entry = self.cache[str(path)]["entries"][pointer["cache_entry_index"]]
        else:
            need(path == GEOMETRY and record["kind"] == "reference", "SUPPLEMENTAL_POINTER")
            entry = next(r for r in self.geometry["supplemental_gallery_entries"] if r["physical_row"] == record["physical_row"])
        tokens = entry["tokens"]
        need(tokens.dtype == self.torch.float16 and list(tokens.shape) == [pointer["grid_shape"][0] * pointer["grid_shape"][1], 128], "TOKEN_SHAPE")
        need(token_sha(tokens) == pointer["tokens_sha256"] == entry["tokens_sha256"], "TOKEN_HASH")
        return tokens

    def full_episodes(self):
        torch = self.torch
        for shard in sorted(self.full["shards"], key=lambda x: x["shard"]):
            path = FULL / shard["path"]
            need(sha(path) == shard["sha256"], "FULL_SHARD_PIN")
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            token_archives = {}
            for row in payload["records"]:
                source = row["token_source"]
                if source["path"] not in token_archives:
                    token_archives[source["path"]] = torch.load(checked(source), map_location="cpu", mmap=True, weights_only=True)
                archive = token_archives[source["path"]]
                qr = archive["records"][source["record_index"]]
                need(qr["query_id"] == row["query_id"] and qr["execution_ordinal"] == row["execution_ordinal"] and qr["candidate_physical_rows"] == row["candidate_physical_rows"], "FULL_QUERY_AXIS")
                need(qr["query_source_sha256"] == self.queries[row["query_id"]]["source_image_sha256"], "FULL_QUERY_IMAGE_BINDING")
                qt = qr["query_tokens"]
                need(token_sha(qt) == row["query_tokens_sha256"], "FULL_QUERY_TOKEN_SHA")
                cs = []
                for position, c in enumerate(row["candidates"]):
                    need(c["candidate_position"] == position and c["physical_row"] == row["candidate_physical_rows"][position], "FULL_CANDIDATE_AXIS")
                    rt = archive["references"][c["physical_row"]]["tokens"]
                    need(token_sha(rt) == c["reference_tokens_sha256"], "FULL_REFERENCE_TOKEN_SHA")
                    cs.append({"position": position, "physical_row": c["physical_row"], "r_tokens": rt, "wq": c["query_visibility"], "wr": c["reference_visibility"], "qshift": max(1, qt.shape[0] // 2), "rshift": max(1, rt.shape[0] // 2), "expected": {k: float(c["old_scores"][k]) for k in SCALARS}, "query_map_sha256": c["query_map_sha256"], "reference_map_sha256": c["reference_map_sha256"], "reference_tokens_sha256": c["reference_tokens_sha256"]})
                raw = row["candidate_raw_scores"]
                yield {"kind": "FULL", "role": row["role"], "episode_key": f"FULL_{row['execution_ordinal']:04d}", "query_id": row["query_id"], "execution_ordinal": row["execution_ordinal"], "q_tokens": qt, "q_grid_shape": row["query_grid_shape"], "q_tokens_sha256": row["query_tokens_sha256"], "query_source_image_sha256": self.queries[row["query_id"]]["source_image_sha256"], "axis": row["candidate_physical_rows"], "raw": raw, "winner": max(range(128), key=lambda i: (raw[i], -row["candidate_physical_rows"][i])), "source": {"maps": {"path": str(path.relative_to(ROOT)), "sha256": shard["sha256"]}, "tokens": source}, "candidates": cs}
            del token_archives, payload

    def pair_episodes(self):
        for row in self.pair["records"]:
            qr = self.queries[row["query_id"]]
            need(qr["execution_ordinal"] == row["execution_ordinal"] and qr["candidate_physical_rows"] == row["candidate_physical_rows"], "PAIR_QUERY_AXIS")
            qt = self._cache_entry(qr)
            cs = []
            for position in sorted(row["fixed_candidate_maps"]):
                c = row["fixed_candidate_maps"][position]
                need(c["candidate_position"] == position and c["physical_row"] == row["candidate_physical_rows"][position], "PAIR_CANDIDATE_AXIS")
                rt = self._cache_entry(self.references[c["physical_row"]])
                need(token_sha(qt) == c["query_tokens_sha256"] and token_sha(rt) == c["reference_tokens_sha256"], "PAIR_SOURCE_TOKEN_BINDING")
                cs.append({"position": position, "physical_row": c["physical_row"], "r_tokens": rt, "wq": c["query_map"], "wr": c["reference_map"], "qshift": c["query_control_shift"], "rshift": c["reference_control_shift"], "expected": {k: float(row["evidence"]["C_PAIRED"][position][k]) for k in SCALARS}, "query_map_sha256": c["query_map_sha256"], "reference_map_sha256": c["reference_map_sha256"], "reference_tokens_sha256": c["reference_tokens_sha256"]})
            yield {"kind": "PAIR", "role": "TRAIN", "episode_key": f"PAIR_{row['execution_ordinal']:04d}", "pair_cohort": row["pair_cohort"], "query_id": row["query_id"], "execution_ordinal": row["execution_ordinal"], "q_tokens": qt, "q_grid_shape": qr["colnomic_cache"]["grid_shape"], "q_tokens_sha256": qr["colnomic_cache"]["tokens_sha256"], "query_source_image_sha256": qr["source_image_sha256"], "axis": row["candidate_physical_rows"], "raw": row["base_scores"].tolist(), "winner": row["base_winner_position"], "challenger_positions": list(row["challenger_positions"]), "switch_label": bool(row["switch_label"]), "inner_fold": row["inner_fold"], "source": {"pair_payload": bind(PAIR / "payload.pt"), "query_pointer": qr["colnomic_cache"]}, "old_native_features": {arm: tensor.tolist() for arm, tensor in row["real_native_features"].items()}, "candidates": cs}

    def episodes(self):
        yield from self.full_episodes()
        yield from self.pair_episodes()


def reconstruct(wq, wr_mean, wr_rolled_mean, b, br, qshift):
    import torch
    mass = torch.sqrt(wq.mean() * wr_mean)
    shifted = wq.roll(qshift)
    qmass = torch.sqrt(shifted.mean() * wr_mean)
    rmass = torch.sqrt(wq.mean() * wr_rolled_mean)
    return {"real_score": float(mass * (wq * b).sum() / wq.sum().clamp_min(1e-12)), "visibility_mass": float(mass), "query_control_score": float(qmass * (shifted * b).sum() / shifted.sum().clamp_min(1e-12)), "reference_control_score": float(rmass * (wq * br).sum() / wq.sum().clamp_min(1e-12))}


def locals_from_tokens(qt, rt, wr, rshift):
    import torch.nn.functional as F
    sim = F.normalize(qt.double(), dim=1) @ F.normalize(rt.double(), dim=1).T
    return (sim * wr[None]).max(1).values, (sim * wr.roll(rshift)[None]).max(1).values, sim.max(1).values


def freeze():
    need(not AUTH.exists() and not OUT.exists(), "APPEND_ONLY_FREEZE_EXISTS")
    sources = source_pins()
    inputs = InputBundle(sources)
    leakage = inputs.leakage()
    save(AUTH, {"status": "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_AUTHORIZED", "sources": sources, "leakage": leakage, "full_queries": 64, "pair_queries": 64, "candidate_occurrences": 8320, "training_updates": 0, "new_encoder_or_RoMa_forwards": 0, "evaluation_target_or_outcome_reads_authorized": False, "PAIR_original_training_labels_may_be_copied": True, "HYP_GO_claimed": False})
    print(json.dumps({"status": "FROZEN", "authority_sha256": sha(AUTH), "leakage": leakage}), flush=True)


def authority():
    value = read(AUTH)
    need(value["status"] == "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_AUTHORIZED" and value["sources"] == source_pins(), "AUTHORITY_DRIFT")
    return value


def run():
    import numpy as np
    import torch
    auth = authority()
    need(not OUT.exists(), "APPEND_ONLY_OUTPUT_EXISTS")
    inputs = InputBundle(auth["sources"])
    leakage = inputs.leakage()
    started = time.monotonic()
    entries = []
    occurrences = 0
    for episode in inputs.episodes():
        need(time.monotonic() - started < 3000, "CACHE_WALLTIME_BUDGET")
        arrays = {"q_tokens": episode["q_tokens"].numpy(), "raw_scores": np.asarray(episode["raw"], dtype=np.float64), "candidate_positions": np.asarray([c["position"] for c in episode["candidates"]], dtype=np.int64)}
        rows = {k: [] for k in ("wq", "wr_mean", "wr_rolled_mean", "b", "br", "a", "qshift", "rshift")}
        candidates = []
        for c in episode["candidates"]:
            wq, wr = c["wq"], c["wr"]
            need(wq.dtype == wr.dtype == torch.float64 and wq.ndim == wr.ndim == 1 and wq.numel() == episode["q_tokens"].shape[0] and wr.numel() == c["r_tokens"].shape[0], "WEIGHT_DOMAIN")
            need(raw_tensor_sha(wq) == c["query_map_sha256"] and raw_tensor_sha(wr) == c["reference_map_sha256"], "MAP_SHA")
            b, br, a = locals_from_tokens(episode["q_tokens"], c["r_tokens"], wr, c["rshift"])
            wm, wrm = wr.mean(), wr.roll(c["rshift"]).mean()
            actual = reconstruct(wq, wm, wrm, b, br, c["qshift"])
            need(all(actual[k].hex() == c["expected"][k].hex() for k in SCALARS), "ORIGINAL_SCALAR_BIT_DRIFT:" + episode["episode_key"] + ":" + str(c["position"]))
            for k, tensor in (("wq", wq), ("wr_mean", wm), ("wr_rolled_mean", wrm), ("b", b), ("br", br), ("a", a)):
                rows[k].append(tensor.numpy())
            rows["qshift"].append(c["qshift"])
            rows["rshift"].append(c["rshift"])
            candidates.append({"candidate_position": c["position"], "physical_row": c["physical_row"], "reference_token_count": wr.numel(), "reference_tokens_sha256": c["reference_tokens_sha256"], "query_map_sha256": c["query_map_sha256"], "reference_map_sha256": c["reference_map_sha256"], "old_scalars_binary64": {k: actual[k].hex() for k in SCALARS}})
            occurrences += 1
        for key, values in rows.items():
            arrays[key] = np.asarray(values, dtype=np.int64 if key.endswith("shift") else np.float64)
        folder = OUT / "episodes" / episode["episode_key"]
        folder.mkdir(parents=True, exist_ok=False)
        array_path = folder / "arrays.npz"
        with array_path.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_path.chmod(0o444)
        meta = {k: v for k, v in episode.items() if k not in ("q_tokens", "candidates", "raw")}
        meta.update({"schema": "rc_shared_query_target_prior_episode_v1", "arrays": bind(array_path), "array_schema": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}, "candidates": candidates, "original_C_scalars_bit_exact": True, "new_forward_count": 0, "evaluation_label_or_outcome_reads": 0})
        entry = save(folder / "metadata.json", meta)
        entries.append({**entry, "episode_key": episode["episode_key"], "kind": episode["kind"], "role": episode["role"], "query_id": episode["query_id"], "execution_ordinal": episode["execution_ordinal"]})
        print(json.dumps({"event": "ORIGINAL_LOCAL_CACHE_EPISODE_EXACT", "episodes": len(entries), "candidate_occurrences": occurrences, "elapsed_seconds": time.monotonic() - started}), flush=True)
    need(len(entries) == 128 and occurrences == 8320 and len({x["episode_key"] for x in entries}) == 128, "FINAL_POPULATION")
    save(OUT / "manifest.json", {"status": "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_COMPLETE", "schema": "rc_shared_query_target_prior_cache_v1", "authority_sha256": sha(AUTH), "sources": auth["sources"], "records": entries, "leakage": leakage, "counts": {"FULL": 64, "PAIR": 64, "candidate_occurrences": occurrences, "original_C_scalar_hex_checks": 4 * occurrences}, "selected_pair_token_archives": list(inputs.hashes.values()), "elapsed_seconds": time.monotonic() - started, "new_training_updates": 0, "new_encoder_or_RoMa_forwards": 0, "evaluation_target_or_outcome_reads": 0, "HYP_GO_claimed": False, "scientific_GO_or_NO_GO": None})
    print(json.dumps({"status": "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_COMPLETE", "manifest_sha256": sha(OUT / "manifest.json")}), flush=True)


def validate():
    """Fresh source-bound process; recompute every local and legacy scalar."""
    import numpy as np
    import torch
    auth = authority()
    manifest = read(OUT / "manifest.json")
    need(manifest["authority_sha256"] == sha(AUTH) and len(manifest["records"]) == 128, "MANIFEST_AUTHORITY")
    inputs = InputBundle(auth["sources"])
    need(inputs.leakage() == manifest["leakage"], "LEAKAGE_DRIFT")
    saved = {r["episode_key"]: r for r in manifest["records"]}
    counts = collections.Counter()
    for episode in inputs.episodes():
        metadata = read(checked(saved[episode["episode_key"]]))
        need(metadata["axis"] == episode["axis"] and metadata["q_grid_shape"] == episode["q_grid_shape"] and metadata["winner"] == episode["winner"], "EPISODE_AXIS")
        need(all(metadata[k] == episode[k] for k in ("kind", "role", "query_id", "execution_ordinal", "query_source_image_sha256", "q_tokens_sha256")), "EPISODE_SOURCE_OR_ROLE_DRIFT")
        with np.load(checked(metadata["arrays"]), allow_pickle=False) as arrays:
            need(np.array_equal(arrays["q_tokens"], episode["q_tokens"].numpy()) and np.array_equal(arrays["raw_scores"], np.asarray(episode["raw"], dtype=np.float64)), "CACHED_QUERY_OR_RAW_DRIFT")
            if episode["kind"] == "PAIR":
                need(metadata["switch_label"] == episode["switch_label"] and metadata["challenger_positions"] == episode["challenger_positions"] and metadata["old_native_features"] == episode["old_native_features"], "PAIR_TRAINING_FIELDS_DRIFT")
            else:
                need("switch_label" not in metadata and "old_native_features" not in metadata, "FULL_LABEL_LEAK")
            for index, c in enumerate(episode["candidates"]):
                need(int(arrays["candidate_positions"][index]) == c["position"] and int(arrays["qshift"][index]) == c["qshift"] and int(arrays["rshift"][index]) == c["rshift"], "CACHE_POSITION_OR_SHIFT")
                locals_ = locals_from_tokens(episode["q_tokens"], c["r_tokens"], c["wr"], c["rshift"])
                for key, fresh in zip(("b", "br", "a"), locals_):
                    need(arrays[key][index].tobytes() == fresh.numpy().tobytes(), "RAW_LOCAL_BYTE_DRIFT:" + key)
                    counts["raw_local_vectors_bit_exact"] += 1
                need(arrays["wq"][index].tobytes() == c["wq"].numpy().tobytes(), "CACHED_WQ_DRIFT")
                need(float(arrays["wr_mean"][index]).hex() == float(c["wr"].mean()).hex() and float(arrays["wr_rolled_mean"][index]).hex() == float(c["wr"].roll(c["rshift"]).mean()).hex(), "CACHED_WR_MEAN_DRIFT")
                args = [torch.as_tensor(arrays[key][index]) for key in ("wq", "wr_mean", "wr_rolled_mean", "b", "br")]
                actual = reconstruct(*args, int(arrays["qshift"][index]))
                need(all(actual[k].hex() == c["expected"][k].hex() == metadata["candidates"][index]["old_scalars_binary64"][k] for k in SCALARS), "INDEPENDENT_ORIGINAL_SCALAR_DRIFT")
                counts["original_scalar_hex_checks"] += 4
                counts["candidate_occurrences"] += 1
        counts["episodes"] += 1
    need(counts["episodes"] == 128 and counts["candidate_occurrences"] == 8320 and counts["raw_local_vectors_bit_exact"] == 24960 and counts["original_scalar_hex_checks"] == 33280, "VALIDATION_POPULATION")
    save(OUT / "validation.json", {"status": "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_SOURCE_REPLAY_VALIDATION_PASS", "authority_sha256": sha(AUTH), "manifest_sha256": sha(OUT / "manifest.json"), "counts": dict(counts), "leakage": manifest["leakage"], "checks": {"all_cached_local_vectors_recomputed_from_original_tokens": True, "all_original_C_four_scalars_hex_exact": True, "full128_and_pair2_original_axes_unchanged": True, "all_query_token_bytes_and_training_labels_exact": True, "train_eval_exact_image_query_disjoint": True}, "new_training_updates": 0, "HYP_GO_claimed": False})
    print(json.dumps({"status": "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_SOURCE_REPLAY_VALIDATION_PASS", "counts": dict(counts)}), flush=True)


def e0():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(17)
    checks = 0
    for qcount, rcount, qshift, rshift in ((7, 5, 1, 1), (8, 7, 4, 3), (1, 1, 1, 1)):
        qt, rt = torch.randn(qcount, 128).half(), torch.randn(rcount, 128).half()
        wq, wr = torch.rand(qcount, dtype=torch.float64), torch.rand(rcount, dtype=torch.float64)
        for zero in (False, True):
            qweight = torch.zeros_like(wq) if zero else wq
            b, br, a = locals_from_tokens(qt, rt, wr, rshift)
            value = reconstruct(qweight, wr.mean(), wr.roll(rshift).mean(), b, br, qshift)
            expected = {}
            for field, qw, rw in (("real_score", qweight, wr), ("query_control_score", qweight.roll(qshift), wr), ("reference_control_score", qweight, wr.roll(rshift))):
                sim = F.normalize(qt.double(), dim=1) @ F.normalize(rt.double(), dim=1).T
                local = (sim * rw[None]).max(1).values
                mass = torch.sqrt(qw.mean() * rw.mean())
                expected[field] = float(mass * (qw * local).sum() / qw.sum().clamp_min(1e-12))
            expected["visibility_mass"] = float(torch.sqrt(qweight.mean() * wr.mean()))
            need(all(value[k].hex() == expected[k].hex() for k in SCALARS), "SYNTHETIC_ORIGINAL_ORACLE")
            need(a.shape == b.shape == br.shape == (qcount,), "LOCAL_AXIS")
            checks += 1
    print(json.dumps({"status": "RC_SHARED_QUERY_TARGET_PRIOR_CACHE_V1_E0_PASS", "synthetic_oracle_cases": checks, "natural_data_reads": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("e0", "freeze", "run", "validate"), required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    {"e0": e0, "freeze": freeze, "run": run, "validate": validate}[args.phase]()


if __name__ == "__main__":
    main()
