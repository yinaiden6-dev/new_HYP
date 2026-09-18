#!/usr/bin/env python3
"""Target-free TRAIN32 raw-token adapter for the isolated V7 mechanism.

No model is loaded and no query labels are read.  Content is recovered using
source token indices, including for coordinate controls.  A saved bank's
content binding and aligned FP32 token hash are exact.  CPU FP32 cosine replay
is checked against the original GPU FP32 feature with an analytic roundoff
bound; this check is explicitly not represented as bit-exact replay.

Public API: ``bundle = InputBundle(root)``; ``for query in bundle.iter_queries()``;
``bundle.prejoin_receipt()`` after all 32 records.  The returned query contains
sorted candidate keys; normalized FP64 query tokens; FP64 per-channel squared
residuals [128, Q, 128]; validity masks; source/current axes; and the complete
frozen maximal component families for REAL, C_BIND and P_COORD.  P_COORD
aliases REAL content tensors while its geometric families are regenerated.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
GENAUTH = "registry/rc_cycle_closed_region_generation_authority_v1_20260908.json"
GENAUTH_SHA = "5472d55fde3839d668c4cbb94917d9ec2611131a1f7840e1fa988e748a40b85c"
GENROOT = "results/rc_cycle_closed_region_generation_v1"
SANROOT = "results/rc_lth_p_only_sanitized_train_input_v2"
SANVALID_SHA = "50d39d7db0f7c2eaf41d3135480cff5c4e3fb1d6b3c62d8ae0dcde0273ac3877"
SIDEROOT = "results/rc_lth_p_only_reference_content_sidecar_v1"
SIDE_SHA = "c157cb1ecd7adc842b1bbb8661ded140da0e988643780db3d53de60ddcecfb16"
SIDE_RECEIPT_SHA = "e370130dbb32fe514403890bf896cec40e65aee1eeb79d9cb5dcbceb4deb9cb1"
SIDE_VALID_SHA = "2c9aa84bef51155f7f7863ea5dfd5b8ae2f6291a7e0da32300dc20b34fb21732"
CONTROLS = ("REAL", "C_BIND", "P_COORD")
FORBIDDEN = ("d1_mi", "d1-mi", "grozi", "ima++", "gisc_prerecall_universe")


def need(value: Any, code: str) -> None:
    if not bool(value):
        raise RuntimeError(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def logical(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def tensor_sha(value: Any) -> str:
    import torch
    value = torch.as_tensor(value).detach().cpu().contiguous()
    h = hashlib.sha256(str(value.dtype).encode("ascii"))
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_module(path: Path, name: str):
    if name in sys.modules:
        existing = sys.modules[name]
        need(Path(existing.__file__).resolve() == path.resolve(), "ATOM_CLASS_IMPORT_COLLISION")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "SOURCE_IMPORT_FAILURE")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform_grid(tokens, shape, orientation):
    """The eight original EXIF grid transforms, with no image decoding."""
    import torch
    grid = tokens.reshape(int(shape[0]), int(shape[1]), -1)
    if orientation == 1:
        out = grid
    elif orientation == 2:
        out = torch.flip(grid, (1,))
    elif orientation == 3:
        out = torch.flip(grid, (0, 1))
    elif orientation == 4:
        out = torch.flip(grid, (0,))
    elif orientation == 5:
        out = grid.transpose(0, 1)
    elif orientation == 6:
        out = torch.rot90(grid, -1, (0, 1))
    elif orientation == 7:
        out = torch.flip(grid.transpose(0, 1), (0, 1))
    elif orientation == 8:
        out = torch.rot90(grid, 1, (0, 1))
    else:
        raise RuntimeError("EXIF_ORIENTATION_INVALID")
    return out.reshape(-1, out.shape[-1]).contiguous(), tuple(out.shape[:2])


def aligned_tokens(asset, expected_sha, expected_shape):
    """Recover the exact producer tensor by its preexisting content hash.

    This is provenance reconstruction, not a search by recognition scores.
    Equal transforms of symmetric content are harmless: all accepted tensors
    have the same SHA and shape.  No transform can change the frozen target SHA.
    """
    import torch
    raw = torch.as_tensor(asset["tokens"]).detach().cpu().contiguous()
    need(raw.dtype == torch.float16 and raw.ndim == 2 and raw.shape[1] == 128,
         "SANITIZED_TOKEN_SCHEMA_DRIFT")
    need(tensor_sha(raw) == asset["tokens_sha256"], "SANITIZED_TOKEN_HASH_DRIFT")
    shape = tuple(int(x) for x in asset["grid_shape"])
    need(raw.shape[0] == math.prod(shape) and bool(torch.isfinite(raw).all()), "TOKEN_GRID_INVALID")
    # Original atom producer promotes serialized FP16 tokens to FP32.
    raw = raw.to(torch.float32)
    matches = []
    chosen = None
    for orientation in range(1, 9):
        value, transformed_shape = transform_grid(raw, shape, orientation)
        if transformed_shape == tuple(expected_shape) and tensor_sha(value) == expected_sha:
            if chosen is not None:
                need(torch.equal(chosen, value), "TOKEN_HASH_COLLISION")
            chosen = value
            matches.append(orientation)
    need(chosen is not None, "ALIGNED_TOKEN_SOURCE_NOT_RECOVERABLE")
    return chosen, matches


class LabelBarrier:
    def __init__(self, root: Path):
        self.closed = False
        self.blocked_read_attempts = 0
        self.label_paths = {
            str(root / "registry/rc_lth_p_only_natural_optimization_postjoin_authority_v1_20260907.json"),
            str(root / "registry/rc_lth_p_only_natural_optimization_postjoin_authority_v1_20260907.independent_validation.json"),
        }

    def audit(self, event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        name = os.path.abspath(os.fsdecode(args[0]))
        if any(term in name.lower() for term in FORBIDDEN):
            raise PermissionError("V7_PROTECTED_PATH")
        if name in self.label_paths and not self.closed:
            self.blocked_read_attempts += 1
            raise PermissionError("V7_LABEL_READ_BEFORE_COMPLETE_INPUT_CLOSURE")


class InputBundle:
    """Hash-closed input reader; does not join roles or choose a candidate."""

    def __init__(self, root: Path = ROOT, *, expected_manifest_sha256: str | None = None,
                 install_label_barrier: bool = True):
        self.root = Path(root).resolve()
        self.barrier = LabelBarrier(self.root)
        if install_label_barrier:
            sys.addaudithook(self.barrier.audit)
        self.bindings = {}
        self.completed = []
        self._started = False
        authority_path = self.bind(GENAUTH, GENAUTH_SHA)
        self.authority = read(authority_path)
        need(self.authority["status"] == "RC_CYCLE_CLOSED_REGION_GENERATION_AUTHORIZED", "GENERATION_SOURCE_NOT_AUTHORIZED")
        for name in ("core", "atom_class"):
            item = self.authority["sources"][name]
            self.bind(item["path"], item["sha256"])
        self.core = load_module(self.root / self.authority["sources"]["atom_class"]["path"],
                                "rc_lth_p_only_core_standalone_v1")
        self.generator = load_module(self.root / self.authority["sources"]["core"]["path"],
                                     "rc_cycle_closed_generation_v7_input_source")
        sanitized = read(self.bind(SANROOT + "/validation.json", SANVALID_SHA))
        need(sanitized["status"] == "RC_LTH_P_ONLY_SANITIZED_TRAIN_INPUTS_V2_VALIDATION_PASS"
             and sanitized["record_count"] == 32 and sanitized["candidate_occurrence_count"] == 4096
             and sanitized["unique_reference_count"] == 1790 and all(sanitized["checks"].values()),
             "SANITIZED_SOURCE_NOT_VALIDATED")
        self.bind(SIDEROOT + "/sidecar.json", SIDE_SHA)
        self.bind(SIDEROOT + "/receipt.json", SIDE_RECEIPT_SHA)
        sideval = read(self.bind(SIDEROOT + "/validation_v2.json", SIDE_VALID_SHA))
        need(sideval["sidecar_sha256"] == SIDE_SHA and sideval["receipt_sha256"] == SIDE_RECEIPT_SHA
             and sideval["entry_count"] == 1790 and all(sideval["checks"].values()), "REFERENCE_SIDECAR_INVALID")
        self.shards = []
        for item in self.authority["shards"]:
            shard = item["shard"]
            atom_path = self.bind(item["payload"]["path"], item["payload"]["sha256"])
            validation = read(self.bind(item["validation"]["path"], item["validation"]["sha256"]))
            need(validation["payload_sha256"] == item["payload"]["sha256"]
                 and validation["status"] == "RC_LTH_P_ONLY_NATURAL_ATOM_BANK_INDEPENDENT_VALIDATION_PASS"
                 and all(validation["checks"].values()), "ATOM_SOURCE_NOT_VALIDATED")
            srecord = sanitized["shards"][shard]
            need(srecord["shard_index"] == shard, "SANITIZED_SHARD_AXIS_DRIFT")
            p_path = self.bind(f"{SANROOT}/shard{shard:02d}/p_input.pt", srecord["p_input_sha256"])
            receipt_path = self.bind(f"{SANROOT}/shard{shard:02d}/receipt.json")
            receipt = read(receipt_path)
            need(receipt["p_input_sha256"] == srecord["p_input_sha256"]
                 and receipt["broker_sha256"] == srecord["broker_sha256"]
                 and receipt["source_seal"]["reference_sidecar_sha256"] == SIDE_SHA
                 and receipt["source_seal"]["reference_sidecar_receipt_sha256"] == SIDE_RECEIPT_SHA
                 and receipt["source_seal"]["reference_sidecar_validation_sha256"] == SIDE_VALID_SHA,
                 "SANITIZED_REFERENCE_LINEAGE_DRIFT")
            self.shards.append((shard, atom_path, p_path, receipt_path, srecord, item))
        seal = read(self.bind(GENROOT + "/prejoin_seal.json"))
        need(seal["status"] == "GENERATION_PREJOIN_FULL32_C128_CLOSED"
             and seal["authority_sha256"] == GENAUTH_SHA and seal["prejoin_label_reads"] == 0
             and len(seal["records"]) == 32, "GENERATION_PREJOIN_SEAL_INVALID")
        self.generation = {}
        for item in seal["records"]:
            path = self.bind(GENROOT + "/" + item["file"], item["sha256"])
            need(item["query_resource_key"] not in self.generation, "GENERATION_QUERY_DUPLICATE")
            self.generation[item["query_resource_key"]] = path
        self.manifest = {"schema": "RC_GROUPED_RESIDUAL_INPUT_SOURCES_V7", "query_count": 32,
                         "candidate_count": 128, "sources": sorted(self.bindings.values(), key=lambda x: x["path"])}
        self.manifest_sha256 = logical(self.manifest)
        if expected_manifest_sha256 is not None:
            need(self.manifest_sha256 == expected_manifest_sha256, "V7_INPUT_MANIFEST_DRIFT")

    def bind(self, relative: str, expected: str | None = None) -> Path:
        path = self.root / relative
        need(not path.is_symlink() and path.resolve().is_relative_to(self.root)
             and not any(term in str(path).lower() for term in FORBIDDEN), "UNSAFE_INPUT_SOURCE")
        actual = sha(path)
        if expected is not None:
            need(actual == expected, "INPUT_SOURCE_HASH_DRIFT:" + relative)
        self.bindings[relative] = {"path": relative, "sha256": actual}
        return path

    def iter_queries(self):
        import torch
        need(not self._started, "INPUT_BUNDLE_ITERATION_ALREADY_STARTED")
        self._started = True
        for shard, atom_path, p_path, receipt_path, srecord, item in self.shards:
            need(sha(atom_path) == item["payload"]["sha256"] and sha(p_path) == srecord["p_input_sha256"],
                 "SOURCE_CHANGED_AFTER_MANIFEST")
            payload = torch.load(atom_path, map_location="cpu", weights_only=False, mmap=True)
            raw = torch.load(p_path, map_location="cpu", weights_only=True, mmap=True)
            need(payload["access_vector"] == [0] * 12 and payload["record_count"] == 4
                 and payload["candidate_count_per_record"] == 128, "ATOM_PAYLOAD_SCHEMA_DRIFT")
            need(payload["source_seal"]["sanitized_p_input_sha256"] == srecord["p_input_sha256"]
                 and payload["source_seal"]["sanitized_broker_sha256"] == srecord["broker_sha256"]
                 and payload["source_seal"]["sanitized_receipt_sha256"] == sha(receipt_path)
                 and payload["source_seal"]["sanitized_validation_sha256"] == SANVALID_SHA,
                 "ATOM_SANITIZED_BINDING_DRIFT")
            raw_queries = {x["resource_key"]: x for x in raw["records"]}
            need(len(raw_queries) == 4 and set(raw_queries) == {r["resource_key"] for r in payload["records"]},
                 "QUERY_SOURCE_AXIS_DRIFT")
            for record in payload["records"]:
                query = self.build_query(record, raw_queries[record["resource_key"]], raw["reference_assets"],
                                         payload["source_seal"], shard)
                self.completed.append(query["source_receipt"])
                yield query
            del payload, raw
        need(len(self.completed) == 32 and len({x["query_resource_key"] for x in self.completed}) == 32,
             "FULL_TRAIN32_INPUT_CLOSURE_FAILED")

    def build_query(self, record, raw_query, assets, source_seal, shard):
        import torch
        import torch.nn.functional as F
        keys = tuple(sorted(record["candidate_keys"]))
        need(len(keys) == len(set(keys)) == 128 and set(keys) == set(raw_query["candidate_keys"]), "FULL_C128_AXIS_DRIFT")
        query_key = record["resource_key"]
        grid = tuple(record["query_grid_shape"])
        qraw, qorients = aligned_tokens(raw_query, record["query_tokens_sha256"], grid)
        need(torch.equal(qraw, record["query_tokens"]) and tensor_sha(record["query_tokens"]) == record["query_tokens_sha256"]
             and raw_query["content_sha256"] == record["query_content_sha256"], "QUERY_CONTENT_BINDING_DRIFT")
        qvalid = record["query_valid_axis"].clone()
        need(qvalid.dtype == torch.bool and bool(qvalid.all()) and tuple(qvalid.shape) == (math.prod(grid),),
             "V7_REQUIRES_DENSE_VALID_QUERY_GRID")
        need(bool((qraw.double().norm(dim=1) > 0).all()), "ZERO_QUERY_TOKEN")
        qtokens = F.normalize(qraw.to(torch.float64), dim=1)
        real = {b.candidate_resource_key: b for b in record["real_banks"]}
        cb = {b.candidate_resource_key: b for b in record["c_bind_banks"]}
        need(set(real) == set(cb) == set(keys), "BANK_CANDIDATE_AXIS_DRIFT")
        lookup = {k: i for i, k in enumerate(record["candidate_keys"])}
        donor_keys = dict(zip(record["candidate_keys"], record["c_bind_source_keys"]))
        need(set(donor_keys.values()) == set(keys) and all(k != d for k, d in donor_keys.items()), "C_BIND_DONOR_NOT_DERANGED")
        references, normalized_references, reference_receipts = {}, {}, []
        for key in keys:
            j = lookup[key]
            asset = assets[key]
            reference, orientations = aligned_tokens(asset, record["candidate_tokens_sha256"][j], real[key].reference_grid_shape)
            need(asset["content_sha256"] == record["candidate_content_sha256"][j], "REFERENCE_CONTENT_BINDING_DRIFT")
            need(bool((reference.double().norm(dim=1) > 0).all()), "ZERO_REFERENCE_TOKEN")
            references[key] = reference
            normalized_references[key] = F.normalize(reference.to(torch.float64), dim=1)
            reference_receipts.append({"reference_resource_key": key, "aligned_tokens_sha256": tensor_sha(reference),
                                       "matching_original_exif_transforms": orientations})
        coord = {k: self.core.coordinate_destroy_atom_bank(real[k], record["p_coord_derivation"]["namespace"])[0] for k in keys}
        banks = {"REAL": tuple(real[k] for k in keys), "C_BIND": tuple(cb[k] for k in keys),
                 "P_COORD": tuple(coord[k] for k in keys)}
        query_permutation = {}
        query_inverse_permutation = {}
        for control in CONTROLS:
            permutation = banks[control][0].query_coordinate_permutation
            need(all(torch.equal(b.query_coordinate_permutation, permutation) for b in banks[control]),
                 "CANDIDATE_DEPENDENT_QUERY_COORDINATE_CONTROL")
            need(torch.equal(permutation.sort().values, torch.arange(qtokens.shape[0])), "INVALID_QUERY_PERMUTATION")
            query_permutation[control] = permutation
            query_inverse_permutation[control] = torch.argsort(permutation)
            if control != "P_COORD":
                need(torch.equal(permutation, torch.arange(qtokens.shape[0])), "ORIGINAL_QUERY_COORDINATES_NOT_IDENTITY")
            need(all(bool(b.reference_valid_axis.any()) for b in banks[control]), "V7_EMPTY_REFERENCE_ASSET")
        for key in keys:
            need(real[key].reference_resource_key == key and cb[key].reference_resource_key == donor_keys[key],
                 "C_BIND_REFERENCE_RESOURCE_BINDING_DRIFT")
            expected_source_assignment = self.core.recompute_source_assignment_sha256(
                dataclasses.replace(real[donor_keys[key]], candidate_resource_key=key))
            need(cb[key].source_assignment_sha256 == expected_source_assignment
                 and cb[key].source_binding_sha256 == real[donor_keys[key]].source_binding_sha256,
                 "C_BIND_SOURCE_ASSIGNMENT_DRIFT")
        residuals, valid, families, axes = {}, {}, {}, {}
        cosine_receipts = []
        qn32 = F.normalize(qraw, dim=1)
        # u=2^-24. gamma_(8D+16) conservatively covers norm accumulation,
        # square root, division, product, and dot accumulation in FP32 at D=128.
        # Unit-norm dots have sum(abs(products)) <= 1. Tokens are represented
        # FP16 promoted to FP32, excluding underflow/overflow in this calculation.
        unit = 2.0 ** -24
        roundoff = ((8 * 128 + 16) * unit) / (1.0 - (8 * 128 + 16) * unit)
        qaxis = torch.arange(qtokens.shape[0], dtype=torch.long)
        for control in CONTROLS:
            if control == "P_COORD":
                residuals[control] = residuals["REAL"]
                valid[control] = valid["REAL"]
            else:
                residuals[control] = torch.empty((128, qtokens.shape[0], 128), dtype=torch.float64)
                valid[control] = torch.stack([bank.valid_mask for bank in banks[control]])
            families[control], axes[control] = [], []
            for ci, bank in enumerate(banks[control]):
                self.core.validate_atom_bank(bank)
                need(torch.equal(bank.source_query_indices, qaxis), "NONCANONICAL_SOURCE_QUERY_AXIS")
                refkey = bank.reference_resource_key
                reference = references[refkey]
                j = lookup[refkey]
                expected_binding = self.core.logical_sha256({
                    "reference_resource_key": refkey, "query_content_sha256": record["query_content_sha256"],
                    "query_tokens_sha256": record["query_tokens_sha256"],
                    "reference_content_sha256": record["candidate_content_sha256"][j],
                    "reference_tokens_sha256": record["candidate_tokens_sha256"][j],
                    "core_sha256": source_seal["core_sha256"], "roma_checkpoint_sha256": source_seal["roma_checkpoint_sha256"],
                })
                need(bank.source_binding_sha256 == expected_binding, "EXACT_RAW_TOKEN_BANK_SOURCE_BINDING_MISMATCH")
                # P_COORD changes current coordinate bindings, never content.
                if control != "P_COORD":
                    rn64 = normalized_references[refkey]
                    assigned64 = rn64[bank.source_reference_indices]
                    residuals[control][ci] = (qtokens[bank.source_query_indices] - assigned64).square()
                    rn32 = F.normalize(reference, dim=1)
                    replay32 = (qn32[bank.source_query_indices] * rn32[bank.source_reference_indices]).sum(1).clamp(-1, 1)
                    expected64 = (qtokens[bank.source_query_indices] * assigned64).sum(1).clamp(-1, 1)
                    saved = bank.features[:, 0].to(torch.float64)
                    need(bool((saved[~bank.valid_mask] == 0).all()), "INVALID_SAVED_COSINE_NOT_ZERO")
                    replay32 = torch.where(bank.valid_mask, replay32, torch.zeros_like(replay32))
                    expected64 = torch.where(bank.valid_mask, expected64, torch.zeros_like(expected64))
                    err_cpu = float((replay32.double() - saved).abs().max())
                    err_saved64 = float((saved - expected64).abs().max())
                    err_cpu64 = float((replay32.double() - expected64).abs().max())
                    need(err_saved64 <= roundoff and err_cpu64 <= roundoff and err_cpu <= 2 * roundoff,
                         "ASSIGNED_COSINE_FORWARD_ERROR_BOUND_FAILED")
                    cosine_receipts.append({"control": control, "candidate_resource_key": bank.candidate_resource_key,
                                            "fp32_cpu_saved_bit_exact": torch.equal(replay32, bank.features[:, 0]),
                                            "max_cpu_saved_abs_error": err_cpu, "max_saved_fp64_abs_error": err_saved64,
                                            "max_cpu_fp64_abs_error": err_cpu64})
                else:
                    original = real[bank.candidate_resource_key]
                    need(torch.equal(bank.source_reference_indices, original.source_reference_indices)
                         and torch.equal(bank.valid_mask, original.valid_mask)
                         and torch.equal(bank.features, original.features), "P_COORD_CHANGED_SOURCE_CONTENT")
                family = self.generator.enumerate_cycle_closed_regions(bank)
                families[control].append(family)
                axes[control].append({"source_query_indices": bank.source_query_indices,
                                      "source_reference_indices": bank.source_reference_indices,
                                      "query_indices": bank.query_indices, "reference_indices": bank.reference_indices,
                                      "query_rc": bank.query_rc, "reference_rc": bank.reference_rc,
                                      "reference_grid_shape": tuple(bank.reference_grid_shape),
                                      "reference_resource_key": refkey})
        saved_generation = read(self.generation[query_key])
        need(saved_generation["authority_sha256"] == GENAUTH_SHA, "GENERATION_LINEAGE_DRIFT")
        for control in CONTROLS:
            by_key = {x["candidate_resource_key"]: x for x in saved_generation["families"][control]}
            need(families[control] == [by_key[k] for k in keys], "COMPLETE_COMPONENT_GENERATION_REPLAY_MISMATCH")
        tensor_hashes = {"query_tokens_fp64": tensor_sha(qtokens)}
        tensor_hashes["query_permutation_axis"] = logical({c: tensor_sha(p) for c, p in query_permutation.items()})
        tensor_hashes["reference_tokens_fp64_axis"] = logical([(k, tensor_sha(normalized_references[k])) for k in keys])
        for control in ("REAL", "C_BIND"):
            tensor_hashes[control + "_residual_squared"] = tensor_sha(residuals[control])
            tensor_hashes[control + "_valid"] = tensor_sha(valid[control])
        receipt = {"query_resource_key": query_key, "shard": shard, "candidate_keys": list(keys),
                   "input_manifest_sha256": self.manifest_sha256, "tensor_sha256": tensor_hashes,
                   "family_axis_sha256": logical(families), "query_matching_original_exif_transforms": qorients,
                   "reference_alignment_sha256": logical(reference_receipts),
                   "cosine_replay": {"status": "EXACT_CONTENT_HASH_BINDING_AND_ANALYTIC_FP32_ERROR_BOUND_PASS",
                                      "saved_feature_bit_exact_all": all(x["fp32_cpu_saved_bit_exact"] for x in cosine_receipts),
                                      "single_implementation_absolute_roundoff_bound": roundoff,
                                      "max_cpu_saved_abs_error": max(x["max_cpu_saved_abs_error"] for x in cosine_receipts),
                                      "max_saved_fp64_abs_error": max(x["max_saved_fp64_abs_error"] for x in cosine_receipts),
                                      "records_sha256": logical(cosine_receipts)},
                   "P_COORD_source_content_aliases_REAL": True, "label_reads": 0,
                   "new_model_forward_count": 0, "generation_replay_all_controls": True}
        receipt["logical_sha256"] = logical(receipt)
        return {"query_resource_key": query_key, "candidate_keys": keys, "query_grid_shape": grid,
                "qtokens": qtokens, "query_tokens": qtokens, "query_valid_axis": qvalid,
                "query_coordinate_axis": record["query_coordinate_axis"],
                "query_permutation": query_permutation, "query_inverse_permutation": query_inverse_permutation,
                "query_only_features": record["query_only_features"].double(),
                "residual_squared": residuals, "valid": valid, "families": families,
                "reference_tokens": {control: tuple(normalized_references[b.reference_resource_key] for b in banks[control])
                                     for control in CONTROLS},
                "reference_valid": {control: tuple(b.reference_valid_axis for b in banks[control])
                                    for control in CONTROLS},
                "axes": axes, "banks_by_control": banks, "source_receipt": receipt}

    def prejoin_receipt(self):
        need(len(self.completed) == 32 and len({r["query_resource_key"] for r in self.completed}) == 32,
             "V7_PREJOIN_CLOSURE_REQUIRES_ALL32")
        need(self.barrier.blocked_read_attempts == 0, "V7_PREJOIN_EARLY_LABEL_READ_ATTEMPT")
        value = {"status": "RC_GROUPED_RESIDUAL_INPUTS_V7_FULL32_C128_PREJOIN_CLOSED",
                 "input_manifest_sha256": self.manifest_sha256,
                 "records": sorted(self.completed, key=lambda x: x["query_resource_key"]),
                 "prejoin_label_reads": 0, "formal_query_count": 0, "new_model_forward_count": 0}
        value["logical_sha256"] = logical(value)
        self.barrier.closed = True
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--stage", choices=("inspect", "smoke", "prepare"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    bundle = InputBundle(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "source_manifest.json").write_bytes(canonical(bundle.manifest) + b"\n")
    if args.stage == "inspect":
        result = {"status": "V7_TARGET_FREE_SOURCE_MANIFEST_INSPECTED", "manifest_sha256": bundle.manifest_sha256,
                  "source_count": len(bundle.bindings), "tensor_payloads_materialized": 0, "label_reads": 0}
    else:
        for i, query in enumerate(bundle.iter_queries()):
            (args.out / (query["query_resource_key"] + ".json")).write_bytes(canonical(query["source_receipt"]) + b"\n")
            print(json.dumps({"event": "V7_RAW_TOKEN_INPUT_REPLAY", "query_count": i + 1,
                              "query": query["query_resource_key"]}), flush=True)
            if args.stage == "smoke":
                break
        result = (bundle.prejoin_receipt() if args.stage == "prepare" else
                  {"status": "V7_ONE_QUERY_SCHEMA_SMOKE_ONLY", "query_count": 1, "full_input_closure": False,
                   "label_reads": 0, "source_receipt": query["source_receipt"]})
    (args.out / "result.json").write_bytes(canonical(result) + b"\n")
    print(json.dumps({"status": result["status"], "out": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
