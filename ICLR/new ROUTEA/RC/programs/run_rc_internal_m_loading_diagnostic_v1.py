#!/usr/bin/env python3
"""One TRAIN image diagnostic; never changes or advances the frozen M pilot.

Hold image/frame/reference tokens/action fixed and vary processor and retrieval
LoRA dtype independently. Then replay the original from_pretrained entrypoint.
This is a loading diagnosis, not an experiment result or permission to loosen
the original source-token/source-content/source-action parity requirements.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "programs"), str(ROOT / "src")]
import run_rc_prellm_m_pilot_v1 as P


def tensor_binding(value):
    import torch
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())


def compare(a, b):
    import torch
    if a.shape != b.shape:
        return dict(shape_equal=False, a_shape=list(a.shape), b_shape=list(b.shape))
    delta = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return dict(shape_equal=True, exact=torch.equal(a.cpu(), b.cpu()),
                max_abs=float(delta.max()) if delta.numel() else 0.,
                mean_abs=float(delta.mean()) if delta.numel() else 0.)


def weight_summary(model):
    """Complete LoRA fingerprints and the small retrieval projection, not base copies."""
    values = {}
    for name, value in model.named_parameters():
        if "lora_" in name or "custom_text_proj" in name:
            canonical = name.removeprefix("base_model.model.")
            values[canonical] = tensor_binding(value)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return dict(parameters=values,
                adapter_dtypes=sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n}),
                backbone_dtype=str(base.dtype),
                attention=str(getattr(base.config, "_attn_implementation", None)),
                text_attention=str(getattr(base.config.text_config, "_attn_implementation", None)),
                vision_attention=str(getattr(base.config.vision_config, "_attn_implementation", None)))


def restore_lora_source_fp32(model, model_path):
    """Restore original FP32 values; promoting BF16-rounded tensors is insufficient."""
    import torch
    from safetensors.torch import load_file
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    source = load_file(str(Path(model_path) / "adapter_model.safetensors"), device="cpu")
    converted = {k.replace("base_model.model.model.", "base_model.model.language_model.", 1): v
                 for k, v in source.items()}
    P.need(all("lora_" in k for k in converted), "DIAGNOSTIC_EXPECTS_LORA_ONLY_CHECKPOINT")
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.data = parameter.data.float()
    loaded = set_peft_model_state_dict(model, converted, adapter_name="default")
    P.need(not loaded.unexpected_keys, "FP32_LORA_UNEXPECTED")
    actual = get_peft_model_state_dict(model, adapter_name="default")
    P.need(set(actual) == set(converted), "FP32_LORA_KEY_COVERAGE")
    P.need(all(v.dtype == torch.float32 and torch.equal(v.cpu(), converted[k])
               for k, v in actual.items()), "FP32_LORA_SOURCE_VALUES")
    model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", default="H593-979d4429602e6c4a69aa6724")
    parser.add_argument("--skip-native", action="store_true")
    args = parser.parse_args()
    P.need(os.environ.get("SLURM_JOB_ID"), "GPU_DIAGNOSTIC_REQUIRES_SLURM")
    a, manifest = P.guard("loading-diagnostic")
    torch = P.torch_setup(a)
    P.need(torch.cuda.is_available(), "GPU_REQUIRED")
    rows = [r for r in manifest["train_rows"] if r["query_id"] == args.query_id]
    P.need(len(rows) == 1, "ONE_PREEXISTING_TRAIN_QUERY_REQUIRED")
    # Only input, model, and target-free action fields are consulted below.
    keys = ("query_id", "image_path", "source_image_sha256", "frame", "query_tokens",
            "reference_tokens", "M", "L0", "raw_scores", "winner_index", "candidate_ids")
    row = {key: rows[0][key] for key in keys}
    P.need(P.bind(row["image_path"])["sha256"] == row["source_image_sha256"], "IMAGE_SHA")
    receipt_path = P.OUT / "model_source_validation.json"
    receipt = P.read(receipt_path)
    P.need(receipt["authority"] == P.bind(P.AUTH), "ORIGINAL_SOURCE_VALIDATION_AUTH")
    P.need(receipt["status"] == "MODEL_SOURCE_HASHES_PASS", "ORIGINAL_WEIGHT_HASHES_REQUIRED")
    by_path = {x["binding"]["path"]: x for x in receipt["files"]}
    for binding in manifest["encoder"]["model_sources"].values():
        previous = by_path[binding["path"]]
        P.need(previous["binding"] == binding, "SOURCE_BINDING_DRIFT")
        stat = Path(binding["path"]).stat()
        P.need(previous["stat"] == [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns],
               "SOURCE_STAT_DRIFT:" + binding["path"])
    out = ROOT / "results/rc_internal_m_loading_diagnostic_v1" / str(os.environ["SLURM_JOB_ID"])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status="DIAGNOSTIC_RUNNING", authority=P.bind(P.AUTH),
                  program=P.bind(__file__), source_validation=P.bind(receipt_path),
                  query_id=row["query_id"], query_frame=row["frame"],
                  query_image_sha256=row["source_image_sha256"],
                  gate=a["source_parity"], variants={}, training_updates=0,
                  target_or_label_values_consulted=0, reference_encoder_forwards=0,
                  original_pilot_modified=False,
                  versions={n: importlib.metadata.version(n) for n in
                            ("torch", "transformers", "peft", "colpali-engine", "pillow", "torchvision")},
                  device=torch.cuda.get_device_name())

    def checkpoint():
        report["seconds"] = time.monotonic() - start
        P.write(out / "report.json", report)

    checkpoint()
    from PIL import Image, ImageOps
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from rc_prellm_m_adapter_v1 import _load_frozen_colnomic_weights
    with Image.open(row["image_path"]) as opened:
        if row["frame"] in ("EXIF_TRANSPOSED", "EXIF_NORMALIZED", "EXIF_ORIENTED_BEFORE_RESIZE"):
            opened = ImageOps.exif_transpose(opened)
        else:
            P.need(row["frame"] == "DECODED_RAW_BEFORE_EXIF", "FRAME")
        image = opened.convert("RGB")
    batches = {}
    report["processors"] = {}
    for label, kwargs in (("explicit_false", dict(use_fast=False)), ("original_default", {})):
        processor = ColQwen2_5_Processor.from_pretrained(a["model"], local_files_only=True, **kwargs)
        batch = processor.process_images([image])
        report["processors"][label] = dict(
            processor_class=type(processor).__module__ + "." + type(processor).__name__,
            image_processor_class=type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__,
            backend=str(getattr(processor.image_processor, "backend", None)),
            image_processor_config=processor.image_processor.to_dict(),
            tensors={k: tensor_binding(v) for k, v in batch.items() if isinstance(v, torch.Tensor)})
        batches[label] = batch.to("cuda")
    left, right = batches["explicit_false"], batches["original_default"]
    report["processor_comparison"] = {k: compare(left[k], right[k]) for k in left
                                       if isinstance(left[k], torch.Tensor) and k in right}
    checkpoint()
    prior = P.old_tokens(row["query_tokens"]).to(device="cuda", dtype=torch.float32)
    refs = P.references(row)
    with torch.no_grad():
        old_l0 = torch.stack([P.content_score(prior, ref) for ref in refs]).cpu()
    P.need(float((old_l0 - torch.tensor(row["L0"], dtype=torch.float64)).abs().max()) < 2e-10,
           "ORIGINAL_L0_REPLAY")
    old_decision = P.decision(row, old_l0, manifest)
    report["original_prediction_position"] = old_decision["prediction_position"]
    outputs = {}

    def forward_variant(model, name, processor_label):
        batch = batches[processor_label]
        with torch.no_grad():
            full = model(**batch)
            mask = batch["input_ids"].eq(model.config.image_token_id)
            tokens = full[mask].float()
            result = dict(processor=processor_label, token_binding=tensor_binding(tokens),
                          token_shape_equal=list(tokens.shape) == list(prior.shape))
            if result["token_shape_equal"]:
                cosines = torch.nn.functional.cosine_similarity(prior, tokens, dim=-1)
                result.update(token_mean_cosine=float(cosines.mean()), token_min_cosine=float(cosines.min()),
                              tokens_half_exact=torch.equal(tokens.half(), prior.half()))
            fresh = torch.stack([P.content_score(tokens, ref) for ref in refs]).cpu()
        decision = P.decision(row, fresh, manifest)
        result.update(content_max_drift=float((fresh - old_l0).abs().max()),
                      prediction_position=decision["prediction_position"],
                      source_action_exact=decision["prediction_position"] == old_decision["prediction_position"],
                      fresh_L0=fresh.tolist())
        result["original_gates_pass"] = (result.get("token_mean_cosine", -1.) >= a["source_parity"]["mean_cosine_min"]
            and result["content_max_drift"] <= a["source_parity"]["max_content_error"]
            and result["source_action_exact"])
        for previous, previous_tokens in outputs.items():
            result.setdefault("against_variants", {})[previous] = compare(tokens.cpu(), previous_tokens)
        outputs[name] = tokens.cpu()
        P.save(out / (name + ".pt"), dict(query_id=row["query_id"], tokens=tokens.cpu()))
        report["variants"][name] = result
        checkpoint()
        P.emit(event="LOADING_DIAGNOSTIC_VARIANT", name=name,
               token_mean_cosine=result.get("token_mean_cosine"),
               content_max_drift=result["content_max_drift"], original_gates_pass=result["original_gates_pass"])

    model, loading = _load_frozen_colnomic_weights(a["model"], "cuda", attention=a["attention"],
                                                 verify_weight_hashes=False)
    report["strict_loading"] = loading
    report["strict_bf16_weights"] = weight_summary(model)
    for processor_label in batches:
        forward_variant(model, "strict_bf16_" + processor_label, processor_label)
    restore_lora_source_fp32(model, a["model"])
    report["strict_source_fp32_weights"] = weight_summary(model)
    for processor_label in batches:
        forward_variant(model, "strict_source_fp32_" + processor_label, processor_label)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not args.skip_native:
        try:
            # Original qualified call, adding only offline and loading-report options.
            model, loading = ColQwen2_5.from_pretrained(
                a["model"], torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
            model = model.to("cuda").eval().requires_grad_(False)
            report["native_loading"] = loading
            report["native_weights"] = weight_summary(model)
            for processor_label in batches:
                forward_variant(model, "native_" + processor_label, processor_label)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as error:
            report["native_error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            checkpoint()
    report["status"] = "DIAGNOSTIC_COMPLETE_NO_PILOT_ADVANCE"
    report["passing_variants"] = [k for k, v in report["variants"].items() if v["original_gates_pass"]]
    checkpoint()
    P.emit(status=report["status"], report=P.bind(out / "report.json"), passing_variants=report["passing_variants"])


if __name__ == "__main__":
    main()
