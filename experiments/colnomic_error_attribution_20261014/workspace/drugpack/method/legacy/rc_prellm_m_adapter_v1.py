"""Candidate-quality residual adapters for the local ColQwen2_5 implementation.

The frozen vision merger produces spatially ordered [patch, text_hidden] tokens.
``prellm`` adapts those tokens before the original language-model forward;
``postllm`` adapts only image positions immediately before custom_text_proj.
Both arms use the same module and a raw RoMa mass, never a candidate identity.
This module does not modify installed model source or any backbone parameter.

The cache path reproduces the pixel unpadding in the installed ColQwen2_5 and
keeps input_ids, image_grid_thw, mm_token_type_ids, positions and attention_mask.
The language forward must run with autograd enabled for prellm training, even
when every backbone parameter is frozen. Reference final tokens stay external.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _file_binding(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def _load_frozen_colnomic_weights(
    model_path: str | Path, device: str | torch.device, attention: str = "sdpa",
    *, verify_weight_hashes: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strict local legacy-key loading without Transformers' add_adapter API.

    The installed PEFT 0.18.1 cannot use Transformers 5.6.2 add_adapter (which
    requires >=0.18.2). PEFT's own get_peft_model/set_peft_model_state_dict APIs
    are callable locally. This path checks every retrieval/base weight rather
    than accepting a model with silently missing embedding or projection keys.
    """
    from colpali_engine.models import ColQwen2_5
    from peft import PeftConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_path = Path(model_path).resolve()
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text())
    base_path = Path(adapter_config["base_model_name_or_path"]).resolve()
    if not base_path.is_dir():
        raise ValueError("the adapter must refer to an existing local base checkpoint")
    index_path = base_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        base_files = sorted({base_path / value for value in index["weight_map"].values()})
        configuration_files = [base_path / "config.json", index_path, adapter_config_path]
    else:
        base_files = [base_path / "model.safetensors"]
        configuration_files = [base_path / "config.json", adapter_config_path]
    adapter_weights_path = adapter_path / "adapter_model.safetensors"
    sources = [_file_binding(path) for path in configuration_files]
    weight_files = base_files + [adapter_weights_path]
    if verify_weight_hashes:
        sources.extend(_file_binding(path) for path in weight_files)
    weight_stats = []
    for path in weight_files:
        stat = path.stat()
        weight_stats.append({"path": str(path), "device": stat.st_dev, "inode": stat.st_ino,
                             "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    key_mapping = {
        r"^model\.embed_tokens\.": "language_model.embed_tokens.",
        r"^model\.norm\.": "language_model.norm.",
        r"^model\.layers\.": "language_model.layers.",
        r"^model\.language_model\.": "language_model.",
        r"^model\.visual\.": "visual.",
    }
    base, information = ColQwen2_5.from_pretrained(
        str(base_path), dtype=torch.bfloat16, attn_implementation=attention,
        local_files_only=True, key_mapping=key_mapping, output_loading_info=True,
    )
    if information.get("missing_keys") or information.get("mismatched_keys") or information.get("error_msgs"):
        raise ValueError(f"incomplete base/retrieval checkpoint load: {information}")
    unexpected = set(information.get("unexpected_keys", []))
    if unexpected - {"lm_head.weight"}:
        raise ValueError(f"unexpected base checkpoint keys: {sorted(unexpected)}")
    weights = load_file(str(adapter_weights_path), device="cpu")
    converted: dict[str, torch.Tensor] = {}
    for original, tensor in weights.items():
        name = original.replace("base_model.model.model.", "base_model.model.language_model.", 1)
        if name in converted:
            raise ValueError("duplicate adapter weight after legacy-key conversion")
        converted[name] = tensor
    config = PeftConfig.from_pretrained(str(adapter_path), local_files_only=True)
    config.inference_mode = True
    # Accommodate saved retrieval-projector modules if a checkpoint contains
    # them; the actual ColNomic checkpoint currently has projector LoRA only.
    saved_projection = any("custom_text_proj" in key and "lora_" not in key for key in converted)
    if saved_projection:
        config.modules_to_save = sorted(set(config.modules_to_save or []) | {"custom_text_proj"})
    # Match the BF16 from_pretrained backbone path rather than silently enabling
    # PEFT's optional FP32 upcast. The original checkpoint tensors remain bound.
    encoder = get_peft_model(base, config, autocast_adapter_dtype=False)
    loaded = set_peft_model_state_dict(encoder, dict(converted), adapter_name="default")
    required = {name for name, _ in encoder.named_parameters() if "lora_" in name or "modules_to_save.default" in name}
    missing_adapter = sorted(set(loaded.missing_keys) & required)
    if missing_adapter or loaded.unexpected_keys:
        raise ValueError(f"incomplete retrieval LoRA load: missing={missing_adapter}, unexpected={loaded.unexpected_keys}")
    actual = get_peft_model_state_dict(encoder, adapter_name="default")
    if set(actual) != set(converted):
        raise ValueError(f"adapter key coverage mismatch: missing={sorted(set(converted)-set(actual))}, extra={sorted(set(actual)-set(converted))}")
    for name, tensor in actual.items():
        if not torch.equal(tensor.detach().cpu(), converted[name].to(dtype=tensor.dtype)):
            raise ValueError(f"retrieval LoRA tensor was not loaded exactly: {name}")
    encoder = encoder.to(device).eval().requires_grad_(False)
    report = {
        "status": "STRICT_LOCAL_COLNOMIC_WEIGHTS_PASS",
        "loader": "explicit base key mapping plus local PEFT get_peft_model and set_peft_model_state_dict",
        "base_path": str(base_path), "adapter_path": str(adapter_path), "sources": sources,
        "weight_hashes_verified_here": verify_weight_hashes, "weight_file_stats": weight_stats,
        "base_key_mapping": key_mapping, "base_missing_keys": [],
        "base_unused_keys": sorted(unexpected), "adapter_source_tensor_count": len(weights),
        "adapter_loaded_tensor_count": len(actual), "adapter_tensor_value_parity": True,
        "adapter_dtype": sorted({str(tensor.dtype) for tensor in actual.values()}),
        "adapter_source_dtype": sorted({str(tensor.dtype) for tensor in weights.values()}),
        "modules_to_save": config.modules_to_save, "backbone_dtype": str(base.dtype),
        "attention": attention, "trainable_parameters": 0,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "colpali-engine")},
    }
    return encoder, report


def load_frozen_colnomic(
    model_path: str | Path,
    device: str | torch.device,
    attention: str = "sdpa",
    use_fast: bool = False,
    *,
    verify_weight_hashes: bool = True,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load all frozen base/retrieval weights and processor with an audit report."""
    from colpali_engine.models import ColQwen2_5_Processor

    encoder, report = _load_frozen_colnomic_weights(model_path, device, attention, verify_weight_hashes=verify_weight_hashes)
    processor = ColQwen2_5_Processor.from_pretrained(str(model_path), local_files_only=True, use_fast=use_fast)
    report["processor_use_fast"] = use_fast
    report["processor_class"] = type(processor).__name__
    report["processor_sources"] = [_file_binding(path) for path in sorted(Path(model_path).glob("*.json"))
                                   if path.name != "adapter_config.json"]
    return encoder, processor, report


class QualityResidualAdapter(nn.Module):
    """Zero-initialized, content-dependent residual conditioned on log(M).

    ``mass_log_mean/std`` must be fixed using training masses only. ``constant``
    and ``none`` set the standardized condition to zero, retaining identical
    parameter shapes. They are the same functional no-information control;
    they must not be presented as two independent scientific ablations.
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        bottleneck: int = 16,
        conditioning: str = "real",
        mass_log_mean: float = 0.0,
        mass_log_std: float = 1.0,
        mass_epsilon: float = 1e-8,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if conditioning not in {"real", "constant", "none"}:
            raise ValueError("conditioning must be real, constant, or none")
        if hidden_size < 1 or bottleneck < 1:
            raise ValueError("hidden_size and bottleneck must be positive")
        constants = torch.tensor([mass_log_mean, mass_log_std, mass_epsilon, residual_scale])
        if not bool(torch.isfinite(constants).all()) or mass_log_std <= 0 or mass_epsilon <= 0 or residual_scale <= 0:
            raise ValueError("finite constants, positive std, epsilon and residual_scale required")
        self.hidden_size = int(hidden_size)
        self.bottleneck = int(bottleneck)
        self.conditioning = conditioning
        self.register_buffer("mass_log_mean", torch.tensor(float(mass_log_mean)))
        self.register_buffer("mass_log_std", torch.tensor(float(mass_log_std)))
        self.register_buffer("mass_epsilon", torch.tensor(float(mass_epsilon)))
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        self.down = nn.Linear(hidden_size + 1, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def standardized_mass(self, mass: Any, tokens: torch.Tensor) -> torch.Tensor:
        # RoMa is frozen; labels and a teacher gradient cannot enter this branch.
        values = torch.as_tensor(mass, device=tokens.device, dtype=torch.float32).detach()
        if values.ndim > 2 or (values.ndim == 2 and values.shape[-1] != 1):
            raise ValueError("mass must be a scalar or one scalar per patch")
        values = values.reshape(-1)
        if values.numel() not in {1, tokens.shape[0]}:
            raise ValueError("mass length does not match merged image-token count")
        if not bool(torch.isfinite(values).all()) or bool(((values < 0) | (values > 1)).any()):
            raise ValueError("raw visibility mass must be finite and in [0,1]")
        if values.numel() == 1:
            values = values.expand(tokens.shape[0])
        if self.conditioning in {"constant", "none"}:
            return torch.zeros_like(values).unsqueeze(-1)
        return ((values.clamp_min(self.mass_epsilon).log() - self.mass_log_mean) / self.mass_log_std).unsqueeze(-1)

    def forward(self, tokens: torch.Tensor, mass: Any) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != self.hidden_size or tokens.shape[0] == 0:
            raise ValueError("tokens must have shape [positive patch count, hidden_size]")
        if tokens.device != self.down.weight.device:
            raise ValueError("adapter and visual tokens must be on the same device")
        if self.down.weight.dtype != torch.float32:
            raise ValueError("keep the small adapter in float32; only backbone tokens use bfloat16")
        # Avoid autocast silently reducing the trainable small-module precision.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            condition = self.standardized_mass(mass, tokens)
            normalized = F.layer_norm(tokens.float(), (self.hidden_size,))
            features = torch.cat((normalized, condition), dim=-1)
            residual = self.up(F.gelu(self.down(features))) * self.residual_scale
        return tokens + residual.to(dtype=tokens.dtype)


def find_colnomic_base(encoder: nn.Module) -> nn.Module:
    """Locate the actual ColQwen2_5, including a possible PEFT wrapper."""
    matches = [module for module in encoder.modules() if type(module).__name__ == "ColQwen2_5"]
    if len(matches) != 1:
        raise ValueError(f"expected one local ColQwen2_5, found {len(matches)}")
    base = matches[0]
    if not all(hasattr(base, key) for key in ("visual", "language_model", "custom_text_proj")):
        raise ValueError("local ColQwen2_5 insertion interface changed")
    return base


def find_visual_module(encoder: nn.Module) -> nn.Module:
    return find_colnomic_base(encoder).visual


def _image_counts(grid: torch.Tensor, spatial_merge_size: int) -> list[int]:
    if grid.ndim != 2 or grid.shape[-1] != 3 or grid.shape[0] == 0:
        raise ValueError("image_grid_thw must have shape [positive image count,3]")
    if bool((grid <= 0).any()) or bool((grid[:, 0] != 1).any()):
        raise ValueError("this image-only adapter requires positive grids with temporal size one")
    if bool((grid[:, 1:] % spatial_merge_size != 0).any()):
        raise ValueError("image grid must be divisible by the visual spatial merge size")
    return (grid.prod(-1) // spatial_merge_size**2).tolist()


def _patch_masses(mass: Any, counts: list[int], device: torch.device) -> torch.Tensor:
    masses = torch.as_tensor(mass, device=device, dtype=torch.float32).detach().reshape(-1)
    if masses.numel() != len(counts):
        raise ValueError("provide exactly one candidate-pair mass per image, in processor order")
    return masses.repeat_interleave(torch.tensor(counts, device=device, dtype=torch.long))


@torch.no_grad()
def cache_merged_visual_tokens(encoder: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """Run the frozen visual encoder once, returning reusable pre-LLM tokens.

    ``batch`` is the original processor.process_images output on model device.
    Save grid/image lineage with any persistent cache; this tensor itself carries
    no authorization to reuse a cache for another image or preprocessing path.
    """
    base = find_colnomic_base(encoder)
    if batch.get("pixel_values_videos") is not None:
        raise ValueError("video inputs are outside the image retrieval contract")
    grid = batch["image_grid_thw"]
    counts = _image_counts(grid, base.visual.spatial_merge_size)
    pixels = batch["pixel_values"]
    if pixels.ndim != 3 or pixels.shape[0] != len(counts):
        raise ValueError("expected the ColQwen processor's padded [image,patch,feature] pixels")
    offsets = grid[:, 1] * grid[:, 2]
    if any(int(offset) > pixels.shape[1] for offset in offsets):
        raise ValueError("padded pixels do not contain every grid patch")
    unpadded = torch.cat([sequence[:int(offset)] for sequence, offset in zip(pixels, offsets)], dim=0)
    output = base.visual(unpadded.to(dtype=base.visual.dtype), grid_thw=grid, return_dict=True)
    features = output.pooler_output
    if features.shape != (sum(counts), base.custom_text_proj.in_features):
        raise ValueError("visual output does not match the local merger-to-LLM interface")
    return features.detach()


@contextmanager
def _postllm_hook(
    projection: nn.Module,
    adapter: QualityResidualAdapter,
    image_mask: torch.Tensor,
    patch_masses: torch.Tensor,
) -> Iterator[None]:
    calls = 0

    def before_projection(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        if calls != 1 or len(args) != 1 or args[0].shape[:2] != image_mask.shape:
            raise ValueError("unexpected custom_text_proj call structure")
        hidden = args[0]
        changed = hidden.clone()
        changed[image_mask] = adapter(hidden[image_mask], patch_masses)
        return (changed,)

    handle = projection.register_forward_pre_hook(before_projection)
    try:
        yield
        if calls != 1:
            raise ValueError("custom_text_proj was not called exactly once")
    finally:
        handle.remove()


def conditioned_colnomic_forward(
    encoder: nn.Module,
    batch: Mapping[str, Any],
    adapter: QualityResidualAdapter | None,
    mass: Any,
    *,
    location: str = "prellm",
    cached_visual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return original full-sequence retrieval tokens with image-only injection.

    Frozen backbone parameters are the caller's responsibility. The adapter is
    deliberately held outside encoder so freezing encoder cannot freeze it.
    Each candidate requires a fresh language-model forward for prellm. Do not
    surround training calls with no_grad/inference_mode. Calls sharing one
    encoder are sequential because the postllm arm uses a temporary module hook.
    """
    if location not in {"prellm", "postllm"}:
        raise ValueError("location must be prellm or postllm")
    base = find_colnomic_base(encoder)
    if any(batch.get(key) is not None for key in ("pixel_values_videos", "video_grid_thw", "inputs_embeds", "past_key_values")):
        raise ValueError("expected a fresh image processor batch without video, cache, or input embeddings")
    grid = batch["image_grid_thw"]
    counts = _image_counts(grid, base.visual.spatial_merge_size)
    visual = cache_merged_visual_tokens(encoder, batch) if cached_visual is None else cached_visual
    if visual.ndim != 2 or visual.shape != (sum(counts), base.custom_text_proj.in_features):
        raise ValueError("cached visual shape does not match this processor batch")
    if visual.requires_grad:
        raise ValueError("cached visual features must be detached from the frozen visual encoder")
    patch_masses = _patch_masses(mass, counts, visual.device)
    image_mask = batch["input_ids"].eq(base.config.image_token_id)
    if int(image_mask.sum()) != visual.shape[0]:
        raise ValueError("image placeholder count does not match cached visual features")
    if bool((image_mask & ~batch["attention_mask"].bool()).any()):
        raise ValueError("image positions cannot be padding")
    if adapter is not None and location == "prellm":
        visual = adapter(visual, patch_masses)
    input_embeds = base.get_input_embeddings()(batch["input_ids"])
    input_embeds = input_embeds.masked_scatter(
        image_mask.unsqueeze(-1).expand_as(input_embeds),
        visual.to(device=input_embeds.device, dtype=input_embeds.dtype),
    )
    # Keep input_ids and grid metadata so the original 3-D RoPE computation runs.
    kwargs = {key: value for key, value in batch.items() if key != "pixel_values"}
    kwargs["inputs_embeds"] = input_embeds
    if adapter is not None and location == "postllm":
        with _postllm_hook(base.custom_text_proj, adapter, image_mask, patch_masses):
            encoded = encoder(**kwargs)
    else:
        encoded = encoder(**kwargs)
    # ColQwen's original image-only mask is guarded by pixel_values presence;
    # restore it after using cached input embeddings without pixel_values.
    if base.mask_non_image_embeddings:
        encoded = encoded * image_mask.unsqueeze(-1)
    return encoded


@torch.no_grad()
def capture_frozen_hidden(
    encoder: nn.Module,
    batch: Mapping[str, Any],
    cached_visual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cache the frozen last language hidden state, before the 128-D projector.

    Only postllm may reuse this tensor across candidate masses. The prellm arm
    changes language inputs and therefore must recompute language hidden states.
    """
    base = find_colnomic_base(encoder)
    captured: list[torch.Tensor] = []

    def capture(module: nn.Module, args: tuple[Any, ...]) -> None:
        if len(args) != 1:
            raise ValueError("unexpected custom_text_proj arguments")
        captured.append(args[0].detach().clone())

    handle = base.custom_text_proj.register_forward_pre_hook(capture)
    try:
        conditioned_colnomic_forward(encoder, batch, None, [0.0] * len(batch["image_grid_thw"]), cached_visual=cached_visual)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("expected one frozen last hidden-state capture")
    return captured[0]


def project_cached_hidden(
    encoder: nn.Module,
    hidden: torch.Tensor,
    batch: Mapping[str, Any],
    adapter: QualityResidualAdapter | None,
    mass: Any,
) -> torch.Tensor:
    """Postllm candidate forward: reuse language state, adapt image tokens only."""
    base = find_colnomic_base(encoder)
    if hidden.requires_grad:
        raise ValueError("cached frozen language hidden state must be detached")
    if hidden.ndim != 3 or hidden.shape[:2] != batch["input_ids"].shape or hidden.shape[-1] != base.custom_text_proj.in_features:
        raise ValueError("cached language hidden shape does not match the input batch")
    counts = _image_counts(batch["image_grid_thw"], base.visual.spatial_merge_size)
    image_mask = batch["input_ids"].eq(base.config.image_token_id)
    if int(image_mask.sum()) != sum(counts):
        raise ValueError("image placeholders and grids disagree")
    if bool((image_mask & ~batch["attention_mask"].bool()).any()):
        raise ValueError("image positions cannot be padding")
    if adapter is not None:
        patch_masses = _patch_masses(mass, counts, hidden.device)
        adapted = hidden.clone()
        adapted[image_mask] = adapter(hidden[image_mask], patch_masses)
    else:
        adapted = hidden
    # Deliberately mirror the installed ColQwen2_5 projection/normalization,
    # including its dtype behavior, instead of introducing a different scorer.
    encoded = base.custom_text_proj(adapted)
    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    encoded = encoded * batch["attention_mask"].unsqueeze(-1)
    if base.mask_non_image_embeddings:
        encoded = encoded * image_mask.unsqueeze(-1)
    return encoded
