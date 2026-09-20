"""Frozen Hugging Face DINOv2-with-registers input path for RCDE V1.2.

This module is deliberately limited to representation materialisation.  It
does not know labels, candidates, ranks, losses, or retrieval results.  The
three contracts implemented here are:

* verify and load the pinned local ``dinov2-with-registers-base`` checkpoint
  without consulting the network or a home-directory cache;
* build the frozen, aspect-preserving full-frame 518px input and its purely
  geometric valid-patch mask; and
* extract post-final-LayerNorm patch tokens from 1-based blocks 3/6/9/12,
  removing CLS and all four register tokens before producing an FP16 cache.

Hugging Face ``hidden_states`` contains the embedding output at index zero.
Consequently outputs after 1-based blocks 3/6/9/12 are at indices 3/6/9/12.
This differs syntactically from the official DINO ``get_intermediate_layers``
zero-based block list ``[2, 5, 8, 11]`` but is semantically identical.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import torch


SCHEMA_VERSION = "dino_rcde_hf_with_registers_v1_2"
MODEL_ID = "facebook/dinov2-with-registers-base"
MODEL_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
MODEL_DIRECTORY_NAME = "dinov2-with-registers-base"
MODEL_FILE_SHA256 = {
    "config.json": "6af60aa760138fc90db0ba37b7701730b140b9bf1742514412d03050e72bf7e0",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "model.safetensors": "7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362",
}
# Official DINO get_intermediate_layers arguments versus Hugging Face output
# indexing.  HF prepends the embedding state at hidden_states[0].
BLOCK_INDICES_ZERO_BASED = (2, 5, 8, 11)
LAYER_NUMBERS = (3, 6, 9, 12)
HIDDEN_STATE_INDICES = tuple(index + 1 for index in BLOCK_INDICES_ZERO_BASED)
LONG_SIDE = 518
PATCH_SIZE = 14
NUM_REGISTERS = 4
HIDDEN_SIZE = 768
LAYER_NORM_EPS = 1.0e-6
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_VALID_TOKEN_NORM = 1.0e-12
FORBIDDEN_ATTENTION_KERNEL_SUBSTRINGS = (
    "flash_attn", "flash_attention", "flash_sdp", "_scaled_dot_product_flash_attention",
    "memory_efficient_attention", "mem_efficient_sdp", "efficient_attention", "xformers",
)


class RCDEWithRegistersError(RuntimeError):
    """A frozen model, preprocessing, token, or cache invariant failed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU payload exactly."""

    if not isinstance(value, torch.Tensor):
        raise RCDEWithRegistersError("tensor hash input is not a tensor")
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(header)
    # Flatten first so zero-dimensional tensors (for example a pair logit) have
    # a legal byte view.  This does not change row-major bytes for non-scalars.
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def default_model_root() -> Path:
    """Resolve the workspace-local pinned model without HOME lookup."""

    workspace = Path(__file__).resolve().parents[5]
    return workspace / "models" / "downloaded_models" / MODEL_DIRECTORY_NAME


def _metadata_revision(path: Path) -> str:
    metadata = (
        path
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    if not metadata.is_file():
        raise RCDEWithRegistersError("checkpoint revision metadata is missing")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RCDEWithRegistersError("checkpoint revision metadata is empty")
    return lines[0].strip()


def audit_local_checkpoint(model_root: Path | str | None = None) -> dict[str, Any]:
    """Fail closed unless the exact local with-registers checkpoint is present."""

    root = Path(model_root) if model_root is not None else default_model_root()
    root = root.resolve()
    if root.name != MODEL_DIRECTORY_NAME or not root.is_dir():
        raise RCDEWithRegistersError("pinned local DINOv2-with-registers directory is absent")
    observed_hashes: dict[str, str] = {}
    for name, expected in MODEL_FILE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise RCDEWithRegistersError(f"pinned checkpoint file is missing: {name}")
        observed = file_sha256(path)
        if observed != expected:
            raise RCDEWithRegistersError(f"pinned checkpoint hash drift: {name}")
        observed_hashes[name] = observed
    revision = _metadata_revision(root)
    if revision != MODEL_REVISION:
        raise RCDEWithRegistersError("pinned checkpoint revision drift")
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RCDEWithRegistersError("invalid pinned model config") from error
    expected_config = {
        "model_type": "dinov2_with_registers",
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": 12,
        "num_register_tokens": NUM_REGISTERS,
        "patch_size": PATCH_SIZE,
        "layer_norm_eps": LAYER_NORM_EPS,
        "interpolate_antialias": True,
        "interpolate_offset": 0.0,
    }
    for field, expected in expected_config.items():
        if config.get(field) != expected:
            raise RCDEWithRegistersError(f"pinned model config drift: {field}")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "revision": revision,
        "model_root": str(root),
        "file_sha256": observed_hashes,
        "config_contract": expected_config,
        "layers_1_based": list(LAYER_NUMBERS),
        "official_block_indices_zero_based": list(BLOCK_INDICES_ZERO_BASED),
        "hf_hidden_state_indices": list(HIDDEN_STATE_INDICES),
        "processor_file_is_provenance_only": True,
        "hf_processor_used_for_pixels": False,
        "network_allowed": False,
        "home_cache_allowed": False,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return receipt


@contextmanager
def _offline_nonhome_environment(cache_root: Path) -> Iterator[None]:
    """Force HF into an ephemeral non-HOME, offline cache during model load."""

    cache_root = cache_root.resolve()
    configured_home = os.environ.get("HOME")
    if configured_home:
        home = Path(configured_home).resolve()
        if cache_root == home or home in cache_root.parents:
            raise RCDEWithRegistersError("Hugging Face cache root may not be inside HOME")
    names = ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME",
             "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    old = {name: os.environ.get(name) for name in names}
    cache_root.mkdir(parents=True, exist_ok=True)
    values = {
        "HF_HOME": str(cache_root / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def configure_deterministic_eager_inference() -> dict[str, Any]:
    """Apply the frozen eager/math-only inference backend contract."""

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled())
        if hasattr(torch.backends.cuda, "flash_sdp_enabled") else False,
        "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled())
        if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled") else False,
        "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled())
        if hasattr(torch.backends.cuda, "math_sdp_enabled") else True,
    }


def audit_attention_kernel_event_names(event_names: Sequence[str]) -> dict[str, Any]:
    """Classify a real profiler trace; configuration flags alone are insufficient."""

    names = sorted({str(name) for name in event_names})
    forbidden = sorted({
        name for name in names
        if any(pattern in name.lower() for pattern in FORBIDDEN_ATTENTION_KERNEL_SUBSTRINGS)
    })
    if not names:
        raise RCDEWithRegistersError("attention execution trace is empty")
    if forbidden:
        raise RCDEWithRegistersError(
            "forbidden attention kernel appeared in execution trace: " + ",".join(forbidden)
        )
    return {
        "execution_trace_observed": True,
        "event_name_count": len(names),
        "event_names_sha256": canonical_sha256(names),
        "forbidden_kernel_patterns": list(FORBIDDEN_ATTENTION_KERNEL_SUBSTRINGS),
        "forbidden_kernel_event_names": [],
        "forbidden_kernel_execution_trace_gate": True,
    }


def audit_attention_kernel_execution(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Profile one real model forward and fail closed if tracing is unavailable."""

    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # A small, patch-aligned full model input exercises the selected
        # attention implementation without consuming natural data.
        probe = torch.zeros((1, 3, 28, 28), dtype=torch.float32, device=device)
        with torch.inference_mode(), torch.profiler.profile(
            activities=activities, record_shapes=False, profile_memory=False, with_stack=False,
        ) as profiler:
            model(pixel_values=probe, output_hidden_states=True, return_dict=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        receipt = audit_attention_kernel_event_names([event.key for event in profiler.key_averages()])
    except RCDEWithRegistersError:
        raise
    except Exception as error:  # fail closed by contract
        raise RCDEWithRegistersError("attention execution trace could not be completed") from error
    receipt.update({
        "trace_device_type": device.type,
        "trace_probe_shape": [1, 3, 28, 28],
        "trace_output_hidden_states_requested": True,
    })
    return receipt


def runtime_version_receipt(transformers_version: str, device: torch.device) -> dict[str, Any]:
    versions = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": (
            torch.backends.cudnn.version()
            if device.type == "cuda" and hasattr(torch.backends, "cudnn") else None
        ),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python_executable_sha256": file_sha256(Path(sys.executable).resolve()),
    }
    versions["logical_sha256"] = canonical_sha256(versions)
    return versions


@dataclass(frozen=True)
class FrozenBackbone:
    model: torch.nn.Module
    receipt: Mapping[str, Any]


def load_frozen_backbone(
    model_root: Path | str | None = None,
    *,
    device: torch.device | str = "cpu",
    cache_root: Path | str | None = None,
) -> FrozenBackbone:
    """Load exactly one local HF model in FP32, eager mode, fully frozen.

    ``from_pretrained`` receives a resolved local directory and
    ``local_files_only=True``.  During the call every Hugging Face cache is
    redirected to a disposable non-HOME directory and both offline switches
    are set.  No ``AutoModel`` or remote-code path is used.
    """

    checkpoint = audit_local_checkpoint(model_root)
    root = Path(checkpoint["model_root"])
    backend = configure_deterministic_eager_inference()

    def load_under_offline_environment(nonhome_cache: Path) -> tuple[torch.nn.Module, str]:
        # Import under the same redirected cache environment as from_pretrained;
        # this prevents even import-time library probes from consulting HOME.
        with _offline_nonhome_environment(nonhome_cache):
            try:
                from transformers import (
                    Dinov2WithRegistersModel,
                    __version__ as transformers_version,
                )
            except ImportError as error:  # pragma: no cover - environment qualification
                raise RCDEWithRegistersError(
                    "transformers with DINOv2-with-registers support is required"
                ) from error
            loaded = Dinov2WithRegistersModel.from_pretrained(
                str(root), local_files_only=True, use_safetensors=True,
                dtype=torch.float32, attn_implementation="eager",
            )
        return loaded, transformers_version

    if cache_root is None:
        with tempfile.TemporaryDirectory(prefix="rcde-hf-offline-", dir="/tmp") as temporary:
            model, transformers_version = load_under_offline_environment(Path(temporary))
    else:
        model, transformers_version = load_under_offline_environment(
            Path(cache_root).resolve()
        )
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)
    audit_hf_model_semantics(model)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters() if parameter.is_floating_point()):
        raise RCDEWithRegistersError("backbone parameter dtype is not float32")
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise RCDEWithRegistersError("backbone did not remain frozen in eval mode")
    execution_trace = audit_attention_kernel_execution(model, resolved_device)
    versions = runtime_version_receipt(transformers_version, resolved_device)
    receipt = dict(checkpoint)
    receipt.update(
        {
            "transformers_version": transformers_version,
            "model_class": type(model).__name__,
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "parameter_dtype": "torch.float32",
            "parameter_requires_grad_count": sum(int(p.requires_grad) for p in model.parameters()),
            "backend": backend,
            "attention_kernel_execution_trace": execution_trace,
            "runtime_versions": versions,
            "from_pretrained_local_files_only": True,
            "from_pretrained_use_safetensors": True,
        }
    )
    if receipt["attention_implementation"] != "eager":
        raise RCDEWithRegistersError("HF attention implementation is not eager")
    receipt["logical_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "logical_sha256"}
    )
    return FrozenBackbone(model=model, receipt=receipt)


def audit_hf_model_semantics(model: torch.nn.Module) -> dict[str, Any]:
    """Check the HF model surface required for official-equivalent extraction."""

    config = getattr(model, "config", None)
    layernorm = getattr(model, "layernorm", None)
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    expected = {
        "model_type": "dinov2_with_registers",
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": 12,
        "num_register_tokens": NUM_REGISTERS,
        "patch_size": PATCH_SIZE,
        "layer_norm_eps": LAYER_NORM_EPS,
        "interpolate_antialias": True,
        "interpolate_offset": 0.0,
    }
    if config is None:
        raise RCDEWithRegistersError("model has no HF DINOv2-with-registers config")
    for field, value in expected.items():
        if getattr(config, field, None) != value:
            raise RCDEWithRegistersError(f"loaded HF model semantic drift: {field}")
    if not isinstance(layernorm, torch.nn.LayerNorm):
        raise RCDEWithRegistersError("loaded model has no final LayerNorm")
    if layernorm.normalized_shape != (HIDDEN_SIZE,) or layernorm.eps != LAYER_NORM_EPS:
        raise RCDEWithRegistersError("loaded model final LayerNorm semantic drift")
    if layers is None or len(layers) != 12:
        raise RCDEWithRegistersError("loaded model encoder depth drift")
    return {
        "hf_hidden_states_include_embedding_at_index_zero": True,
        "selected_hidden_state_indices": list(HIDDEN_STATE_INDICES),
        "official_block_indices_zero_based": list(BLOCK_INDICES_ZERO_BASED),
        "selected_blocks_1_based": list(LAYER_NUMBERS),
        "final_layernorm_applied_to_each_selected_full_sequence": True,
        "removed_prefix_tokens": 1 + NUM_REGISTERS,
    }


def _round_half_up(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise RCDEWithRegistersError("invalid resize extent")
    return max(1, int(math.floor(value + 0.5)))


def exif_orient_without_metadata_rewrite(image: Any) -> tuple[Any, int]:
    """Apply tag 274 without serialising unrelated, possibly malformed EXIF.

    ``PIL.ImageOps.exif_transpose`` rewrites the full EXIF payload after the
    pixel transform.  Some frozen gallery images contain malformed unrelated
    rational fields, so that rewrite can fail even when orientation is valid.
    Reading only tag 274 and applying the standard transform is semantically
    identical for pixels and avoids touching any other metadata.
    """

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise RCDEWithRegistersError("Pillow is required") from error
    try:
        raw = image.getexif().get(274, 1)
        orientation = int(raw)
    except (AttributeError, TypeError, ValueError, OSError):
        orientation = 1
    if orientation not in range(1, 9):
        orientation = 1
    transforms = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    return (image.transpose(transforms[orientation]) if orientation in transforms else image.copy()), orientation


@dataclass(frozen=True)
class CanonicalGeometry:
    decoded_hw: tuple[int, int]
    oriented_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    grid_hw: tuple[int, int]
    valid_patch_mask: torch.Tensor
    scale_xy: tuple[float, float]

    def __post_init__(self) -> None:
        pairs = (self.decoded_hw, self.oriented_hw, self.resized_hw, self.padded_hw, self.grid_hw)
        if any(len(pair) != 2 or min(pair) <= 0 for pair in pairs):
            raise RCDEWithRegistersError("canonical geometry has invalid dimensions")
        if max(self.resized_hw) != LONG_SIDE:
            raise RCDEWithRegistersError("canonical resize long side drift")
        if self.padded_hw != (
            math.ceil(self.resized_hw[0] / PATCH_SIZE) * PATCH_SIZE,
            math.ceil(self.resized_hw[1] / PATCH_SIZE) * PATCH_SIZE,
        ):
            raise RCDEWithRegistersError("canonical bottom/right padding drift")
        if self.grid_hw != (self.padded_hw[0] // PATCH_SIZE, self.padded_hw[1] // PATCH_SIZE):
            raise RCDEWithRegistersError("canonical grid shape drift")
        mask = torch.as_tensor(self.valid_patch_mask, dtype=torch.bool).detach().cpu().contiguous()
        expected = torch.zeros(self.grid_hw, dtype=torch.bool)
        expected[: self.resized_hw[0] // PATCH_SIZE, : self.resized_hw[1] // PATCH_SIZE] = True
        if not torch.equal(mask, expected):
            raise RCDEWithRegistersError("valid mask is not the complete-footprint geometry mask")
        object.__setattr__(self, "valid_patch_mask", mask)

    def receipt(self) -> dict[str, Any]:
        sx, sy = self.scale_xy
        boundary_affine = [[sx / PATCH_SIZE, 0.0, 0.0], [0.0, sy / PATCH_SIZE, 0.0], [0.0, 0.0, 1.0]]
        return {
            "schema_version": SCHEMA_VERSION,
            "decoded_hw_before_exif": list(self.decoded_hw),
            "oriented_hw_after_exif": list(self.oriented_hw),
            "resized_hw": list(self.resized_hw),
            "padded_hw": list(self.padded_hw),
            "grid_hw": list(self.grid_hw),
            "scale_xy": [sx, sy],
            "padding_tblr": [0, self.padded_hw[0] - self.resized_hw[0], 0,
                             self.padded_hw[1] - self.resized_hw[1]],
            "oriented_pixel_boundary_to_patch_grid_boundary_affine": boundary_affine,
            "resize_rule": "long_side_518_round_half_up_aspect_preserving_no_crop",
            "resize_kernel": "torchvision_bicubic_antialias_true_align_corners_false",
            "padding_rule": "right_bottom_rgb_imagenet_mean_before_normalization",
            "validity_rule": "complete_14x14_footprint_inside_resized_rectangle",
            "valid_patch_count": int(self.valid_patch_mask.sum()),
            "valid_patch_mask_sha256": tensor_sha256(self.valid_patch_mask),
            "geometry_mask_is_target_or_foreground_supervision": False,
        }


def build_canonical_geometry(
    oriented_hw: tuple[int, int], *, decoded_hw: tuple[int, int] | None = None
) -> CanonicalGeometry:
    h, w = (int(oriented_hw[0]), int(oriented_hw[1]))
    if h <= 0 or w <= 0:
        raise RCDEWithRegistersError("image has an empty dimension")
    scale = LONG_SIDE / max(h, w)
    resized = (_round_half_up(h * scale), _round_half_up(w * scale))
    padded = (
        math.ceil(resized[0] / PATCH_SIZE) * PATCH_SIZE,
        math.ceil(resized[1] / PATCH_SIZE) * PATCH_SIZE,
    )
    grid = (padded[0] // PATCH_SIZE, padded[1] // PATCH_SIZE)
    valid = torch.zeros(grid, dtype=torch.bool)
    valid[: resized[0] // PATCH_SIZE, : resized[1] // PATCH_SIZE] = True
    return CanonicalGeometry(
        decoded_hw=decoded_hw or (h, w),
        oriented_hw=(h, w),
        resized_hw=resized,
        padded_hw=padded,
        grid_hw=grid,
        valid_patch_mask=valid,
        scale_xy=(resized[1] / w, resized[0] / h),
    )


def preprocess_canonical_full_frame(
    image: Any, *, raw_image_sha256: str | None = None
) -> tuple[torch.Tensor, CanonicalGeometry, dict[str, Any]]:
    """EXIF-orient, RGB-convert, full-frame resize, mean-pad, normalize."""

    try:
        import numpy as np
        import PIL
        from PIL import Image, features
        import torchvision
        from torchvision.transforms.functional import InterpolationMode, resize
    except ImportError as error:  # pragma: no cover - environment qualification
        raise RCDEWithRegistersError("Pillow, NumPy, and torchvision are required") from error
    if not isinstance(image, Image.Image):
        raise RCDEWithRegistersError("canonical preprocessor requires one PIL image")
    decoded_hw = (int(image.height), int(image.width))
    oriented_raw, exif_orientation = exif_orient_without_metadata_rewrite(image)
    oriented = oriented_raw.convert("RGB")
    geometry = build_canonical_geometry(
        (int(oriented.height), int(oriented.width)), decoded_hw=decoded_hw
    )
    array = np.asarray(oriented, dtype=np.uint8).copy()
    rgb_u8 = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    rgb = rgb_u8.to(torch.float32).div(255.0)
    resized = resize(
        rgb,
        list(geometry.resized_hw),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    mean = resized.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = resized.new_tensor(IMAGENET_STD)[:, None, None]
    padded = mean.expand(3, *geometry.padded_hw).clone()
    padded[:, : geometry.resized_hw[0], : geometry.resized_hw[1]] = resized
    pixels = ((padded - mean) / std).unsqueeze(0).contiguous()
    if pixels.dtype != torch.float32 or not bool(torch.isfinite(pixels).all()):
        raise RCDEWithRegistersError("canonical pixel tensor is not finite float32")
    # Mean-colour padding must become exact normalized zero.
    if geometry.padded_hw[0] > geometry.resized_hw[0] and not torch.equal(
        pixels[..., geometry.resized_hw[0] :, :],
        torch.zeros_like(pixels[..., geometry.resized_hw[0] :, :]),
    ):
        raise RCDEWithRegistersError("bottom padding is not exact normalized zero")
    if geometry.padded_hw[1] > geometry.resized_hw[1] and not torch.equal(
        pixels[..., :, geometry.resized_hw[1] :],
        torch.zeros_like(pixels[..., :, geometry.resized_hw[1] :]),
    ):
        raise RCDEWithRegistersError("right padding is not exact normalized zero")
    receipt = geometry.receipt()
    preprocessing_versions = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "libjpeg": features.version_codec("jpg"),
    }
    preprocessing_versions["logical_sha256"] = canonical_sha256(preprocessing_versions)
    receipt.update(
        {
            "raw_image_sha256": raw_image_sha256,
            "oriented_rgb_u8_sha256": tensor_sha256(rgb_u8),
            "pixel_values_fp32_sha256": tensor_sha256(pixels),
            "exif_transpose_applied_before_geometry": True,
            "exif_orientation_tag_274": exif_orientation,
            "unrelated_exif_metadata_rewritten": False,
            "input_mode_after_conversion": "RGB",
            "imagenet_mean": list(IMAGENET_MEAN),
            "imagenet_std": list(IMAGENET_STD),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "pillow_version": PIL.__version__,
            "libjpeg_version": features.version_codec("jpg"),
            "preprocessing_runtime_versions": preprocessing_versions,
        }
    )
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return pixels, geometry, receipt


def preprocess_canonical_image_path(
    image_path: Path | str,
) -> tuple[torch.Tensor, CanonicalGeometry, dict[str, Any]]:
    """Decode a local image while binding the exact source bytes."""

    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise RCDEWithRegistersError("Pillow is required") from error
    source = Path(image_path)
    payload = source.read_bytes()
    with Image.open(BytesIO(payload)) as image:
        return preprocess_canonical_full_frame(image, raw_image_sha256=_sha256_bytes(payload))


@dataclass(frozen=True)
class IntermediateTokenBatch:
    tokens_fp32: torch.Tensor  # [B,4,N,768], all patch positions retained
    valid_patch_mask: torch.Tensor  # [B,Hg,Wg]
    special_token_norms_fp32: torch.Tensor  # [B,4,5], receipt-only
    last_hidden_state_closure_exact: bool
    last_hidden_state_closure_max_abs: float
    layer_numbers: tuple[int, ...] = LAYER_NUMBERS

    def __post_init__(self) -> None:
        tokens = self.tokens_fp32.detach().cpu().contiguous()
        valid = self.valid_patch_mask.detach().cpu().bool().contiguous()
        special = self.special_token_norms_fp32.detach().cpu().contiguous()
        if (
            tokens.dtype != torch.float32
            or tokens.ndim != 4
            or tokens.shape[1] != 4
            or tokens.shape[3] != HIDDEN_SIZE
            or valid.ndim != 3
            or tokens.shape[0] != valid.shape[0]
            or tokens.shape[2] != valid.shape[1] * valid.shape[2]
            or special.shape != (tokens.shape[0], 4, 1 + NUM_REGISTERS)
            or not bool(torch.isfinite(tokens).all())
            or not bool(torch.isfinite(special).all())
            or self.last_hidden_state_closure_exact is not True
            or not math.isfinite(float(self.last_hidden_state_closure_max_abs))
            or float(self.last_hidden_state_closure_max_abs) != 0.0
        ):
            raise RCDEWithRegistersError("intermediate token batch contract drift")
        flat_valid = valid.flatten(1)
        norms = torch.linalg.vector_norm(tokens, dim=-1)
        if not bool(norms[flat_valid[:, None, :].expand_as(norms)].gt(MIN_VALID_TOKEN_NORM).all()):
            raise RCDEWithRegistersError("geometry-valid DINO token has zero/near-zero norm")
        object.__setattr__(self, "tokens_fp32", tokens)
        object.__setattr__(self, "valid_patch_mask", valid)
        object.__setattr__(self, "special_token_norms_fp32", special)


def extract_intermediate_patch_tokens(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    valid_patch_mask: torch.Tensor,
) -> IntermediateTokenBatch:
    """Extract official-equivalent post-norm blocks 3/6/9/12 patch tokens."""

    audit_hf_model_semantics(model)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RCDEWithRegistersError("DINO backbone must be frozen and eval-mode")
    pixels = torch.as_tensor(pixel_values)
    if pixels.dtype != torch.float32 or pixels.ndim != 4 or pixels.shape[1] != 3:
        raise RCDEWithRegistersError("pixel_values must be float32 [B,3,H,W]")
    if not bool(torch.isfinite(pixels).all()):
        raise RCDEWithRegistersError("pixel_values contains nonfinite values")
    if pixels.shape[-2] % PATCH_SIZE or pixels.shape[-1] % PATCH_SIZE:
        raise RCDEWithRegistersError("pixel dimensions are not patch aligned")
    valid = torch.as_tensor(valid_patch_mask, dtype=torch.bool)
    if valid.ndim == 2:
        valid = valid.unsqueeze(0)
    expected_grid = (pixels.shape[-2] // PATCH_SIZE, pixels.shape[-1] // PATCH_SIZE)
    if valid.shape != (pixels.shape[0], *expected_grid):
        raise RCDEWithRegistersError("valid mask does not match pixel batch/grid")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = pixels.device
    with torch.inference_mode():
        output = model(
            pixel_values=pixels.to(device=device, dtype=torch.float32),
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = getattr(output, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) != 13:
        raise RCDEWithRegistersError("HF hidden_states must contain stem plus 12 block outputs")
    prefix = 1 + NUM_REGISTERS
    patch_count = expected_grid[0] * expected_grid[1]
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if (
        not isinstance(last_hidden_state, torch.Tensor)
        or last_hidden_state.dtype != torch.float32
        or last_hidden_state.shape != (pixels.shape[0], prefix + patch_count, HIDDEN_SIZE)
        or not bool(torch.isfinite(last_hidden_state).all())
    ):
        raise RCDEWithRegistersError("HF last_hidden_state semantic drift")
    patches: list[torch.Tensor] = []
    special_norms: list[torch.Tensor] = []
    for hidden_index in HIDDEN_STATE_INDICES:
        sequence = hidden_states[hidden_index]
        if (
            not isinstance(sequence, torch.Tensor)
            or sequence.dtype != torch.float32
            or sequence.shape != (pixels.shape[0], prefix + patch_count, HIDDEN_SIZE)
            or not bool(torch.isfinite(sequence).all())
        ):
            raise RCDEWithRegistersError(f"HF hidden-state semantic drift at index {hidden_index}")
        # This is the checkpoint's one final LayerNorm, deliberately applied to
        # the complete sequence before CLS/register removal for every layer.
        normalized = model.layernorm(sequence)
        if normalized.dtype != torch.float32 or not bool(torch.isfinite(normalized).all()):
            raise RCDEWithRegistersError("post-final-LayerNorm sequence is nonfinite or non-FP32")
        special_norms.append(torch.linalg.vector_norm(normalized[:, :prefix], dim=-1))
        patches.append(normalized[:, prefix:])
    final_normalized = model.layernorm(hidden_states[HIDDEN_STATE_INDICES[-1]])
    closure_delta = (final_normalized - last_hidden_state).abs()
    closure_max_abs = float(closure_delta.max())
    closure_exact = torch.equal(final_normalized, last_hidden_state)
    if not closure_exact:
        raise RCDEWithRegistersError(
            f"final normalized hidden state does not exactly close to HF last_hidden_state: {closure_max_abs}"
        )
    return IntermediateTokenBatch(
        tokens_fp32=torch.stack(patches, dim=1).detach().cpu(),
        valid_patch_mask=valid.detach().cpu(),
        special_token_norms_fp32=torch.stack(special_norms, dim=1).detach().cpu(),
        last_hidden_state_closure_exact=closure_exact,
        last_hidden_state_closure_max_abs=closure_max_abs,
    )


@dataclass(frozen=True)
class FP16TokenCache:
    tokens_fp16: torch.Tensor
    valid_patch_mask: torch.Tensor
    receipt: Mapping[str, Any]


def build_fp16_cache(
    batch: IntermediateTokenBatch,
    *,
    geometry_receipts: Sequence[Mapping[str, Any]],
    model_receipt_sha256: str,
) -> FP16TokenCache:
    """Perform the only permitted FP32-to-FP16 cache boundary."""

    if len(geometry_receipts) != batch.tokens_fp32.shape[0]:
        raise RCDEWithRegistersError("geometry receipt count does not match token batch")
    for index, geometry in enumerate(geometry_receipts):
        if geometry.get("grid_hw") != list(batch.valid_patch_mask.shape[1:]):
            raise RCDEWithRegistersError(f"geometry grid mismatch at batch row {index}")
        if geometry.get("valid_patch_mask_sha256") != tensor_sha256(batch.valid_patch_mask[index]):
            raise RCDEWithRegistersError(f"geometry valid-mask hash mismatch at batch row {index}")
    cached = batch.tokens_fp32.to(torch.float16).contiguous()
    if not bool(torch.isfinite(cached).all()):
        raise RCDEWithRegistersError("FP32 to FP16 cast produced nonfinite tokens")
    restored = cached.to(torch.float32)
    cast_error = (batch.tokens_fp32 - restored).abs()
    valid = batch.valid_patch_mask.flatten(1)
    source_norms = torch.linalg.vector_norm(batch.tokens_fp32, dim=-1)
    cached_norms = torch.linalg.vector_norm(restored, dim=-1)
    expanded_valid = valid[:, None, :].expand_as(source_norms)
    if not bool(cached_norms[expanded_valid].gt(MIN_VALID_TOKEN_NORM).all()):
        raise RCDEWithRegistersError("FP16 cache erased a geometry-valid token")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cache_dtype": "torch.float16",
        "read_dtype": "torch.float32",
        "layer_numbers_1_based": list(batch.layer_numbers),
        "shape": list(cached.shape),
        "tokens_fp32_source_sha256": tensor_sha256(batch.tokens_fp32),
        "tokens_fp16_payload_sha256": tensor_sha256(cached),
        "valid_patch_mask_sha256": tensor_sha256(batch.valid_patch_mask),
        "special_token_norms_receipt_only_sha256": tensor_sha256(batch.special_token_norms_fp32),
        "last_hidden_state_closure_exact": batch.last_hidden_state_closure_exact,
        "last_hidden_state_closure_max_abs": batch.last_hidden_state_closure_max_abs,
        "geometry_logical_sha256": [item.get("logical_sha256") for item in geometry_receipts],
        "model_receipt_sha256": model_receipt_sha256,
        "cast_replay_exact": torch.equal(cached, batch.tokens_fp32.to(torch.float16)),
        "max_abs_fp32_to_fp16_roundtrip_error": float(cast_error.max()),
        "valid_fp32_norm_min": float(source_norms[expanded_valid].min()),
        "valid_fp16_restored_norm_min": float(cached_norms[expanded_valid].min()),
        "cls_and_register_tokens_in_payload": False,
        "invalid_geometry_tokens_retained_for_grid_alignment": True,
        "invalid_geometry_tokens_must_be_masked_downstream": True,
        "invalid_geometry_tokens_are_decoder_evidence": False,
    }
    receipt["logical_sha256"] = canonical_sha256(receipt)
    return FP16TokenCache(
        tokens_fp16=cached,
        valid_patch_mask=batch.valid_patch_mask.clone(),
        receipt=receipt,
    )


def validate_fp16_cache(cache: FP16TokenCache) -> None:
    """Independently validate a cache object before restoring FP32 tokens."""

    tokens = cache.tokens_fp16.detach().cpu().contiguous()
    valid = cache.valid_patch_mask.detach().cpu().bool().contiguous()
    receipt = dict(cache.receipt)
    if (
        tokens.dtype != torch.float16
        or tokens.ndim != 4
        or tokens.shape[1] != 4
        or tokens.shape[-1] != HIDDEN_SIZE
    ):
        raise RCDEWithRegistersError("FP16 cache tensor contract drift")
    if valid.ndim != 3 or valid.shape[0] != tokens.shape[0] or valid.shape[1] * valid.shape[2] != tokens.shape[2]:
        raise RCDEWithRegistersError("FP16 cache valid-mask geometry drift")
    if not bool(torch.isfinite(tokens).all()):
        raise RCDEWithRegistersError("FP16 cache contains nonfinite values")
    if receipt.get("shape") != list(tokens.shape):
        raise RCDEWithRegistersError("FP16 cache receipt shape drift")
    if receipt.get("tokens_fp16_payload_sha256") != tensor_sha256(tokens):
        raise RCDEWithRegistersError("FP16 cache payload hash drift")
    if receipt.get("valid_patch_mask_sha256") != tensor_sha256(valid):
        raise RCDEWithRegistersError("FP16 cache valid-mask hash drift")
    if receipt.get("last_hidden_state_closure_exact") is not True or receipt.get("last_hidden_state_closure_max_abs") != 0.0:
        raise RCDEWithRegistersError("FP16 cache lacks exact last_hidden_state closure")
    logical = receipt.pop("logical_sha256", None)
    if logical != canonical_sha256(receipt):
        raise RCDEWithRegistersError("FP16 cache logical receipt drift")
    restored_norms = torch.linalg.vector_norm(tokens.to(torch.float32), dim=-1)
    expanded_valid = valid.flatten(1)[:, None, :].expand_as(restored_norms)
    if not bool(restored_norms[expanded_valid].gt(MIN_VALID_TOKEN_NORM).all()):
        raise RCDEWithRegistersError("FP16 cache has zero/near-zero valid tokens")


__all__ = [
    "CanonicalGeometry",
    "BLOCK_INDICES_ZERO_BASED",
    "FP16TokenCache",
    "FrozenBackbone",
    "IntermediateTokenBatch",
    "RCDEWithRegistersError",
    "audit_attention_kernel_event_names",
    "audit_attention_kernel_execution",
    "audit_hf_model_semantics",
    "audit_local_checkpoint",
    "build_canonical_geometry",
    "build_fp16_cache",
    "configure_deterministic_eager_inference",
    "default_model_root",
    "extract_intermediate_patch_tokens",
    "exif_orient_without_metadata_rewrite",
    "load_frozen_backbone",
    "preprocess_canonical_full_frame",
    "preprocess_canonical_image_path",
    "runtime_version_receipt",
    "tensor_sha256",
    "validate_fp16_cache",
]
