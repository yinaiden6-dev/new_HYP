"""Exact frozen ColNomic projection for post-LLM M adapters.

Only the 3584->128 retrieval projector and its saved LoRA tensors are loaded.
The base and LoRA GEMMs stay unmerged and BF16, matching installed PEFT's eval
forward. Frozen cached language hidden states are reused; no encoder is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            d.update(block)
    return d.hexdigest()


def _tensor_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class FrozenProjection(nn.Module):
    """Vanilla PEFT LoRA eval equation, retaining BF16 operation ordering."""
    def __init__(self, weight, bias, lora_a, lora_b, scaling):
        super().__init__()
        self.register_buffer('weight', weight.to(dtype=torch.bfloat16))
        self.register_buffer('bias', bias.to(dtype=torch.bfloat16))
        self.register_buffer('lora_a', lora_a.to(dtype=torch.bfloat16))
        self.register_buffer('lora_b', lora_b.to(dtype=torch.bfloat16))
        self.scaling = float(scaling)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.report = {}

    def forward(self, hidden):
        # Mirrors peft.tuners.lora.layer.Linear.forward in eval, unmerged mode:
        # base(x) + B(A(dropout(x))) * alpha/r; eval dropout is identity.
        if hidden.dtype != torch.bfloat16:
            raise ValueError('qualified frozen hidden states must stay BF16')
        result = F.linear(hidden, self.weight, self.bias)
        update = F.linear(F.linear(hidden.to(self.lora_a.dtype), self.lora_a), self.lora_b)
        return (result + update * self.scaling).to(result.dtype)


def _verified_sources(a: dict, used: list[Path]) -> list[dict]:
    """Reuse sealed full-file hashes only when exact inode/size/mtime match.

    GPFS mount device numbers differ across login/compute nodes; they are
    recorded but cannot be used as cross-node file identity.
    """
    binding = a.get('model_source_validation')
    if not binding:
        raise ValueError('authority must bind the qualified V2 model source receipt')
    receipt_path = Path(binding['path'])
    if _sha(receipt_path) != binding['sha256']:
        raise ValueError('model source receipt SHA mismatch')
    receipt = json.loads(receipt_path.read_text())
    if receipt['status'] != 'MODEL_SOURCE_HASHES_PASS':
        raise ValueError('unqualified model source receipt')
    records = {str(Path(x['binding']['path']).resolve()): x for x in receipt['files']}
    sources = []
    for path in used:
        path = path.resolve()
        s = path.stat()
        stat = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]
        old = records.get(str(path))
        if old is not None:
            if stat[1:] != old['stat'][1:]:
                raise ValueError(f'qualified model source changed: {path}')
            if s.st_size <= 2 << 20 and _sha(path) != old['binding']['sha256']:
                raise ValueError(f'qualified configuration SHA changed: {path}')
            sources.append({'path': str(path), 'sha256': old['binding']['sha256'],
                            'stat': stat, 'sealed_stat': old['stat'],
                            'verification': 'sealed hash plus exact inode/size/mtime; small configuration rehashed'})
        else:
            # Small index/config inputs absent from a receipt can be hashed now.
            if s.st_size > 2 << 20:
                raise ValueError(f'large weight source absent from qualified receipt: {path}')
            sources.append({'path': str(path), 'sha256': _sha(path), 'stat': stat,
                            'verification': 'current full SHA256'})
    return sources


def load_projection(a: dict, device='cuda') -> FrozenProjection:
    """Return a frozen projector; provenance is stored in ``model.report``.

    ``a`` must carry the same ``model`` and ``model_source_validation`` fields
    as V4. This implementation fails closed for unsupported PEFT variants or
    unexpected saved projection tensors instead of dropping their parameters.
    """
    from safetensors import safe_open
    adapter_root = Path(a['model']).resolve()
    cfg_path = adapter_root/'adapter_config.json'
    config = json.loads(cfg_path.read_text())
    forbidden = ['use_dora', 'use_rslora', 'fan_in_fan_out', 'lora_bias']
    if any(config.get(k, False) for k in forbidden):
        raise ValueError('only the qualified vanilla BF16 projection LoRA is supported')
    if config.get('modules_to_save') or config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('unexpected modules_to_save or per-module LoRA overrides')
    if config.get('bias', 'none') != 'none' or config['peft_type'] != 'LORA':
        raise ValueError('unexpected PEFT projector configuration')
    base_root = Path(config['base_model_name_or_path']).resolve()
    index_path = base_root/'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    base_keys = ['custom_text_proj.weight', 'custom_text_proj.bias']
    if any(k not in index for k in base_keys):
        raise ValueError('qualified base retrieval projection missing')
    weight_file = adapter_root/'adapter_model.safetensors'
    shards = sorted({base_root/index[k] for k in base_keys})
    sources = _verified_sources(a, [cfg_path, index_path, weight_file] + shards)
    tensors = {}
    for shard in shards:
        with safe_open(str(shard), framework='pt', device='cpu') as f:
            for key in base_keys:
                if base_root/index[key] == shard:
                    tensors[key] = f.get_tensor(key)
    adapter_keys = ['base_model.model.custom_text_proj.lora_A.weight',
                    'base_model.model.custom_text_proj.lora_B.weight']
    with safe_open(str(weight_file), framework='pt', device='cpu') as f:
        actual = sorted(k for k in f.keys() if 'custom_text_proj' in k)
        if actual != adapter_keys:
            raise ValueError(f'unexpected saved projection state: {actual}')
        for key in adapter_keys:
            tensors[key] = f.get_tensor(key)
    w, b = (tensors[k] for k in base_keys)
    la, lb = (tensors[k] for k in adapter_keys)
    rank = int(config['r'])
    if (tuple(w.shape), tuple(b.shape), tuple(la.shape), tuple(lb.shape)) != (
            (128, 3584), (128,), (rank, 3584), (128, rank)):
        raise ValueError('projector dimensions differ from qualified ColNomic')
    model = FrozenProjection(w, b, la, lb, float(config['lora_alpha'])/rank)
    model = model.to(device).eval().requires_grad_(False)
    model.report = {
        'status': 'FROZEN_COLNOMIC_PROJECTION_ONLY_LOADED', 'sources': sources,
        'projection_tensor_sources': {k: {'shape': list(t.shape), 'source_dtype': str(t.dtype),
            'source_tensor_sha256': _tensor_sha(t)} for k, t in tensors.items()},
        'frozen_parameters': sum(t.numel() for t in model.buffers()),
        'trainable_parameters': 0, 'base_dtype': 'torch.bfloat16', 'lora_dtype': 'torch.bfloat16',
        'projection_merged': False, 'lora_scaling': model.scaling,
        'model_forward_calls': 0, 'language_model_loaded': False,
        'semantics': 'base linear + unmerged BF16 LoRA B(A(x))*alpha/r; eval dropout identity',
    }
    return model


def predict_full_projection(model, cache, small, mass):
    """Project/normalize full sequence; caller selects image positions."""
    hidden = cache['hidden']
    image_mask = cache['image_mask']
    batch = cache['batch']
    if hidden.requires_grad or hidden.ndim != 3 or hidden.shape[-1] != 3584:
        raise ValueError('expected detached [batch, sequence,3584] cached hidden state')
    if hidden.shape[:2] != image_mask.shape or image_mask.dtype != torch.bool:
        raise ValueError('hidden/image-mask mismatch')
    if batch['attention_mask'].shape != image_mask.shape:
        raise ValueError('attention-mask mismatch')
    if hidden.device != model.weight.device or image_mask.device != hidden.device:
        raise ValueError('cache/projector device mismatch')
    if bool((image_mask & ~batch['attention_mask'].bool()).any()) or not bool(image_mask.any()):
        raise ValueError('invalid image token mask')
    if small is not None:
        adapted = hidden.clone()
        adapted[image_mask] = small(hidden[image_mask], mass)
    else:
        adapted = hidden
    # Preserve the original full-sequence GEMM shape and BF16 norm semantics.
    encoded = model(adapted)
    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    encoded = encoded * batch['attention_mask'].unsqueeze(-1)
    return encoded


def predict_projection(model, cache, small, mass):
    """Return image-only [patch,128], same as OLD.predict_tokens(postllm)."""
    return predict_full_projection(model, cache, small, mass)[cache['image_mask']]


@torch.no_grad()
def validate_cached_projection(model, cache, *, atol=0.0):
    """Engineering native-token parity, no training and no encoder forward.

    Exact native-cache parity is the GPU qualification gate. CPU BF16 kernels
    can round differently from the GPU cache; callers may diagnose CPU output
    but must not loosen the GPU gate based on such diagnostics.
    """
    actual = predict_projection(model, cache, None, 0.0)
    expected = cache['native_tokens'][cache['image_mask']]
    error = float((actual.float()-expected.float()).abs().max())
    return {'status': 'POSTLLM_NATIVE_TOKEN_PARITY_PASS' if error <= atol else 'POSTLLM_NATIVE_TOKEN_PARITY_FAIL',
            'query_id': cache.get('query_id'), 'patches': int(actual.shape[0]),
            'max_abs_error': error, 'atol': float(atol), 'exact_equal': bool(torch.equal(actual, expected)),
            'device': str(actual.device), 'dtype': str(actual.dtype), 'new_encoder_forwards': 0}


def validate_against_installed_peft(model, cache):
    """Small independent installed-PEFT equation/input-gradient conformance.

    Uses the same four frozen tensors and no backbone. Does not update either
    model. This is separate from GPU parity with previously cached outputs.
    """
    import inspect
    from peft.tuners.lora.layer import Linear as PeftLinear
    base = nn.Linear(model.in_features, model.out_features, bias=True,
                     device=model.weight.device, dtype=torch.bfloat16)
    with torch.no_grad():
        base.weight.copy_(model.weight)
        base.bias.copy_(model.bias)
    rank = model.lora_a.shape[0]
    wrapped = PeftLinear(base, adapter_name='default', r=rank,
                         lora_alpha=model.scaling*rank, lora_dropout=0.1,
                         init_lora_weights=False).to(dtype=torch.bfloat16)
    with torch.no_grad():
        wrapped.lora_A['default'].weight.copy_(model.lora_a)
        wrapped.lora_B['default'].weight.copy_(model.lora_b)
    wrapped.eval().requires_grad_(False)
    actual_input = cache['hidden'].detach().clone().requires_grad_(True)
    reference_input = cache['hidden'].detach().clone().requires_grad_(True)
    actual = model(actual_input)
    reference = wrapped(reference_input)
    # Fixed coordinate objective; checks gradients flowing to a later adapter.
    actual[..., 0].float().sum().backward()
    reference[..., 0].float().sum().backward()
    error = float((actual.float()-reference.float()).abs().max())
    grad_error = float((actual_input.grad.float()-reference_input.grad.float()).abs().max())
    source = Path(inspect.getfile(PeftLinear))
    return {'status': 'INSTALLED_PEFT_PROJECTOR_CONFORMANCE_PASS' if error == 0 and grad_error == 0
            else 'INSTALLED_PEFT_PROJECTOR_CONFORMANCE_FAIL',
            'projection_max_abs_error': error, 'hidden_gradient_max_abs_error': grad_error,
            'source': {'path': str(source), 'sha256': _sha(source)},
            'dtype': str(actual.dtype), 'device': str(actual.device), 'new_encoder_forwards': 0}
