"""Independent tensor-aware canonical hashes for Track-R OOF validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def _array_record(array: np.ndarray, *, marker: str) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        marker: hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
    }


def normalize_independent(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        return {
            "__tensor_sha256__": hashlib.sha256(
                tensor.numpy().tobytes(order="C")
            ).hexdigest(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
    if isinstance(value, np.ndarray):
        return _array_record(value, marker="__ndarray_sha256__")
    if isinstance(value, Mapping):
        return {
            str(key): normalize_independent(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [normalize_independent(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"unsupported independent OOF canonical type: {type(value).__name__}"
    )


def canonical_sha256_independent(value: Any) -> str:
    payload = json.dumps(
        normalize_independent(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def logical_sha256_independent(value: Mapping[str, Any]) -> str:
    return canonical_sha256_independent(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


__all__ = [
    "canonical_sha256_independent",
    "logical_sha256_independent",
    "normalize_independent",
]
