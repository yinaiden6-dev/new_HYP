"""Tensor-aware canonical hashes for speculative Track-R OOF artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
import torch


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()
    return {
        "__tensor_sha256__": digest,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }


def normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_record(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray_sha256__": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
    if isinstance(value, Mapping):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported OOF canonical type: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def logical_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: item for key, item in value.items() if key != "logical_sha256"}
    )


__all__ = ["canonical_sha256", "logical_sha256", "normalize"]
