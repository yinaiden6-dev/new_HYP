"""Hash-bound frozen ColNomic C128 score source for RGH V9 losses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Sequence

import torch


LEDGER_RELATIVE_PATH = Path(
    "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_ledger.pt"
)
RECEIPT_RELATIVE_PATH = Path(
    "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_receipt.json"
)
VALIDATION_RELATIVE_PATH = Path(
    "results/dino_rcde_sr0_mt_i0_v1/formal_job5075941/prejoin_validation.json"
)
SOURCE_RELATIVE_PATH = Path(
    "results/cw0_rgh_xf_v2_p0_a0_manifest_v2/source_manifest.json"
)
LEDGER_SHA256 = "3540980f34bd67f62509595c9aef86da2059174be638d909660c891a12aa109a"
RECEIPT_SHA256 = "c93a8f0d157cf58670f35f8ff580a5308cdef2f012bd5f932a101d0530eb5a86"
VALIDATION_SHA256 = "8f3e212769c30e7966ccf80925058ec7f7714323c47877f2805718358bf49c50"
SOURCE_SHA256 = "e0be35125eddec391a92398e84103e77ca0a65f661452b9190c2b81f3fdbea23"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_binary64(value: str) -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise ValueError("invalid binary64 score bits")
    result = struct.unpack(">d", bytes.fromhex(value))[0]
    if not torch.isfinite(torch.tensor(result, dtype=torch.float64)):
        raise ValueError("non-finite frozen ColNomic score")
    return result


class FrozenColNomicBaseV1:
    def __init__(self, rc_root: Path):
        self.root = Path(rc_root).resolve()
        ledger_path = self.root / LEDGER_RELATIVE_PATH
        receipt_path = self.root / RECEIPT_RELATIVE_PATH
        validation_path = self.root / VALIDATION_RELATIVE_PATH
        source_path = self.root / SOURCE_RELATIVE_PATH
        for path, expected in (
            (ledger_path, LEDGER_SHA256),
            (receipt_path, RECEIPT_SHA256),
            (validation_path, VALIDATION_SHA256),
            (source_path, SOURCE_SHA256),
        ):
            if not path.is_file() or path.is_symlink() or _sha(path) != expected:
                raise RuntimeError(f"frozen ColNomic binding drift: {path}")
        receipt = json.loads(receipt_path.read_text())
        validation = json.loads(validation_path.read_text())
        if not (
            receipt["status"] == "RCDE_SR0_MT_I0_PREJOIN_MATERIALIZATION_PASS"
            and validation["status"] == "RCDE_SR0_MT_I0_PREJOIN_INDEPENDENT_VALIDATION_PASS"
            and receipt["prejoin_ledger_sha256"] == LEDGER_SHA256
            and validation["prejoin_ledger_sha256"] == LEDGER_SHA256
        ):
            raise RuntimeError("frozen ColNomic receipt/validation drift")
        ledger = torch.load(ledger_path, map_location="cpu", weights_only=False)
        source = json.loads(source_path.read_text())
        if not (
            ledger["status"] == "RCDE_SR0_MT_I0_PREJOIN_LEDGER_SEALED"
            and ledger["query_count"] == source["query_count"] == 600
            and ledger["candidate_count_per_query"] == source["candidate_count_per_query"] == 128
            and ledger["target_or_label_join_count"] == 0
            and ledger["label_read_count"] == 0
        ):
            raise RuntimeError("frozen ColNomic ledger envelope drift")
        old = {int(item["execution_ordinal"]): item for item in ledger["records"]}
        new = {int(item["execution_ordinal"]): item for item in source["records"]}
        if set(old) != set(new) or len(old) != 600:
            raise RuntimeError("frozen ColNomic execution population drift")
        self._records = {}
        for execution in sorted(new):
            left, right = old[execution], new[execution]
            if not (
                left["query_id"] == right["query_id"]
                and left["source_image_sha256"] == right["source_image_sha256"]
                and left["candidate_physical_rows"] == right["candidate_physical_rows"]
                and len(left["candidate_raw_score_bits"]) == 128
            ):
                raise RuntimeError(f"frozen ColNomic C128 axis drift at {execution}")
            self._records[execution] = left

    def scores(
        self, execution_ordinal: int, candidate_physical_rows: Sequence[int]
    ) -> torch.Tensor:
        record = self._records.get(int(execution_ordinal))
        if record is None or list(candidate_physical_rows) != record["candidate_physical_rows"]:
            raise RuntimeError("requested ColNomic candidate axis drift")
        return torch.tensor(
            [_decode_binary64(value) for value in record["candidate_raw_score_bits"]],
            dtype=torch.float64,
        )


__all__ = ["FrozenColNomicBaseV1"]

