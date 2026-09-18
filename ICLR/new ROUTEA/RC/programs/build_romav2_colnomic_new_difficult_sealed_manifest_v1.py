#!/usr/bin/env python3
"""Bind the frozen new_difficult endpoint without decoding image pixels."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data/route_a_v2_new_difficult_v1"
IDENTITY = DATA / "identity_manifest_v1.csv"
PHOTO = DATA / "photo_manifest_v1.csv"
SPLIT = DATA / "frozen_split_v1.json"
LINEAGE = DATA / "freeze_lineage_v1.json"
CONTRACT = ROOT / "plan/ROMAV2_COLNOMIC_FULL_NEGATIVE_EXTERNAL_CONFIRMATION_CONTRACT_V1_20260831.md"
AUTHORITY = ROOT / "registry/romav2_colnomic_full_negative_external_confirmation_authority_v1_20260831.json"
PREFLIGHT = ROOT / "results/romav2_colnomic_external_confirmation_contract_v1/preflight.json"
OUT = ROOT / "results/romav2_colnomic_new_difficult_sealed_manifest_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def logical(value: dict) -> str:
    projection = dict(value)
    projection.pop("logical_sha256", None)
    payload = json.dumps(
        projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: dict) -> None:
    value["logical_sha256"] = logical(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"immutable output already exists: {OUT}")

    authority = json.loads(AUTHORITY.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    lineage = json.loads(LINEAGE.read_text())
    split = json.loads(SPLIT.read_text())

    assert authority["status"] == "EXTERNAL_CONFIRMATION_CONTRACT_FROZEN_INPUT_BINDING_REQUIRED"
    assert authority["permissions"]["sealed_manifest_metadata_binding_authorized"] is True
    assert preflight["status"] == "SEALED_INPUT_MANIFEST_BINDING_PREFLIGHT_READY"
    assert sha(CONTRACT) == authority["contract_sha256"]
    assert sha(IDENTITY) == "1f447d67b0164250de750744c6da73b90874ef0caf4b34fda12766fdfcd16c75"
    assert sha(PHOTO) == lineage["frozen_split_sha256"] or sha(PHOTO) == split["source"]["photo_manifest_sha256"]
    assert sha(PHOTO) == "98fc888f4e481a075e7827b23e956fffd71231f76a5cead232c44b2f7aaf9e34"
    assert sha(SPLIT) == lineage["frozen_split_sha256"]
    assert split["status"] == "FROZEN_BEFORE_RETRIEVAL_EVALUATION"

    identities = {r["identity_id"]: r for r in csv.DictReader(IDENTITY.open(newline=""))}
    photos = list(csv.DictReader(PHOTO.open(newline="")))
    selected = [
        r
        for r in photos
        if r["split"] == "test"
        and r["formal_primary"] == "True"
        and r["endpoint_status"] == "test_primary_sealed"
        and r["use_class"] == "primary_clean_core"
    ]
    selected.sort(key=lambda r: (r["identity_id"].encode(), int(r["pose_index"]), r["photo_filename"].encode()))
    assert len(photos) == 176
    assert len(selected) == 31
    assert len({r["identity_id"] for r in selected}) == 8
    assert {identities[r["identity_id"]]["split"] for r in selected} == {"test"}

    prejoin_rows = []
    join_rows = []
    for ordinal, row in enumerate(selected):
        path = Path(row["photo_path"])
        assert path.is_file()
        assert path.stat().st_size == int(row["bytes"])
        observed_sha = sha(path)
        assert observed_sha == row["sha256"]
        query_id = "NDS-" + hashlib.sha256(
            ("romav2-colnomic-ext-v1|" + observed_sha).encode()
        ).hexdigest()[:16]
        prejoin_rows.append(
            {
                "query_ordinal": ordinal,
                "query_id": query_id,
                "query_path": str(path),
                "query_sha256": observed_sha,
                "bytes": int(row["bytes"]),
                "raw_width": int(row["raw_width"]),
                "raw_height": int(row["raw_height"]),
                "exif_orientation": int(row["exif_orientation"]),
                "oriented_width": int(row["oriented_width"]),
                "oriented_height": int(row["oriented_height"]),
            }
        )
        ident = identities[row["identity_id"]]
        assert int(row["gallery_index"]) == int(ident["gallery_index"])
        assert row["reference_label"] == ident["reference_label"]
        join_rows.append(
            {
                "query_ordinal": ordinal,
                "query_id": query_id,
                "identity_id": row["identity_id"],
                "supergroup_id": ident["supergroup_id"],
                "target_exact_label": row["reference_label"],
                "target_gallery_physical_row": int(row["gallery_index"]),
                "target_reference_sha256": ident["reference_sha256"],
            }
        )

    OUT.mkdir(parents=True, exist_ok=False)
    prejoin = {
        "schema_version": "rc_romav2_colnomic_new_difficult_prejoin_manifest_v1_20260831",
        "status": "NEW_DIFFICULT_SEALED_PREJOIN_MANIFEST_BOUND",
        "query_count": 31,
        "identity_count_hidden_from_scoring_process": 8,
        "target_field_count": 0,
        "target_join_path_exposed_to_scoring_process": False,
        "rows": prejoin_rows,
        "logical_sha256": "",
    }
    write_json(OUT / "prejoin_manifest.json", prejoin)

    target_join = {
        "schema_version": "rc_romav2_colnomic_new_difficult_target_join_v1_20260831",
        "status": "NEW_DIFFICULT_SEALED_TARGET_JOIN_BOUND_REDUCER_ONLY",
        "query_count": 31,
        "identity_count": 8,
        "scoring_process_read_authorized": False,
        "rows": join_rows,
        "logical_sha256": "",
    }
    write_json(OUT / "target_join_manifest.json", target_join)

    receipt = {
        "schema_version": "rc_romav2_colnomic_new_difficult_manifest_binding_receipt_v1_20260831",
        "status": "NEW_DIFFICULT_SEALED_MANIFEST_BINDING_PASS",
        "claim_level": "SEALED_METADATA_AND_HASH_BINDING_ONLY_NO_PIXEL_DECODE_NO_SCORING",
        "sources": {
            "identity_manifest_sha256": sha(IDENTITY),
            "photo_manifest_sha256": sha(PHOTO),
            "frozen_split_sha256": sha(SPLIT),
            "freeze_lineage_sha256": sha(LINEAGE),
            "contract_sha256": sha(CONTRACT),
            "authority_sha256": sha(AUTHORITY),
            "preflight_sha256": sha(PREFLIGHT),
        },
        "outputs": {
            "prejoin_manifest_sha256": sha(OUT / "prejoin_manifest.json"),
            "target_join_manifest_sha256": sha(OUT / "target_join_manifest.json"),
        },
        "inventory": {
            "frozen_file_count": len(photos),
            "test_file_count": sum(r["split"] == "test" for r in photos),
            "test_primary_query_count": len(selected),
            "test_identity_count": len({r["identity_id"] for r in selected}),
            "test_auxiliary_excluded_count": sum(
                r["split"] == "test" and r["formal_primary"] == "False" for r in photos
            ),
        },
        "access": {
            "metadata_row_read_count": len(photos),
            "sealed_image_hash_read_count": len(selected),
            "sealed_pixel_decode_count": 0,
            "sealed_model_scoring_count": 0,
            "sealed_result_join_count": 0,
        },
        "next_authorized_stage": "SEALED_SCORING_SOURCE_CLOSURE_AND_E0",
        "logical_sha256": "",
    }
    write_json(OUT / "binding_receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "inventory": receipt["inventory"],
                "access": receipt["access"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
