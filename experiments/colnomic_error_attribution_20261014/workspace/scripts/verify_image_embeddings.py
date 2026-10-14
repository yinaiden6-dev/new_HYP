"""Validate image banks against the released CSV inventories and model contracts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/image_embeddings"
sys.path.insert(0,str(ROOT))
from evaluation.dense import DENSE_DIMENSIONS
from evaluation.data import validate_encoding_manifest


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Wait for each model's merge metadata")
    parser.add_argument('--ready',action='store_true',help='Verify complete banks while other models are pending')
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.monotonic()
    report = {"status": "complete", "models": {}, "total_embeddings": 0}
    metadata_path = CACHE / "metadata.json"
    while args.wait and not metadata_path.exists():
        time.sleep(5)
    all_metadata = json.loads(metadata_path.read_text())
    specs=json.loads((ROOT/'configs/baselines.json').read_text())['models']
    models=[name for name in specs if name in all_metadata['models'] and
            set(all_metadata['models'][name].get('files',{}))=={'photos','references'}]
    missing=sorted(set(specs)-set(models))
    assert args.ready or not missing,f'Missing image banks: {missing}'
    report['status']='building' if missing else 'complete'
    report['pending_models']=missing
    for model in models:
        metadata = all_metadata["models"][model]
        populations = {}
        for population in ("references", "photos"):
            path = CACHE / model / f"{population}.pt"
            data = torch.load(path, weights_only=True, map_location="cpu")
            csv_path = ROOT / f"dataset/lists/{population}.csv"
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            expected_count = len(rows)
            id_key = "photo_id" if population == "photos" else "reference_id"
            assert data["schema"] == "release.image_embeddings.v1"
            assert data["view"] == "image" and data["population"] == population
            assert data["model"] == metadata["model"]
            assert data['model']==specs[model]['image_model']
            if 'contract' in data:
                assert data['contract']==specs[model]['image_contract']
            assert len(data["embeddings"]) == len(rows) == expected_count
            assert len(data["provenance"]) == expected_count
            assert data["ids"] == [r[id_key] for r in rows]
            assert len(set(data["ids"])) == expected_count
            validate_encoding_manifest(ROOT, population, CACHE / data['manifest'], data)
            assert data["sha256"] == [r["sha256"] for r in rows]
            assert data["source_sha256"] == [r.get("source_sha256", r["sha256"]) for r in rows]
            file_hash = sha256(path)
            assert file_hash == metadata["files"][population]["sha256"]
            lengths, norms, origins = [], [], {"reused": 0, "generated": 0}
            for embedding, source in zip(data["embeddings"], data["provenance"], strict=True):
                assert torch.isfinite(embedding).all()
                if model in DENSE_DIMENSIONS:
                    assert embedding.shape == (DENSE_DIMENSIONS[model],) and embedding.dtype == torch.float32
                    norm = embedding.norm()
                    assert abs(float(norm) - 1) < 0.01
                    lengths.append(1)
                else:
                    assert embedding.ndim == 2 and embedding.shape[1] == 128
                    assert embedding.dtype == torch.float16
                    assert embedding.shape[0] > 0
                    norm = embedding.float().norm(dim=-1)
                    nonzero = norm[norm > 0]
                    assert len(nonzero) > 0
                    assert ((nonzero - 1).abs() < 0.03).all()
                    norm = nonzero.mean()
                    lengths.append(embedding.shape[0])
                norms.append(float(norm))
                origins[source["kind"]] += 1
                if "contract" in source:
                    assert source["contract"]["view"] == "image"
                    assert source["contract"]["text"] == ""
            if population == "photos":
                assert origins["reused"] == metadata["reused_photos"]
                assert origins["generated"] == metadata["generated_photos"]
            else:
                assert origins['reused']==metadata['reused_references']
                assert origins['generated']==metadata['generated_references']
            populations[population] = dict(count=expected_count, origins=origins,
                  shape=f'N x {DENSE_DIMENSIONS[model]}' if model in DENSE_DIMENSIONS else "list of tokens x 128",
                  token_count_min=min(lengths), token_count_max=max(lengths),
                  mean_nonzero_token_norm_range=[min(norms), max(norms)],
                  bytes=path.stat().st_size, sha256=file_hash,
                  order_matches_csv=True, finite=True, normalized=True)
            report["total_embeddings"] += expected_count
            print(model, population, expected_count, "verified", flush=True)
        report["models"][model] = populations
    if not missing:
        assert report["total_embeddings"] == all_metadata["total_embeddings"]
    report["seconds"] = time.monotonic() - start
    (CACHE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VERIFIED:", report["total_embeddings"], "embeddings; pending:",missing,flush=True)


if __name__ == "__main__":
    main()
