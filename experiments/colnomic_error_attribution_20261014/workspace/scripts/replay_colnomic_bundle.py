"""Verify original files and complete token similarity matrices after extraction."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.supplement_colnomic_archives import BANK_PATHS, file_hash, dense_similarity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Extracted F03 or B03 directory')
    parser.add_argument('--verify-files', action='store_true')
    parser.add_argument('--verify-matrices', action='store_true')
    parser.add_argument('--pair', nargs=2, metavar=('PHOTO_ID','REFERENCE_ID'))
    args = parser.parse_args()
    folder = args.bundle.resolve()
    reproduction = folder/'reproduction'
    torch.set_num_threads(2)
    if args.verify_files:
        entries = (reproduction/'FILES.sha256').read_text().splitlines()
        for i, line in enumerate(entries, 1):
            sha, relative = line.split('  ', 1)
            if file_hash(folder/relative) != sha:
                raise ValueError('SHA-256 mismatch: '+relative)
            if i % 100 == 0 or i == len(entries):
                print(f'File hashes checked {i}/{len(entries)}', flush=True)
    pairs = json.loads((reproduction/'pair_index.json').read_text())
    if args.pair:
        pairs = [row for row in pairs if (row['photo_id'],row['reference_id']) == tuple(args.pair)]
        if len(pairs) != 1:
            parser.error('Pair not in bundled candidate set')
    if not args.verify_matrices and not args.pair:
        print(json.dumps(json.loads((reproduction/'validation.json').read_text()),ensure_ascii=False,indent=2))
        return
    banks = {pop:torch.load(reproduction/path,map_location='cpu',weights_only=True,mmap=True)
             for pop,path in BANK_PATHS.items()}
    maximum_error, checked_csvs = 0., set()
    for i, pair in enumerate(pairs, 1):
        q = banks['photos']['embeddings'][pair['query_bank_index']]
        r = banks['references']['embeddings'][pair['reference_bank_index']]
        assert banks['photos']['ids'][pair['query_bank_index']] == pair['photo_id']
        assert banks['references']['ids'][pair['reference_bank_index']] == pair['reference_id']
        matrix = np.load(folder/pair['matrix_file'], mmap_mode='r', allow_pickle=False)
        expected, qi, matches, maxima = dense_similarity(q, r)
        assert matrix.dtype == np.float32 and list(matrix.shape) == pair['shape']
        assert file_hash(folder/pair['matrix_file']) == pair['sha256']
        np.testing.assert_allclose(matrix, expected, rtol=0, atol=2e-6)
        score_error = abs(float(maxima.mean())-pair['cached_image_score'])
        assert score_error < 2e-6
        maximum_error = max(maximum_error, score_error)
        for relation in pair['relations']:
            if relation['source'] != 'original_contrast':
                continue
            role, csvpath = relation['candidate_role'], relation['token_csv']
            if (csvpath,role) in checked_csvs:
                continue
            checked_csvs.add((csvpath,role))
            with (folder/csvpath).open() as stream:
                rows = list(csv.DictReader(stream))
            prefix = 'pos' if role=='positive' else 'neg'
            assert [int(row['q_token_index']) for row in rows] == qi.tolist()
            assert [int(row[prefix+'_token_index']) for row in rows] == matches.tolist()
            np.testing.assert_allclose([float(row['c_'+role]) for row in rows], maxima, rtol=0, atol=2e-6)
        if args.pair:
            print(json.dumps(pair,ensure_ascii=False,indent=2))
        if i % 100 == 0 or i == len(pairs):
            print(f'Matrices replayed {i}/{len(pairs)}; max score error {maximum_error:.3g}',flush=True)
    print(f'PASS: {len(pairs)} full matrices, {len(checked_csvs)} original CSV candidate sides',flush=True)


if __name__ == '__main__':
    main()
