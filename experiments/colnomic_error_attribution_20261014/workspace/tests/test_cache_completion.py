"""Regression cases for path-only OCR reuse and safe score-row reuse."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from evaluation import ocr
from evaluation.cache import Cache, save_tensor
from evaluation.complete_scores import reusable_rows
from evaluation.data import digest, file_hash, read_json, write_json


class CompletionTests(unittest.TestCase):
    def test_text_runtime_keeps_each_banks_original_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'cache/ocr_embeddings'
            write_json(directory / 'manifests/photos.csv', ['photo inputs'])
            write_json(directory / 'manifests/references.csv', ['original reference inputs'])
            write_json(root / 'dataset/lists/references.csv', ['updated labels'])
            photo = dict(model={'pinned': 'unit'}, family='unit', ids=['p0'], has_text=[True],
                         input_token_counts=[1], contract={}, manifest='manifests/photos.csv',
                         manifest_sha256=file_hash(directory / 'manifests/photos.csv'))
            reference = dict(photo, ids=['r0'], manifest='manifests/references.csv',
                             manifest_sha256=file_hash(directory / 'manifests/references.csv'))
            save_tensor(directory / 'colpali/photos.pt', photo)
            save_tensor(directory / 'colpali/references.pt', reference)
            dataset = SimpleNamespace(root=root, photos=[{}], references=[{}], records=lambda population: [{}])
            cache = Cache(dataset, {'models': {'colpali': {'text_model': {'pinned': 'unit'}}}})
            cache.update_metadata('colpali', 'text', 'photos', photo, directory / 'colpali/photos.pt')
            inputs = read_json(directory / 'metadata.json')['models']['colpali']['runtime']['input_files']
            self.assertEqual(inputs['cache/ocr_embeddings/manifests/references.csv'], reference['manifest_sha256'])
            self.assertNotIn('dataset/lists/references.csv', inputs)

    def test_moved_image_reuses_ocr_without_loading_model(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'dataset/moved/image.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'unchanged pixels')
            manifest = root / 'dataset/lists/photos.csv'
            manifest.parent.mkdir(parents=True)
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['photo_id', 'path', 'sha256'])
                writer.writeheader()
                writer.writerow(dict(photo_id='p0', path='moved/image.jpg', sha256=file_hash(image)))
            write_json(root / 'configs/ppocrv6.json', dict(model='unit'))
            write_json(root / 'configs/ppocrv6_models.json', {})
            old = dict(image_path='dataset/old/image.jpg', image_sha256=file_hash(image),
                       text='existing text', normalized_text='existing text', lines=[])
            write_json(root / 'cache/ocr/photos.json', dict(p0=old))
            with patch.object(ocr, 'OCR', side_effect=AssertionError('Unexpected model loading')):
                ocr.ensure(root, 'photos', 'cpu')
            saved = read_json(root / 'cache/ocr/photos.json')['p0']
            self.assertEqual(saved, dict(old, image_path='dataset/moved/image.jpg'))

    def fixtures(self):
        old = dict(ids=['p0', 'p1'], model='unit', view='image', contract={'unit': True},
                   sha256=['h0', 'h1'], source_sha256=['h0', 'h1'], embeddings=torch.eye(2))
        query = dict(old, ids=['p1', 'new', 'p0'], sha256=['h1', 'new', 'h0'],
                     source_sha256=['h1', 'new', 'h0'], embeddings=torch.tensor([[0., 1.], [1., 1.], [1., 0.]]))
        recipe = dict(query='new_hash', reference='gallery_hash', scorer_sha256='scorer')
        previous_recipe = dict(recipe, query='old_hash')
        previous = dict(recipe=previous_recipe, signature=digest(previous_recipe),
                        query_ids=old['ids'], reference_ids=['r0'], scores=torch.tensor([[1.], [2.]]))
        return old, query, previous, recipe

    def test_insertions_and_reordering_reuse_matching_rows(self):
        old, query, previous, recipe = self.fixtures()
        self.assertEqual(reusable_rows(old, query, previous, recipe, ['r0'], 'old_hash'), [(0, 1), (2, 0)])

    def test_changed_vector_or_image_is_recomputed(self):
        old, query, previous, recipe = self.fixtures()
        query['embeddings'][0] = torch.tensor([1., 0.])
        self.assertEqual(reusable_rows(old, query, previous, recipe, ['r0'], 'old_hash'), [(2, 0)])
        query['sha256'][2] = 'changed'
        self.assertEqual(reusable_rows(old, query, previous, recipe, ['r0'], 'old_hash'), [])

    def test_changed_scorer_gallery_or_provenance_disables_reuse(self):
        old, query, previous, recipe = self.fixtures()
        for changed in [dict(recipe, reference='different'), dict(recipe, scorer_sha256='different')]:
            self.assertEqual(reusable_rows(old, query, previous, changed, ['r0'], 'old_hash'), [])
        self.assertEqual(reusable_rows(old, query, previous, recipe, ['r0'], 'wrong_hash'), [])
        previous['signature'] = 'wrong'
        self.assertEqual(reusable_rows(old, query, previous, recipe, ['r0'], 'old_hash'), [])


if __name__ == '__main__':
    unittest.main()
