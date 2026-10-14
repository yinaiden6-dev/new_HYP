"""Split isolation and narrowly scoped compatibility for historical Full-1k."""
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/tools'))
from allocate_splits import apportion, balance_scenes
from evaluation.data import Dataset, file_hash, encoding_inputs


class AllocationTests(unittest.TestCase):
    def test_encoding_identity_excludes_labels_but_includes_pixels_and_order(self):
        rows = [dict(photo_id='a', sha256='a', path='old.jpg', package_id='old'),
                dict(photo_id='b', sha256='b', path='b.jpg', package_id='old')]
        corrected = [dict(rows[0], path='new.jpg', package_id='new'), rows[1]]
        self.assertEqual(encoding_inputs(rows, 'photos'), encoding_inputs(corrected, 'photos'))
        self.assertNotEqual(encoding_inputs(rows, 'photos'), encoding_inputs(rows[::-1], 'photos'))
        corrected[0]['sha256'] = 'different-pixels'
        self.assertNotEqual(encoding_inputs(rows, 'photos'), encoding_inputs(corrected, 'photos'))

    def test_group_rounding(self):
        self.assertEqual(apportion(106, dict(train=70, val=15, test=15)),
                         dict(train=74, val=16, test=16))

    def test_scene_balance_preserves_groups_categories_and_protected_roles(self):
        original = dict(train={'a', 'b'}, val={'c', 'd'}, test={'e', 'f'}, review={'g'})
        categories = {g: 'otc' for ids in original.values() for g in ids}
        features = {g: v for g, v in zip('abcdefg',
                    ([10, 5, 1], [10, 5, 1], [10, 10, 8], [10, 10, 8],
                     [10, 0, 0], [10, 0, 0], [10, 4, 2]))}
        actual, audit = balance_scenes(original, categories, features, {'a'}, 42)
        self.assertIn('a', actual['train'])
        self.assertEqual(actual['review'], {'g'})
        self.assertEqual(Counter(g for ids in actual.values() for g in ids), Counter('abcdefg'))
        for name in original:
            self.assertEqual(Counter(categories[g] for g in original[name]),
                             Counter(categories[g] for g in actual[name]))
        self.assertLess(audit['final_objective'], audit['initial_objective'])
        self.assertEqual((actual, audit), balance_scenes(original, categories, features, {'a'}, 42))
        self.assertEqual(original['train'], {'a', 'b'})


class FullInventoryCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.lists = self.root / 'dataset/lists'
        self.lists.mkdir(parents=True)
        self.archive = self.root / 'backup/groups.csv'
        self.archive.parent.mkdir()
        self.archive.write_text('package_id,group_id,split\np1,g1,test\np2,g2,train\n')
        self.ds = Dataset.__new__(Dataset)
        self.ds.root, self.ds.lists = self.root, self.lists
        self.ds.groups = {'p1': 'g1', 'p2': 'g2'}
        self.ds.hashes = {'photos.csv': 'photos', 'references.csv': 'refs',
                          'positives.csv': 'positives', 'groups.csv': 'new-roles'}
        self.previous = dict(self.ds.hashes, **{'groups.csv': file_hash(self.archive)})
        self.write_protocol()

    def tearDown(self):
        self.temp.cleanup()

    def write_protocol(self):
        (self.lists / 'protocol.json').write_text(json.dumps(dict(previous_protocol=dict(
            directory='../backup', groups_sha256=self.previous['groups.csv']))))

    def test_role_only_change_accepts_verified_history(self):
        self.assertTrue(self.ds.matches_full_inventory(self.previous))
        self.assertTrue(self.ds.matches_full_inventory(self.ds.hashes))

    def test_changed_images_relevance_or_unknown_hash_rejected(self):
        for field in self.previous:
            changed = dict(self.previous)
            changed[field] = 'changed'
            self.assertFalse(self.ds.matches_full_inventory(changed), field)
        self.assertFalse(self.ds.matches_full_inventory(None))

    def test_changed_group_membership_rejected_even_with_valid_archive_hash(self):
        self.archive.write_text('package_id,group_id,split\np1,g2,test\np2,g2,train\n')
        self.previous['groups.csv'] = file_hash(self.archive)
        self.write_protocol()
        self.assertFalse(self.ds.matches_full_inventory(self.previous))

    def test_archive_corruption_or_deletion_rejected(self):
        self.archive.write_text(self.archive.read_text() + '\n')
        self.assertFalse(self.ds.matches_full_inventory(self.previous))
        self.archive.unlink()
        self.assertFalse(self.ds.matches_full_inventory(self.previous))


class ReconstructionInputTests(unittest.TestCase):
    def test_explicit_skip_only_omits_declared_historical_inputs(self):
        from splits import verify_input_hashes
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / 'current.csv'
            current.write_text('current frozen inventory\n')
            protocol = dict(previous_protocol=dict(directory='../old/lists'), source_sha256={
                'current.csv': file_hash(current),
                '../old/lists/protocol.json': 'missing-old-protocol',
                '../old/lists/groups.csv': 'missing-old-groups'})
            self.assertEqual(len(verify_input_hashes(root, protocol)), 2)
            self.assertEqual(verify_input_hashes(root, protocol, reconstruct=False), [])
            current.write_text('tampered\n')
            self.assertEqual(verify_input_hashes(root, protocol, reconstruct=False),
                             ['Protocol input checksum differs: current.csv'])

    def test_skip_does_not_ignore_arbitrary_missing_inputs(self):
        from splits import verify_input_hashes
        with tempfile.TemporaryDirectory() as folder:
            protocol = dict(previous_protocol=dict(directory='../old/lists'),
                            source_sha256={'../other-backup/required.csv': 'required'})
            self.assertEqual(verify_input_hashes(Path(folder), protocol, reconstruct=False),
                             ['Protocol input checksum differs: ../other-backup/required.csv'])


if __name__ == '__main__':
    unittest.main()
