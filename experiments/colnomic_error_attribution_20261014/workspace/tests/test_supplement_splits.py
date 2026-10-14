"""Source-group integrity when extending a frozen photograph collection."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/tools'))
from supplement_splits import select_groups, source_groups


def package(pid, document=None, set_id=None, product=None):
    docs = []
    if document:
        docs.append(dict(document_id=document, set_id=set_id or document,
            products=[dict(code_system='ndc', code=product)] if product else []))
    return dict(package_id=pid, identity=pid, design=dict(sha256=pid), documents=docs)


class SupplementGroupingTests(unittest.TestCase):
    def test_document_and_product_links_are_transitive(self):
        packages = [package('p1', 'doc1'), package('p2', 'doc1', product='123'),
                    package('p3', 'doc2', product='123'), package('p4', 'doc4')]
        groups, roles = source_groups(packages, [], [])
        self.assertEqual(groups['p1'], groups['p2'])
        self.assertEqual(groups['p2'], groups['p3'])
        self.assertNotEqual(groups['p1'], groups['p4'])
        self.assertEqual(roles, {})
        self.assertEqual((groups, roles), source_groups(packages[::-1], [], []))

    def test_new_revision_inherits_existing_set_id_group_and_role(self):
        packages = [package('old', 'v1', 'set1'), package('new', 'v2', 'set1')]
        previous = [dict(package_id='old', group_id='frozen', split='val')]
        groups, roles = source_groups(packages, [], previous)
        self.assertEqual(groups, {'new': 'frozen', 'old': 'frozen'})
        self.assertEqual(roles, {'frozen': 'val'})

    def test_frozen_group_members_remain_joined_without_direct_links(self):
        previous = [dict(package_id=p, group_id='frozen', split='train') for p in ('a', 'b')]
        groups, roles = source_groups([package('a'), package('b')], [], previous)
        self.assertEqual(groups, {'a': 'frozen', 'b': 'frozen'})
        self.assertEqual(roles, {'frozen': 'train'})

    def test_photo_source_hash_links_distinct_release_hashes(self):
        photos = [dict(package_id=p, sha256=p, source_sha256='same-source') for p in ('a', 'b')]
        groups, _ = source_groups([package('a'), package('b')], photos, [])
        self.assertEqual(groups['a'], groups['b'])

    def test_bridge_between_frozen_groups_is_rejected(self):
        packages = [package('a', 'doc1'), package('b', 'doc2')]
        bridge = package('new', 'doc1')
        bridge['documents'] += package('x', 'doc2')['documents']
        previous = [dict(package_id='a', group_id='g1', split='train'),
                    dict(package_id='b', group_id='g2', split='test')]
        with self.assertRaisesRegex(ValueError, 'distinct frozen groups'):
            source_groups(packages + [bridge], [], previous)

    def test_equal_design_bytes_override_different_names(self):
        a, b = package('a'), package('b')
        b['design']['sha256'] = a['design']['sha256']
        groups, _ = source_groups([a, b], [], [])
        self.assertEqual(groups['a'], groups['b'])


class SupplementAllocationTests(unittest.TestCase):
    def test_selection_is_deterministic_and_category_proportional(self):
        categories = {str(i): ('animal' if i < 3 else 'otc' if i < 7 else 'prescription')
                      for i in range(11)}
        chosen = select_groups(set(categories), 7, categories, 20261010, 'train')
        self.assertEqual(Counter(categories[g] for g in chosen),
                         Counter(animal=2, otc=3, prescription=2))
        self.assertEqual(chosen, select_groups(set(reversed(categories)), 7, categories, 20261010, 'train'))
        remaining = set(categories) - chosen
        validation = select_groups(remaining, 2, categories, 20261010, 'val')
        self.assertFalse(chosen & validation)
        self.assertEqual(len(remaining - validation), 2)

    def test_invalid_budgets_are_rejected(self):
        for count in (-1, 2):
            with self.assertRaises(ValueError):
                select_groups({'a'}, count, {'a': 'otc'}, 1, 'train')
        self.assertEqual(select_groups(set(), 0, {}, 1, 'train'), set())


if __name__ == '__main__':
    unittest.main()
