"""Real TRAIN-only arithmetic and synthetic end-to-end prelabel closure checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'programs'))
import rc_internal_m_cpu_controls_v1 as c


class CPUControlsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = c.read(ROOT / 'results/rc_prellm_m_adapter_v1/input_manifest.json')
        cls.theta = cls.manifest['frozen_head']['theta']

    def test_existing_additive_and_loss_parity(self):
        import torch
        import run_rc_prellm_m_pilot_v1 as original
        theta, provenance = c.load_additive(self.manifest, ROOT / 'results/rc_h593_simple_explanations_v1')
        self.assertEqual(len(theta), 4)
        self.assertLess(provenance['max_replay_error'], 1e-10)
        for row in self.manifest['train_rows']:
            z = torch.tensor(c.head_prediction(row, row['L0'], self.theta)['logits'], dtype=torch.float64)
            loss = original.cost(row, torch.tensor(row['L0'], dtype=torch.float64), self.manifest)
            self.assertAlmostEqual(float(c.cost1(z, row)), float(loss), places=11)

    def test_fit_only_delta_and_deterministic(self):
        original_theta = list(self.theta)
        first = c.fit_score_update(self.manifest['train_rows'], self.theta)
        second = c.fit_score_update(self.manifest['train_rows'], self.theta)
        self.assertEqual(first, second)
        self.assertEqual(self.theta, original_theta)
        self.assertEqual(len(first['history']), 16)
        self.assertEqual(first['held_label_reads'], 0)
        self.assertTrue(np.isfinite(first['theta']).all())
        self.assertTrue(any(v != 0 for v in first['delta_theta']))

    def test_synthetic_preseal_join_and_fail_closed(self):
        # Contents are intentionally unchanged mock encoder outputs. No real
        # held labels are opened; synthetic target identities below are arbitrary.
        with tempfile.TemporaryDirectory(prefix='internal-m-cpu-controls-') as temporary:
            out = Path(temporary); manifest_path = out / 'manifest.json'
            c.write(manifest_path, self.manifest)
            authority_path = out / 'authority.json'
            c.write(authority_path, dict(manifest=c.bind(manifest_path), arms=list(c.ARMS)))
            authority = c.bind(authority_path)
            arms = {arm: [] for arm in c.ARMS}; adapters = {}
            for arm in c.ARMS:
                c.write(out / arm / 'adapter.json', dict(synthetic=True, arm=arm))
                adapters[arm] = c.bind(out / arm / 'adapter.json')
            synthetic_rows = []
            for split, rows in (('train', self.manifest['train_rows']), ('probe', self.manifest['probe_rows'])):
                for row in rows:
                    q = row['query_id']; directory = out / 'encoder_cache' / q
                    c.write(directory / 'parity.json', dict(query_id=q, original_L0=row['L0'], fresh_L0=row['L0']))
                    c.write(directory / 'validation.json', dict(authority=authority, status='QUERY_ENCODER_CACHE_PASS',
                        held_labels_read=False, parity=c.bind(directory / 'parity.json')))
                    for arm in c.ARMS:
                        for intervention in (('native', 'shuffled') if arm.endswith('REAL') else ('native',)):
                            path = out / arm / 'predictions' / split / (q + '_' + intervention + '.json')
                            c.write(path, dict(authority=authority, adapter=adapters[arm], query_id=q, arm=arm,
                                intervention=intervention, L=row['L0'], M=row['M'], raw_scores=row['raw_scores'],
                                candidate_ids=row['candidate_ids'], held_label_reads=0,
                                decision=c.head_prediction(row, row['L0'], self.theta)))
                            arms[arm].append(c.bind(path))
                    target = 0  # Explicitly synthetic, never the probe's real target.
                    synthetic_rows.append(dict(query_id=q, split=split, component='SYNTHETIC_' + q,
                        identity=row['candidate_identities'][target], target_positions=[target], target_in_C128=True,
                        correct=dict(RAW=target == row['winner_index']),
                        selected_physical_row=dict(RAW=row['candidate_ids'][row['winner_index']])))
            seals = []
            for arm in c.ARMS:
                path = out / arm / 'prediction_seal.json'
                c.write(path, dict(authority=authority, adapter=adapters[arm], predictions=arms[arm], held_label_reads=0))
                seals.append(c.bind(path))
            c.write(out / 'all_predictions_prelabel_seal.json', dict(authority=authority, seals=seals, held_label_reads=0))
            seal = c.run_controls(authority_path, self.manifest, out)
            self.assertEqual(len(seal['predictions']), 24)
            self.assertEqual(seal['held_label_reads'], 0)
            self.assertEqual(c.run_controls(authority_path, self.manifest, out), seal)
            joined = dict(status='INTERNAL_M_PILOT_POSTSEAL_JOIN_COMPLETE', authority=authority, rows=synthetic_rows)
            result = c.summarize_controls(joined, seal)
            self.assertEqual(result['summary']['PROBE8']['count'], 8)
            for panel in result['summary'].values():
                self.assertEqual(set(panel['position_interactions']), {'PRODUCT5', 'ADDITIVE4', 'PURE_L'})
                self.assertTrue(all(v['position_difference_in_net_correct'] == 0 for v in panel['position_interactions'].values()))
            corrupted = copy.deepcopy(self.manifest); corrupted['probe_rows'][0]['target_positions'] = [0]
            c.write(manifest_path, corrupted)
            c.write(authority_path, dict(manifest=c.bind(manifest_path), arms=list(c.ARMS)))
            with self.assertRaisesRegex(RuntimeError, 'CONTROL_LABEL_FREE_PROBE'):
                c.run_controls(authority_path, corrupted, out)


if __name__ == '__main__':
    unittest.main()
