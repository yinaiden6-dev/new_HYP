#!/usr/bin/env python3
"""Independent fixed-panel action/rank audit; labels open only after the fit seal."""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'programs'), str(ROOT / 'src')]
import materialize_rc_new_hyp593_inputs_v1 as M
import run_rc_new_hyp593_oof5_v1 as P

OUT = ROOT / 'results/rc_fixed_panels_train269_group_risk_v1'
AUTH = ROOT / 'registry/rc_fixed_panels_train269_group_risk_authority_v1_20260911.json'
MODELS = ('ORIGINAL7', 'IMAGE269', 'GROUP269')
COMPARISONS = (('RAW', 'ORIGINAL7'), ('ORIGINAL7', 'IMAGE269'),
               ('ORIGINAL7', 'GROUP269'), ('IMAGE269', 'GROUP269'))
HEX7 = ('0x1.01ffd7d7241b3p+0', '-0x1.e3db1bf7526f8p+1',
        '0x1.6ddd66d0df5d3p+2', '0x1.7a67f045c5bd2p+2',
        '-0x1.3095a38e39e50p-3', '-0x1.e8ea4dd00f544p-3',
        '-0x1.7a55fbee7993ap+0')
EC7 = 'ec7df7e5a5b85f725f8729aed91653e1028814a8bbc0474039dbaffab652b263'


def need(value, label):
    if not value:
        raise RuntimeError(label)


def index(rows, key):
    result = {r[key]: r for r in rows}
    need(len(result) == len(rows), 'UNIQUE_' + key)
    return result


def close(actual, expected, label):
    need(math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-12, label)


def fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def action(logits, challengers, winner):
    # Python max preserves the first maximum; zero has the explicit HOLD branch.
    j = max(range(len(logits)), key=lambda i: float(logits[i]))
    return challengers[j] if float(logits[j]) > 0 else winner


def comparison(rows, base, new):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['component']].append(int(r['correct'][new]) - int(r['correct'][base]))
    means = {g: Fraction(sum(v), len(v)) for g, v in sorted(grouped.items())}
    ds = np.asarray([float(v) for v in means.values()], dtype=np.float64)
    rng = np.random.default_rng(20260911)
    samples = ds[rng.integers(0, len(ds), size=(10000, len(ds)))].mean(axis=1)
    summary = dict(rescue=sum(not r['correct'][base] and r['correct'][new] for r in rows),
                   loss=sum(r['correct'][base] and not r['correct'][new] for r in rows),
                   equal_component_difference=float(sum(means.values(), Fraction()) / len(means)),
                   component_bootstrap95=[float(v) for v in np.quantile(samples, [.025, .975])],
                   positive_components=sum(v > 0 for v in means.values()),
                   negative_components=sum(v < 0 for v in means.values()), components=len(means))
    return summary, {g: fraction(v) for g, v in means.items()}


def main():
    need(os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED')
    need(M.datetime.now(M.timezone.utc) < M.DEADLINE, 'USER_CUTOFF')
    authority = M.read(AUTH)
    need(authority['status'] == 'FIXED_OLD_PANELS_FRESH269_GROUP_RISK_AUTHORIZED', 'AUTHORITY_STATUS')
    pins = authority['sources']
    expected_public = {
        'program': ROOT / 'programs/run_rc_fixed_panels_train269_group_risk_v1.py',
        'validator': Path(__file__).resolve(),
        'historic_input_validator': ROOT / 'programs/validate_rc_fixed269_historic_inputs_v1.py',
        'reporter': ROOT / 'programs/report_rc_fixed_panels_train269_group_risk_v1.py',
        'freezer': ROOT / 'programs/freeze_rc_fixed_panels_train269_group_risk_v1.py',
        'input_helper': ROOT / 'programs/materialize_rc_new_hyp593_inputs_v1.py',
        'feature_and_image_fit_helper': ROOT / 'programs/run_rc_new_hyp593_oof5_v1.py',
        'group_fit_helper': ROOT / 'programs/run_rc_h593_group_risk_strong_base_v1.py',
        'original_parameter_seal': ROOT / 'registry/rc_shared_query_target_prior_native7_c_head_parameter_seal_v1_20260909.json',
        'feature_authority': ROOT / 'registry/rc_new_hyp593_feature_authority_v1_20260911.json',
        'panel_manifest': OUT / 'panel_manifest.json', 'train_labels': OUT / 'train_roles.json',
        'metadata_validation': OUT / 'metadata_validation.json',
        'plan': ROOT / 'plan/RC_FIXED_PANELS_TRAIN269_GROUP_RISK_V1_20260911.md',
        'launcher': ROOT / 'slurm/rc_fixed_panels_train269_group_risk_v1.sbatch',
    }
    expected_eval = {
        'eval_labels': OUT / 'eval_curator_roles.json',
        'original128_curator': ROOT / 'results/rc_original7_expanded_eval128_manifest_v1/curator_roles.json',
        'h593_curator': P.OUT / 'metadata/curator_roles.json',
        'original_scope_exclusions': ROOT / 'registry/rc_eval128_identity_group_exclusion_v1_20260910.json',
        'original32_result': ROOT / 'results/routea_matched_three_arm_common3_native7_crossfit_v1/result.json',
        'original128_result': ROOT / 'results/rc_original7_eval128_full_evidence_v1/result.json',
    }
    need(set(pins) == set(expected_public), 'EXACT_PUBLIC_SOURCE_KEYS')
    need(set(authority['evaluation_sources']) == set(expected_eval), 'EXACT_EVAL_SOURCE_KEYS')
    for key, path in expected_public.items():
        need(pins[key] == M.bind(path), 'PUBLIC_SOURCE_' + key)
    for key, path in expected_eval.items():
        need(Path(authority['evaluation_sources'][key]['path']) == path, 'EVAL_SOURCE_PATH_' + key)
    # Enforce the actual read order, including accidental helper reads.
    labels_open = False
    eval_paths = {p.absolute() for p in expected_eval.values()}
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        s = str(p).lower()
        need(not any(x in s for x in ('rc_opened_', 'd1-mi', 'd1_mi', 'grozi', 'gisc_prerecall_universe', '/target_join/')), 'PROTECTED_INPUT')
        need(labels_open or p not in eval_paths, 'FIT_SEAL_BEFORE_EVAL_LABEL_READ')
    sys.addaudithook(audit)
    ab = M.bind(AUTH)
    preflight = M.read(OUT / 'preflight.json')
    need(preflight['status'] == 'FIXED_PANEL_GROUP_RISK_PREFLIGHT_PASS' and preflight['authority'] == ab, 'PREFLIGHT_BINDING')
    fitval = M.read(OUT / 'fits/validation.json')
    receipt = M.read(OUT / 'fits/receipt.json')
    need(fitval['status'] == 'FIXED_PANEL_RETRAIN_AND_PREDICTION_REPLAY_PASS', 'FIT_REPLAY_PASS')
    need(fitval['payload'] == receipt['payload'] and fitval['receipt'] == M.bind(OUT / 'fits/receipt.json'), 'FIT_RECEIPT_BINDING')
    need(Path(receipt['payload']['path']) == OUT / 'fits/payload.pt', 'ACTIVE_FIT_PAYLOAD')
    need(fitval['authority'] == receipt['authority'] == ab, 'FIT_AUTHORITY')
    need(fitval['heldout_label_reads'] == receipt['heldout_label_reads'] == 0 and fitval['prior_593_model_reads'] == 0, 'FIT_READ_ATTESTATIONS')
    need(isinstance(fitval.get('nonce'), str) and len(fitval['nonce']) == 32, 'FRESH_REPLAY_NONCE')
    payload = torch.load(M.checked(receipt['payload']), map_location='cpu', weights_only=True)
    need(payload['authority'] == ab and payload['train_role'] == pins['train_labels'] and payload['panels'] == pins['panel_manifest'], 'PAYLOAD_BINDINGS')
    need(payload['heldout_label_reads'] == payload['prior_593_model_reads'] == 0, 'PAYLOAD_READ_ATTESTATIONS')
    need(payload['steps'] == 2000 and payload['seed'] == 17 and payload['from_zero'] is True and payload['train_images'] == 269, 'FIT_SCOPE')
    seal = M.read(expected_public['original_parameter_seal'])
    need(seal['parameter_sha256'] == EC7 and tuple(seal['weight_binary64'] + [seal['bias_binary64']]) == HEX7, 'FROZEN_EC7_SEAL_HEX')
    need(set(payload['parameters']) == set(MODELS), 'MODEL_SET')
    theta = {}
    for model in MODELS:
        t = payload['parameters'][model]['theta']
        need(isinstance(t, torch.Tensor) and t.dtype == torch.float64 and tuple(t.shape) == (7,) and bool(torch.isfinite(t).all()), 'THETA_' + model)
        need('beta' not in payload['parameters'][model], 'ORIGINAL_SEVEN_PARAMETER_ARCHITECTURE')
        theta[model] = t.numpy()
    need(tuple(float(x).hex() for x in theta['ORIGINAL7']) == HEX7, 'ORIGINAL7_EXACT_BINARY64')
    features, feature_sources = P.features()
    need(payload['feature_sources'] == feature_sources, 'VALIDATED_FEATURE_SOURCE_BINDINGS')
    fi = index(features, 'query_id')
    split = M.read(OUT / 'panel_manifest.json')
    need(set(split['panels']) == {'EVAL32', 'EVAL128'} and split['selection_used_outcomes'] is False, 'FIXED_PANEL_SELECTION')
    train = index(M.read(OUT / 'train_roles.json')['records'], 'query_id')
    need(len(train) == len(split['train_query_ids']) == 269 and set(train) == set(split['train_query_ids']), 'TRAIN269')
    need(M.read(OUT / 'train_roles.json')['evaluation_labels_included'] is False, 'TRAIN_ROLE_NO_EVAL_LABELS')
    qids = [q for qs in split['panels'].values() for q in qs]
    pred = index(payload['predictions'], 'query_id')
    need(len(qids) == len(set(qids)) == len(pred) == 160 and set(qids) == set(pred), 'ALL160_PREDICTIONS')
    chosen = {}
    max_error = 0.0
    need(action([0.0, 0.0], [2, 3], 1) == 1 and action([1.0, 1.0], [2, 3], 1) == 2, 'PYTHON_TIE_AND_HOLD_ZERO')
    for q in qids:
        f, p = fi[q], pred[q]
        for k in ('query_id', 'execution_ordinal', 'candidate_physical_rows', 'raw_ranked_physical_rows', 'winner', 'challenger_positions'):
            need(p[k] == f[k], 'PRED_FEATURE_AXIS_' + k)
        axis = list(f['candidate_physical_rows'])
        raw = list(f['raw_ranked_physical_rows'])
        challengers = list(f['challenger_positions'])
        need(len(axis) == 128 and axis == sorted(set(axis)), 'SORTED_FULL_C128')
        need(len(raw) == len(set(raw)) and set(axis) == set(raw[:128]) and axis[f['winner']] == raw[0], 'FULLGALLERY_RAW_C128')
        need(challengers == [j for j in range(128) if j != f['winner']], '127_CHALLENGERS_IN_ORDER')
        need(set(p['models']) == set(MODELS), 'PRED_MODEL_SET')
        selected = {'RAW': raw[0]}
        for model in MODELS:
            need(set(p['models'][model]) == {'REAL', 'CBIND'}, 'PRED_MODE_SET')
            for mode in ('REAL', 'CBIND'):
                x = f['modes'][mode]['X']
                need(x.dtype == torch.float64 and tuple(x.shape) == (127, 6), 'FEATURE127X6')
                z = np.asarray(x.numpy(), dtype=np.float64) @ theta[model][:6] + theta[model][6]
                reported = p['models'][model][mode]
                observed = reported['logits']
                need(observed.dtype == torch.float64 and tuple(observed.shape) == (127,), '127_LOGITS')
                need(np.isfinite(z).all() and np.isfinite(observed.numpy()).all(), 'FINITE_LOGITS')
                error = float(np.max(np.abs(z - observed.numpy())))
                max_error = max(max_error, error)
                need(error <= 1e-12, 'NUMPY_LOGIT_REPLAY')
                position = action(z, challengers, f['winner'])
                need(position == reported['selected_position'], 'INDEPENDENT_HOLD_SWITCH')
                selected[model if mode == 'REAL' else model + '_CBIND'] = axis[position]
        chosen[q] = selected
    # Only now are evaluation metadata, gallery labels and historical outcomes read.
    labels_open = True
    for key, path in expected_eval.items():
        need(authority['evaluation_sources'][key] == M.bind(path), 'EVAL_SOURCE_' + key)
    roles = index(M.read(expected_eval['eval_labels'])['records'], 'query_id')
    allroles = M.read(expected_eval['h593_curator'])['records']
    allbyid = index(allroles, 'query_id')
    allbyoriginal = index(allroles, 'original_query_id')
    allbysha = index(allroles, 'source_image_sha256')
    old128rows = M.read(expected_eval['original128_curator'])['records']
    old128sha = index(old128rows, 'source_image_sha256')
    exclusions = M.read(expected_eval['original_scope_exclusions'])
    ti, tg = set(exclusions['training_union_identities']), set(exclusions['training_union_groups'])
    expected_train = {r['query_id'] for r in allroles if r['identity'] in ti and r['group'] in tg}
    need(set(train) == expected_train and len({r['identity'] for r in train.values()}) == len({r['group'] for r in train.values()}) == 32, 'ORIGINAL_TRAIN_SCOPE')
    need(split['panels']['EVAL32'] == [allbyoriginal[q]['query_id'] for q in exclusions['split_sets']['FULL_EVAL32']['query_ids']], 'ORIGINAL32_MEMBERSHIP_ORDER')
    need(split['panels']['EVAL128'] == [allbysha[r['source_image_sha256']]['query_id'] for r in old128rows], 'ORIGINAL128_SHA_MEMBERSHIP_ORDER')
    need(set(roles) == set(qids), 'EXACT_EVAL_LABEL_SET')
    safe = ('query_id', 'original_query_id', 'execution_ordinal', 'source_image_sha256', 'identity', 'group', 'component')
    for q, r in list(train.items()) + list(roles.items()):
        need(all(r[k] == allbyid[q][k] for k in safe), 'CURATOR_ROLE_PROVENANCE')
        need(fi[q]['source_image_sha256'] == r['source_image_sha256'] and fi[q]['execution_ordinal'] == r['execution_ordinal'], 'ROLE_FEATURE_IMAGE_BINDING')
    overlaps = {}
    for panel, qs in split['panels'].items():
        need(len(qs) == {'EVAL32': 32, 'EVAL128': 128}[panel], 'PANEL_COUNT')
        overlaps[panel] = {}
        for k in ('query_id', 'source_image_sha256', 'identity', 'group', 'component'):
            overlaps[panel][k] = len({r[k] for r in train.values()} & {roles[q][k] for q in qs})
            need(overlaps[panel][k] == 0, 'TRAIN_EVAL_DISJOINT_' + panel + '_' + k)
    for field in ('query_id', 'source_image_sha256', 'identity', 'group', 'component'):
        need(not ({roles[q][field] for q in split['panels']['EVAL32']} & {roles[q][field] for q in split['panels']['EVAL128']}), 'EVAL_PANELS_MUTUALLY_DISJOINT_' + field)
    metadata = M.read(OUT / 'metadata_validation.json')
    need(metadata['status'] == 'FIXED269_TRAIN_TEST_IMAGE_IDENTITY_GROUP_EXCLUSION_PASS' and metadata['overlaps'] == overlaps, 'METADATA_VALIDATION_RECHECK')
    need(metadata['counts'] == dict(train=269, train_identities=32, train_groups=32, EVAL32=32, EVAL128=128), 'METADATA_COUNTS')
    need(metadata['train_role'] == pins['train_labels'] and metadata['eval_role'] == authority['evaluation_sources']['eval_labels'] and metadata['panels'] == pins['panel_manifest'], 'METADATA_ROLE_BINDINGS')
    for b in metadata['sources'].values():
        M.checked(b)
    from rc_aslo_xf.conditional_rep_sources import build_gallery_source
    labels = build_gallery_source(verify_cache_file_sha256=True).corrected_identities
    eligible = [q for q in split['train_query_ids'] if any(labels[p] == train[q]['identity'] for p in fi[q]['candidate_physical_rows'])]
    counts = Counter(train[q]['component'] for q in eligible)
    need(payload['loss_eligible_images'] == len(eligible) and payload['target_absent_train'] == 269 - len(eligible), 'TRAIN_RECALL_ACCOUNTING')
    need(payload['effective_groups'] == len(counts) and payload['effective_group_image_counts'] == dict(counts), 'TRAIN_GROUP_ACCOUNTING')
    # The producer trains in feature execution order, so validate the weight axis there.
    eligible_set = set(eligible)
    ordered = [r['query_id'] for r in features if r['query_id'] in eligible_set]
    weights = payload['group_weights'].numpy()
    expected_weights = np.array([1 / (len(counts) * counts[train[q]['component']]) for q in ordered])
    need(weights.shape == expected_weights.shape and np.max(np.abs(weights - expected_weights)) <= 1e-15, 'EQUAL_GROUP_TRAIN_WEIGHTS')
    historic = M.read(OUT / 'historic_input_parity.json')
    need(historic['status'] == 'FIXED269_HISTORIC_INPUT_PARITY_PASS' and historic['all_pass'] is True, 'HISTORIC_INPUT_PARITY_PASS')
    need(historic['authority'] == ab and historic['program'] == pins['historic_input_validator'], 'HISTORIC_INPUT_PARITY_BINDING')
    need(len(historic['rows']) == 160 and {r['query_id'] for r in historic['rows']} == set(qids) and all(all(r['checks'].values()) for r in historic['rows']), 'HISTORIC_ALL160_INPUT_CHECKS')
    result = M.read(OUT / 'result.json')
    need(result['status'] == 'FIXED_PANELS_TRAIN269_GROUP_RISK_DEVELOPMENT_COMPLETE' and result['authority'] == ab and result['fit_validation'] == M.bind(OUT / 'fits/validation.json'), 'RESULT_BINDINGS')
    need(result['historically_opened_development'] is True and result['external_confirmation'] is False and result['deployment_changed'] is False, 'EVIDENCE_BOUNDARY')
    need(result['train_images'] == 269 and result['train_groups'] == 32 and result['loss_eligible_images'] == len(eligible) and result['target_absent_train'] == 269 - len(eligible), 'RESULT_TRAIN_SCOPE')
    need(set(result['panels']) == {'EVAL32', 'EVAL128'}, 'RESULT_PANELS')
    old32 = index(M.read(expected_eval['original32_result'])['evaluations']['NATIVE7']['C_PAIRED']['actions'], 'query_id')
    old128 = index(M.read(expected_eval['original128_result'])['actions']['REAL'], 'original_query_id')
    need(len(old32) == 32 and len(old128) == 128, 'OLD_ACTION_COUNTS')
    summaries = {}
    aliases = []
    for panel, qs in split['panels'].items():
        output = result['panels'][panel]
        need(output['population'] == len(qs) and len(output['rows']) == len(qs), 'RESULT_PANEL_POPULATION')
        rows = []
        for q in qs:
            role, f, selected = roles[q], fi[q], chosen[q]
            raw = list(f['raw_ranked_physical_rows'])
            targets = [p for p in raw if labels[p] == role['identity']]
            need(len(targets) == 1, 'UNIQUE_FULLGALLERY_TARGET')
            target = targets[0]
            rawrank = raw.index(target) + 1
            ranks, correct = {}, {}
            for key, physical in selected.items():
                # Literal move-to-front permutation, independent of producer rank formula.
                ranked = raw if key == 'RAW' else [physical] + [p for p in raw if p != physical]
                need(len(ranked) == len(raw) and len(set(ranked)) == len(raw), 'FULLGALLERY_PERMUTATION')
                ranks[key] = ranked.index(target) + 1
                correct[key] = physical == target
            row = dict(query_id=q, original_query_id=role['original_query_id'], component=role['component'], group=role['group'], target_in_C128=target in f['candidate_physical_rows'], correct=correct, ranks=ranks, selected_physical_rows=selected)
            rows.append(row)
            if panel == 'EVAL32':
                old = old32[role['original_query_id']]
                winner, oldtarget, oldrank = old['base_winner_physical_row'], old['target_physical_row'], old['base_target_rank']
                finalrank = old['final_target_rank']
            else:
                historical_role = old128sha[role['source_image_sha256']]
                need(all(role[k] == historical_role[k] for k in ('identity', 'group')), 'OLD128_SHA_IDENTITY_GROUP')
                oldid = historical_role['original_query_id']
                old = old128[oldid]
                winner, oldtarget, oldrank = old['RAW_winner_physical_row'], old['target_physical_row_in_full_rank'], old['raw_target_rank_full_gallery']
                finalrank = old['final_target_rank_full_gallery']
                if oldid != role['original_query_id']:
                    aliases.append(dict(query_id=q, h593_original_query_id=role['original_query_id'], old_original_query_id=oldid, source_image_sha256=role['source_image_sha256']))
            need((selected['RAW'], target, rawrank, selected['ORIGINAL7'], ranks['ORIGINAL7']) == (winner, oldtarget, oldrank, old['final_physical_row'], finalrank), 'OLD_EC7_PERQUERY_ACTION_RANK_PARITY_' + q)
            need(correct['RAW'] == old['base_correct'] and correct['ORIGINAL7'] == old['final_correct'], 'OLD_EC7_CORRECT_PARITY')
        need(output['rows'] == rows, 'EVERY_RESULT_ROW_SELECTION_CORRECT_RANK')
        recall = sum(r['target_in_C128'] for r in rows)
        need(output['recall_C128'] == recall, 'C128_RECALL_COUNT')
        scores = {}
        need(set(output['scores']) == set(rows[0]['correct']), 'ALL_SEVEN_SCORE_ARMS')
        for model in rows[0]['correct']:
            count = sum(r['correct'][model] for r in rows)
            mrr = sum((Fraction(1, r['ranks'][model]) for r in rows), Fraction()) / len(rows)
            need(output['scores'][model]['correct'] == count, 'EXACT_TOP1_COUNT')
            close(output['scores'][model]['MRR'], mrr, 'EXACT_FRACTION_MRR')
            scores[model] = dict(correct=count, MRR=float(mrr), MRR_exact=fraction(mrr))
        need((scores['RAW']['correct'], scores['ORIGINAL7']['correct']) == {'EVAL32': (25, 28), 'EVAL128': (88, 99)}[panel], 'ORIGINAL_PANEL_BASELINE_COUNTS')
        comparisons = {}
        need(set(output['comparisons']) == {a + '__to__' + b for a, b in COMPARISONS}, 'EVERY_COMPARISON')
        for base, new in COMPARISONS:
            key = base + '__to__' + new
            summary, means = comparison(rows, base, new)
            observed = output['comparisons'][key]
            need(set(observed) == set(summary), 'COMPARISON_FIELDS')
            for k, v in summary.items():
                if k == 'component_bootstrap95':
                    need(len(observed[k]) == 2, 'BOOTSTRAP_CI_SHAPE')
                    for x, y in zip(observed[k], v):
                        close(x, y, 'INDEPENDENT_COMPONENT_BOOTSTRAP95')
                elif k == 'equal_component_difference':
                    close(observed[k], v, 'EXACT_COMPONENT_MEAN')
                else:
                    need(observed[k] == v, 'COMPARISON_EXACT_' + k)
            comparisons[key] = dict(summary, component_mean_differences_exact=means)
        summaries[panel] = dict(population=len(rows), recall_C128=recall, scores=scores, comparisons=comparisons, old_ec7_perquery_parity_count=len(rows))
    validation = dict(status='FIXED_PANELS_INDEPENDENT_ACTION_RANK_COUNTS_PASS', result=M.bind(OUT / 'result.json'), program=M.bind(Path(__file__)), producer=pins['program'], historic_input_parity=M.bind(OUT / 'historic_input_parity.json'), authority=ab, fit_validation=M.bind(OUT / 'fits/validation.json'), payload=receipt['payload'], public_sources=pins, evaluation_sources=authority['evaluation_sources'], evaluation_labels_opened_after_fit_seal=True, fit_no_label_read_evidence='producer audit barrier plus authority-bound fresh-process replay and zero-read attestations; this validator does not retrain', independent_logit_method='NumPy float64 X @ theta[:6] + theta[6]', maximum_absolute_logit_error=max_error, logit_tolerance=1e-12, independently_checked_logits=160 * 3 * 2 * 127, action_tie_rule='Python max earliest challenger; HOLD when maximum <= 0', rank_method='literal selected-physical-row move to front of full RAW gallery', MRR_method='exact fractions followed by float comparison at absolute tolerance 1e-12', train_eval_overlaps=overlaps, counts=dict(train=269, EVAL32=32, EVAL128=128, full_evaluation=160), original7_parameter_sha256=EC7, original7_binary64=list(HEX7), old128_image_sha_aliases=aliases, panels=summaries, historically_opened_development=True, external_confirmation=False, deployment_changed=False, slurm_job_id=os.environ['SLURM_JOB_ID'])
    M.write(OUT / 'result_validation.json', validation)
    print(validation['status'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    main()
