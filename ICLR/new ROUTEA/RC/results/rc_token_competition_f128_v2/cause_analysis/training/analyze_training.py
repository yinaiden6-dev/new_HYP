#!/usr/bin/env python3
"""Read-only reconstruction of completed F128 V2 training/selection evidence.

Reads only this result's JSON/CSV plus frozen source text. No torch, model
forward, training, scheduler calls, or new held-label sources are used.
Writes derived files only below RESULT/cause_analysis/training/.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ORIGINAL_RC = Path('/hkfs/work/workspace/scratch/ap7811-benchmark/ICLR/new ROUTEA/RC')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--source-root', type=Path,
                        help='Checkout RC directory containing programs/ and src/; defaults to ROOT/../..')
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = (args.source_root or root.parent.parent).resolve()
    out = root / 'cause_analysis/training'
    out.mkdir(parents=True, exist_ok=True)
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol = read(root / 'protocol.json')
    joined = read(root / 'joined_predictions.json')['records']
    with (root / 'analysis/per_query_diagnostics.csv').open() as stream:
        qmeta = {r['query_id']: r for r in csv.DictReader(stream)}
    bykey = {(r['model'], r['operating_point'], r['query_id']): r for r in joined}
    base = {}
    for c in read(root / 'candidate_predictions.json')['records']:
        if c['model'] == 'TOKEN_QR':
            base.setdefault(c['query_id'], []).append(c)
    for rows in base.values():
        rows.sort(key=lambda r: r['position'])

    def decision(r):
        z, anchor = r['logits'], r['winner']
        challengers = [i for i in range(len(z)) if i != anchor]
        best = max(z[i] for i in challengers)
        tops = [i for i in challengers if z[i] == best]
        return tops[0] if len(tops) == 1 and best > 0 else anchor

    def loss(r):
        z, meta = r['logits'], qmeta[r['query_id']]
        if meta['target_in_c128'] == 'True':
            position = r['axis'].index(int(meta['target_physical_row']))
            peak = max(z)
            return peak + math.log(sum(math.exp(x - peak) for x in z)) - z[position]
        return sum(max(x, 0) + math.log1p(math.exp(-abs(x)))
                   for i, x in enumerate(z) if i != r['winner']) / (len(z) - 1)

    def correct(r):
        meta = qmeta[r['query_id']]
        return (meta['target_in_c128'] == 'True'
                and r['axis'][decision(r)] == int(meta['target_physical_row']))

    def writecsv(name, rows):
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fits, epochs, query_losses, hashes, val_identities = [], [], [], {}, {}
    max_ce_error = 0.0
    epoch_records_count = 0
    source_keys = ['programs/run_token_competition_v2.py',
                   'programs/run_rebut_qr_qrr_frozenbase_v2.py',
                   'src/rc_aslo_xf/token_competition_v2.py',
                   'src/rc_aslo_xf/rebut_qr_qrr_v1.py',
                   'programs/token_competition_data_v2.py']
    for key in source_keys:
        binding = protocol['sources'][key]
        # Relocate this frozen experiment's known RC source prefix only. Never
        # fall back to an absolute workspace path when the checkout file is absent.
        declared_path = Path(binding['path'])
        relative = declared_path.relative_to(ORIGINAL_RC)
        if relative.as_posix() != key or '..' in relative.parts:
            raise ValueError(f'Unexpected frozen source binding: {binding["path"]}')
        resolved_path = source_root / relative
        actual = sha(resolved_path)
        hashes[key] = dict(expected=binding['sha256'], actual=actual,
                           match=actual == binding['sha256'],
                           declared_path=binding['path'], resolved_path=str(resolved_path))
        assert hashes[key]['match'], key

    def query_loss_row(fold, arm, stage, original, predicted):
        q = predicted['query_id']
        baseline_loss, model_loss = loss(original), loss(predicted)
        return dict(fold=fold, arm=arm, stage=stage, query_id=q,
                    identity=qmeta[q]['identity'],
                    target_present=qmeta[q]['target_in_c128'] == 'True',
                    baseline_loss=baseline_loss, model_loss=model_loss,
                    delta_loss=model_loss - baseline_loss,
                    baseline_correct=correct(original), model_correct=correct(predicted),
                    baseline_selected=original['axis'][decision(original)],
                    model_selected=predicted['axis'][decision(predicted)])

    for path in sorted(root.glob('fold*/seed0/*/result.json')):
        result = read(path)
        iv = read(path.parent / 'inner_validation.json')
        progress = read(path.parent / 'progress.json')
        selection = result['selection']
        arm, fold = result['binding']['arm'], result['binding']['fold']
        inner = [h for h in result['history'] if h['stage'] == 'inner']
        refit = [h for h in result['history'] if h['stage'] == 'refit']
        chosen = next(h for h in inner if h['epoch'] == selection['main_epoch'])
        eligible = [h for h in inner if h['fixed_zero_correct'] > inner[0]['fixed_zero_correct']]
        key = lambda h: (h['validation_loss'], h['epoch'])
        expected = min(eligible, key=key) if eligible else inner[0]
        best_ce = min(inner, key=key)
        assert expected['epoch'] == selection['main_epoch'] == best_ce['epoch'] == selection['best_ce_epoch']
        assert iv['selection'] == selection
        b_inner = {r['query_id']: r for r in iv['baseline_records']}
        for h in inner:
            ep = read(path.parent / f"inner/epoch{h['epoch']:03d}.json")
            epoch_records_count += 1
            rows = ep['records']
            error = abs(sum(loss(z) for z in rows) / len(rows) - h['validation_loss'])
            max_ce_error = max(max_ce_error, error)
            assert error < 1e-12
            rescues = sum(correct(z) and not correct(b_inner[z['query_id']]) for z in rows)
            breaks = sum(not correct(z) and correct(b_inner[z['query_id']]) for z in rows)
            cor = sum(correct(z) for z in rows)
            assert (rescues, breaks, cor) == (h['rescues_vs_frozen_baseline'],
                                             h['breaks_vs_frozen_baseline'], h['fixed_zero_correct'])
            assert ep['metrics'] == h
            if h['epoch'] == selection['main_epoch']:
                assert rows == iv['records']
            epochs.append(dict(fold=fold, arm=arm, stage='inner', epoch=h['epoch'],
                               selected=h['epoch'] == selection['main_epoch'],
                               train_loss=h['train_loss'], validation_loss=h['validation_loss'],
                               correct=cor, baseline_correct=h['baseline_fixed_zero_correct'],
                               rescues=rescues, breaks=breaks, net=rescues-breaks,
                               eligible=h['eligible_main']))
        for h in refit:
            epochs.append(dict(fold=fold, arm=arm, stage='refit', epoch=h['epoch'], selected=False,
                               train_loss=h['train_loss'], validation_loss=None, correct=None,
                               baseline_correct=None, rescues=None, breaks=None, net=None, eligible=None))
        for z in iv['records']:
            query_losses.append(query_loss_row(fold, arm, 'selected_inner', b_inner[z['query_id']], z))
        held = [z for z in joined if z['model'] == arm and z['operating_point'] == 'zero' and z['fold'] == fold]
        bh = {z['query_id']: z for z in joined if z['model'] == 'B_CAL'
              and z['operating_point'] == 'zero' and z['fold'] == fold}
        hr = sum(z['correct'] and not bh[z['query_id']]['correct'] for z in held)
        hb = sum(not z['correct'] and bh[z['query_id']]['correct'] for z in held)
        held_losses, base_losses = [], []
        for z in result['predictions']:
            q = z['query_id']
            bz = dict(query_id=q, axis=[c['physical_row'] for c in base[q]],
                      winner=z['winner'], logits=[c['zero_base'] for c in base[q]])
            assert bz['axis'] == z['axis']
            assert correct(bz) == bh[q]['correct']
            assert correct(z) == bykey[arm, 'zero', q]['correct']
            entry = query_loss_row(fold, arm, 'held_outer_refit', bz, z)
            query_losses.append(entry)
            held_losses.append(entry['model_loss'])
            base_losses.append(entry['baseline_loss'])
        phases = ['inner_fit_query_ids', 'inner_val_query_ids', 'heldout_query_ids']
        ids = {k: {qmeta[q]['identity'] for q in result[k]} for k in phases}
        val_identities[str(fold)] = sorted(ids['inner_val_query_ids'])
        assert not ids[phases[0]] & ids[phases[1]]
        assert not (ids[phases[0]] | ids[phases[1]]) & ids[phases[2]]
        gradient = result['gradient_summary']
        assert gradient['inner']['updates'] == len(result[phases[0]]) * inner[-1]['epoch']
        assert gradient['refit']['updates'] == len(result['train_query_ids']) * selection['main_epoch']
        assert progress['updates'] == gradient['inner']['updates'] + gradient['refit']['updates']
        assert progress['status'] == 'COMPLETE' and progress['stage'] == 'done'
        assert result['head_exact_unchanged']
        assert all(v['bit_exact'] and v['maximum_error'] == 0 for v in result['baseline_replay'].values())
        for stage in ['inner', 'refit']:
            assert gradient[stage]['nonzero_reader_updates'] == gradient[stage]['updates']
            assert gradient[stage]['nonzero_stem_updates'] == gradient[stage]['updates'] - 1
        receipt = read(root / f'independent_receipts/fold{fold}_seed0_{arm}.json')
        assert receipt['max_replay_error'] == 0
        assert receipt['result']['sha256'] == sha(path)
        assert receipt['selected_epoch'] == selection['main_epoch']
        fits.append(dict(
            fold=fold, arm=arm, seed=0, fit_n=len(result[phases[0]]), val_n=len(result[phases[1]]),
            outer_train_n=len(result['train_query_ids']), held_n=len(result[phases[2]]),
            fit_identities=len(ids[phases[0]]), val_identities=len(ids[phases[1]]), held_identities=len(ids[phases[2]]),
            trainable_parameters=result['trainable_parameters'], selected_epoch=selection['main_epoch'],
            best_ce_epoch=selection['best_ce_epoch'], stop_epoch=inner[-1]['epoch'], fallback=selection['epoch0_fallback'],
            inner_baseline_correct=inner[0]['fixed_zero_correct'], inner_selected_correct=chosen['fixed_zero_correct'],
            inner_rescues=chosen['rescues_vs_frozen_baseline'], inner_breaks=chosen['breaks_vs_frozen_baseline'],
            inner_net=chosen['net_vs_frozen_baseline'], inner_baseline_loss=inner[0]['validation_loss'],
            inner_selected_loss=chosen['validation_loss'], inner_final_loss=inner[-1]['validation_loss'],
            inner_train_loss_epoch1=inner[1]['train_loss'], inner_train_loss_selected=chosen['train_loss'],
            inner_train_loss_final=inner[-1]['train_loss'], refit_train_loss_epoch1=refit[0]['train_loss'],
            refit_train_loss_final=refit[-1]['train_loss'], held_rescues=hr, held_breaks=hb, held_net=hr-hb,
            held_baseline_loss=sum(base_losses)/len(base_losses), held_model_loss=sum(held_losses)/len(held_losses),
            inner_updates=gradient['inner']['updates'], refit_updates=gradient['refit']['updates'],
            refit_to_inner_epoch_update_ratio=len(result['train_query_ids'])/len(result[phases[0]]),
            inner_max_correct=max(h['fixed_zero_correct'] for h in inner),
            eligible_with_zero_break_exists=any(h['eligible_main'] and h['breaks_vs_frozen_baseline'] == 0 for h in inner),
            all_epoch_metrics_recomputed=True, receipt_sha_verified=True))

    assert len(fits) == 15
    writecsv('per_fit.csv', fits)
    writecsv('epoch_trajectories.csv', epochs)
    writecsv('per_query_loss.csv', query_losses)
    summary = {}
    for arm in protocol['arms']:
        fit = [x for x in fits if x['arm'] == arm]
        val = [x for x in query_losses if x['arm'] == arm and x['stage'] == 'selected_inner']
        held = [x for x in query_losses if x['arm'] == arm and x['stage'] == 'held_outer_refit']
        summary[arm] = dict(
            epochs=[x['selected_epoch'] for x in fit],
            inner_rescues_occurrences=sum(x['inner_rescues'] for x in fit),
            inner_breaks_occurrences=sum(x['inner_breaks'] for x in fit),
            inner_net_occurrences=sum(x['inner_net'] for x in fit), inner_query_occurrences=len(val),
            inner_unique_queries=len({x['query_id'] for x in val}),
            held_rescues=sum(x['held_rescues'] for x in fit), held_breaks=sum(x['held_breaks'] for x in fit),
            held_net=sum(x['held_net'] for x in fit),
            mean_inner_baseline_loss=sum(x['baseline_loss'] for x in val)/len(val),
            mean_inner_selected_loss=sum(x['model_loss'] for x in val)/len(val),
            mean_held_baseline_loss=sum(x['baseline_loss'] for x in held)/len(held),
            mean_held_model_loss=sum(x['model_loss'] for x in held)/len(held),
            held_break_queries=[x['query_id'] for x in held if x['baseline_correct'] and not x['model_correct']])
    audit = dict(
        scope='All15 native token V2 F128 grouped-fivefold seed0; unchanged C128; fixed0 primary; phase-specific same-F128 B_CAL. Existing current-panel labels only. No training, inference, or protected data read.',
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        inputs={name: dict(path=str(root/name), sha256=sha(root/name)) for name in
                ['protocol.json', 'joined_predictions.json', 'candidate_predictions.json', 'analysis/per_query_diagnostics.csv']},
        sources_verified=hashes, checked_fits=len(fits), checked_epoch_records=epoch_records_count,
        maximum_recomputed_inner_loss_error=max_ce_error,
        selected_equals_global_best_ce_all_fits=True, selected_equals_frozen_eligible_min_ce_all_fits=True,
        all_gradient_counts_consistent=True, all_progress_complete=True,
        all_baseline_replay_bit_exact_in_existing_results=True,
        existing_independent_receipt_result_shas_verified=True, existing_receipts_model_replay_error=0,
        new_model_replay_performed=False, inner_identity_sets=val_identities, summary_by_arm=summary,
        caveats=['Inner query occurrences overlap across folds and are selected-on, not independent OOF evidence.',
                 'Train loss averages evolving per-query update losses, not a post-epoch frozen-model evaluation.',
                 'Held model is fresh outer-TRAIN refit; inner-to-held discrepancy mixes unseen identities and refit changes.',
                 'Opened development panel, seed0 only; specific hyperparameter/data-size/architecture causes are not isolated.'])
    (out/'audit_summary.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', checked_fits=len(fits), checked_epoch_records=epoch_records_count,
                          maximum_recomputed_inner_loss_error=max_ce_error, script=audit['script']), indent=2))


if __name__ == '__main__':
    main()
