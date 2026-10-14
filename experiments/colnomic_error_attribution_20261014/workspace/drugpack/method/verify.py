from evaluation.paths import project_path
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from .common import ROOT, WORK, RESULTS, evaluation_group, hidden_file, key, load_bank, read, save, sha, signature
from .ranking import restored_rank, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['caches','final'],default='final')
    parser.add_argument('--datasets',nargs='+',choices=['medicine','medicine_catalog','grozi','milk','forb32'])
    args = parser.parse_args()
    assert not args.datasets or args.stage=='caches', 'Final verification must cover every dataset'
    torch.set_num_threads(2)
    started = time.time()
    bindings = read(RESULTS/'data_bindings.json')
    protocol=read(ROOT/'configs/method/protocol.json')
    for path,digest in bindings.items():
        assert sha(path) == digest, f'Data changed: {path}'
    report = dict(stage=args.stage,started=started,bindings=len(bindings),datasets={},runs={},evaluations=0)
    medicine = read(WORK/'medicine'/'manifest.json')
    training_hashes = {r['sha256'] for r in medicine['rows'] if r['split']=='train'}
    inspected_hidden = {}
    for dataset in args.datasets or ('medicine','medicine_catalog','grozi','milk','forb32'):
        manifest = read(WORK/dataset/'manifest.json')
        payload = {k:v for k,v in manifest.items() if k!='signature'}
        assert signature(payload) == manifest['signature']
        labels = read(WORK/dataset/'labels.json')
        assert set(labels) == {r['id'] for r in manifest['rows']}
        gallery = set(manifest['gallery_indices'])
        assert all(labels[r['id']] and set(labels[r['id']]) <= gallery for r in manifest['rows'])
        if dataset.startswith('medicine'):
            for field in ('identity','group','sha256'):
                sets = [{r[field] for r in manifest['rows'] if r['split']==split}
                        for split in ('train','val','test')]
                assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
            if dataset=='medicine':
                refs = [{rid for r in manifest['rows'] if r['split']==split for rid in labels[r['id']]}
                        for split in ('train','val','test')]
                assert set(manifest['train_references'])==refs[0]
                hashes = [{manifest['references'][rid]['sha256'] for rid in ids} for ids in refs]
                assert all(not hashes[i]&hashes[j] for i in range(3) for j in range(i+1,3))
        else:
            assert not training_hashes & {r['sha256'] for r in manifest['rows']}, 'Transfer query duplicates a training photo'
        inventory = Counter()
        maximum_cache_error = 0.
        native_bank = load_bank(manifest['source']['query_bank']) if dataset=='forb32' else None
        maximum_native_error = 0.
        for row in manifest['rows']:
            hp = hidden_file(manifest,row)
            if str(hp) not in inspected_hidden:
                h = torch.load(hp,weights_only=True)
                assert h['id'] == row['id'] and h['sha256'] == row['sha256']
                assert h['signature'] == manifest.get('hidden_signature',manifest['signature'])
                assert h['projection_error'] == 0 and h['cached_embedding_max_error'] < 1e-6
                assert torch.isfinite(h['hidden']).all() and torch.isfinite(h['native']).all()
                count = int(h['mask'].sum())
                grid = h['grid'][0].tolist()
                assert count == grid[0]*(grid[1]//2)*(grid[2]//2)
                assert h['hidden'].shape == (len(h['mask']),3584)
                assert h['native'].shape == (len(h['mask']),128)
                if native_bank is not None:
                    actual = float((h['native'].float()-native_bank['embeddings'][row['query_index']].float()).abs().max())
                    assert actual < 1e-6, row['id']
                    maximum_native_error = max(maximum_native_error,actual)
                inspected_hidden[str(hp)] = (count,h['cached_embedding_max_error'])
            count,error = inspected_hidden[str(hp)]
            maximum_cache_error = max(maximum_cache_error,error)
            support = torch.load(WORK/dataset/'roma512'/(key(row)+'.pt'),weights_only=True)
            assert support['id']==row['id'] and support['candidates']==row['candidates']
            assert support['recipe']['manifest']==manifest['signature']
            assert support['signature']==signature(support['recipe']) and support['precision']==protocol['roma']['precision']
            assert support['recipe']['descriptor_cache']=='immutable cached features; copy list/dict containers before each consumer v2'
            assert len(support['pairs'])==len(row['candidates'])
            for pair in support['pairs']:
                assert 0 <= pair['M'] <= 1 and len(pair['u'])==count and pair['v'].ndim==1
                for value in (pair['u'],pair['v']):
                    assert torch.isfinite(value).all() and (value>=0).all() and (value<=1).all()
                mass = float((pair['u'].float().mean()*pair['v'].float().mean()).sqrt())
                assert abs(mass-pair['M']) < 5e-4
            feature = torch.load(WORK/dataset/'features'/(key(row)+'.pt'),weights_only=True)
            assert feature['id']==row['id'] and feature['signature']==manifest['signature']
            assert feature['candidates']==row['candidates'] and feature['support_signature']==support['signature']
            assert feature['values'].shape==(len(row['candidates']),8) and torch.isfinite(feature['values']).all()
            assert torch.equal(feature['raw'],torch.tensor(row['raw_scores']))
            assert torch.equal(feature['values'][:,1],torch.tensor([p['M'] for p in support['pairs']]))
            if dataset=='medicine' and row['split']=='train':
                assert set(row['candidates']) == set(manifest['train_references'])
            inventory[row['split']] += 1
        report['datasets'][dataset] = dict(queries=dict(inventory),references=len(manifest['references']),
              groups=len({r['group'] for r in manifest['rows']}),max_embedding_parity_error=maximum_cache_error,
              max_actual_native_error=maximum_native_error if native_bank is not None else None,
              evaluation_groups=len({evaluation_group(dataset,r) for r in manifest['rows']}),
              sha256=manifest['signature'])
        print('VERIFIED_CACHE',dataset,dict(inventory),flush=True)
    if args.stage=='final':
        selection = read(RESULTS/'selection.json')
        assert selection['experiment_protocol_sha256']==sha(ROOT/'configs/method/protocol.json')
        assert selection['data_bindings_sha256']==sha(RESULTS/'data_bindings.json')
        assert selection['baseline_selection_sha256']==sha(RESULTS/'baseline_selection.json')
        frozen = protocol['development_freeze']
        assert len(selection['runs'])==frozen['expected_runs']
        assert sha(ROOT/'configs/method/jobs.json')==frozen['gpu_jobs_sha256']
        assert sha(ROOT/'configs/method/head_jobs.json')==frozen['head_jobs_sha256']
        for name,run in selection['runs'].items():
            assert sha(run['checkpoint'])==run['checkpoint_sha256']
            ckpt = torch.load(project_path(run['checkpoint']),weights_only=True)
            assert ckpt['signature']==signature(ckpt['config'])==run['training_signature']
            config = ckpt['config']
            source = 'medicine_catalog' if name.startswith('medicine_catalog_') else 'medicine'
            m = read(WORK/source/'manifest.json')
            training_ids = set(config.get('train',config.get('training_rows',[])))
            assert training_ids and training_ids <= {r['id'] for r in m['rows'] if r['split']=='train'}
            assert config['manifest']==m['signature']
            first_training_row=next(r for r in m['rows'] if r['id'] in training_ids)
            feature=torch.load(WORK/source/'features'/(key(first_training_row)+'.pt'),weights_only=True)
            assert config['feature_recipe']==feature['support_signature']
            assert torch.isfinite(ckpt['theta']).all()
            state = ckpt.get('adapter',ckpt.get('model',ckpt.get('mlp',{})))
            assert all(torch.isfinite(t).all() for t in state.values())
            errors = [h['replay_error'] for h in ckpt.get('history',[]) if 'replay_error' in h]
            assert all(e < 2e-6 if 'model' in ckpt else e <= 2e-5 for e in errors)
            report['runs'][name] = dict(training_queries=len(training_ids),checkpoint_sha256=run['checkpoint_sha256'])
        for device in ('cpu','gpu'):
            for job in read(ROOT/'configs/method'/f'evaluation_{device}_jobs.json'):
                state = read(WORK/'intermediate/jobs'/(job['id']+'.json'))
                assert state['status']=='complete' and state['returncode']==0,job['id']
                assert state['job']==job,job['id']
        for path in (RESULTS/'evaluation').glob('*/*/*/complete.json'):
            complete = read(path);cfg=complete['config']
            manifest = read(WORK/cfg['dataset']/'manifest.json')
            rows = [r for r in manifest['rows'] if r['split']==cfg['split']]
            labels = read(WORK/cfg['dataset']/'labels.json')
            full_ranks = read(WORK/cfg['dataset']/'full_ranks.json')['positive_ranks']
            assert cfg['labels_sha256']==sha(WORK/cfg['dataset']/'labels.json')
            preds = read(path.parent/'predictions.json')
            assert [p['id'] for p in preds]==[r['id'] for r in rows]
            parts = list((path.parent/'shards').glob('*.json'))
            if parts:
                covered = []
                for part in sorted((read(p) for p in parts),key=lambda p:p['query_start']):
                    assert part['signature']==complete['signature'] and part['total_queries']==len(rows)
                    expected = [r['id'] for r in rows[part['query_start']:part['query_stop']]]
                    assert part['ids']==expected
                    covered.extend(part['ids'])
                assert covered==[r['id'] for r in rows], 'Missing or overlapping query shards'
            for row,pred in zip(rows,preds,strict=True):
                assert pred['group']==evaluation_group(cfg['dataset'],row)
                rank = restored_rank(pred['raw_rank'],row['candidates'][:cfg['k']],set(labels[row['id']]),pred['permutation'])
                assert pred['rank']==rank
                candidates = row['candidates'][:cfg['k']]
                reranked = {str(candidates[p]):i+1 for i,p in enumerate(pred['permutation'])}
                expected = {p:reranked.get(p,r) for p,r in full_ranks[row['id']].items()}
                assert pred['positive_ranks']==expected
            actual = summarize(preds,False)
            for averaging in ('micro','macro'):
                for name,value in actual[averaging].items():
                    assert np.isclose(value,complete['metrics'][averaging][name],rtol=0,atol=1e-12)
            report['evaluations'] += 1
        report['baseline_results'] = 0
        for dataset in ('medicine','grozi','milk','forb32'):
            splits = ('val','test') if dataset=='medicine' else ('transfer',)
            for split in splits:
                for result in read(RESULTS/'baselines'/dataset/f'{split}.json').values():
                    assert sha(result['source'])==result['source_sha256']
                    assert sha(Path(result['source']).parent/'config.json')==result['config_sha256']
                    if result['kind']=='mixed_region_diagnostic':
                        region = result['region_provenance']
                        assert sha(region['source'])==region['source_sha256']
                        assert region['full_counts']=={'human':200,'sam3':800}
                        expected = {'val':{'human':22,'sam3':125},'test':{'human':58,'sam3':89}}
                        assert region['split_counts']==expected[split]
                    actual=summarize(result['predictions'],False)
                    for averaging in ('micro','macro'):
                        for metric,value in actual[averaging].items():
                            assert np.isclose(value,result['metrics'][averaging][metric],rtol=0,atol=1e-12)
                    report['baseline_results'] += 1
        timed = [selection['primary']['strict'],selection['selected']['strict/external'],
                 selection['selected']['strict/reader']]
        hardware = queries = None
        for name in timed:
            result = read(RESULTS/'timing'/name/'complete.json')
            assert result['checkpoint_sha256']==selection['runs'][name]['checkpoint_sha256']
            assert result['queries']==protocol['online_timing']['queries']==len(result['records'])
            assert result['k']==selection['runs'][name]['candidate_k']
            current_hardware = (result['host'],result['visible_devices'],result['gpus'])
            current_queries = [r['id'] for r in result['records']]
            if hardware is None:
                hardware,queries = current_hardware,current_queries
            assert current_hardware==hardware and current_queries==queries
            for record in result['records']:
                assert record['score_parity_error']<1e-5
                stages = ('encoder','full_gallery_search','roma','feature_pooling','learned_reranking')
                assert all(np.isfinite(record[s]) and record[s]>=0 for s in stages)
                assert np.isclose(sum(record[s] for s in stages),record['end_to_end'],rtol=0,atol=1e-8)
        report['online_timing_methods'] = timed
        fusion_settings=read(RESULTS/'fusion_selection.json')
        for setting in fusion_settings.values():
            assert setting['model_selection_sha256']==sha(RESULTS/'selection.json')
        for split in ('val','test'):
            fusion=read(RESULTS/'fusion'/f'{split}.json')
            assert fusion['selection_sha256']==sha(RESULTS/'fusion_selection.json')
            for result in fusion['methods'].values():
                actual=summarize(result['predictions'],False)
                assert actual['micro']==result['metrics']['micro']
                assert actual['macro']==result['metrics']['macro']
    report.update(status='PASS',finished=time.time(),seconds=time.time()-started)
    suffix = '_'.join(args.datasets) if args.datasets else ''
    save(RESULTS/f"verification_{args.stage}{'_'+suffix if suffix else ''}.json",report)
    print('VERIFIED',report['evaluations'],'evaluations',flush=True)


if __name__=='__main__':
    main()
