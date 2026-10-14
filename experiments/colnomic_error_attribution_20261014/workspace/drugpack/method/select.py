import argparse
import time

from .common import RUNS, RESULTS, ROOT, read, save, sha, signature


def family(name):
    name = name.removeprefix('medicine_catalog_')
    if name.startswith('external_'):
        return 'external'
    if name.startswith('reader_'):
        return 'reader'
    if name.startswith('patch_post_real_'):
        return 'patch_hidden'
    if name.startswith('embedding_post_real_'):
        return 'embedding'
    if name.startswith('post_real_') and '_b100_d16' in name:
        return 'hidden'
    return 'ablation'


def run_name(job):
    arguments = [a.replace('drugpack.method.', 'rebut.train.') for a in job['arguments']]
    options = dict(zip(arguments[2::2], arguments[3::2]))
    dataset = options.get('--dataset', 'medicine')
    if arguments[1] == 'rebut.train.heads':
        name = f"external_{options['--kind']}_{options.get('--loss','cost')}_b{options.get('--budget','100')}"
    elif arguments[1] == 'rebut.train.scalar_head':
        name = f"external_mlp_w{options['--width']}_{options.get('--loss','cost')}_s{options.get('--seed','17')}"
    elif arguments[1] == 'rebut.train.reader':
        name = f"reader_{options.get('--arm','SPATIAL')}_s{options.get('--seed','17')}"
    else:
        name = (f"post_{options.get('--arm','real')}_{options.get('--loss','cost')}"
                f"_s{options.get('--seed','17')}_b{options.get('--budget','100')}"
                f"_d{options.get('--bottleneck','16')}")
        if options.get('--location') == 'embedding':
            name = 'embedding_' + name
        if options.get('--condition') == 'patch':
            name = 'patch_' + name
    return (dataset + '_' if dataset != 'medicine' else '') + name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    expected = [run_name(r) for r in read(ROOT/'configs/method/jobs.json')]
    expected += [run_name(r) for r in read(ROOT/'configs/method/head_jobs.json')]
    expected += [prefix + f'external_{kind}_{loss}_b100'
                 for prefix in ('', 'medicine_catalog_')
                 for kind in ('INTERNAL3','RAW_M','ADDITIVE4','PRODUCT5','WEIGHTED7')
                 for loss in ('cost','ce')]
    destination = RESULTS/'selection.json'
    if destination.exists():
        frozen = read(destination)
        assert sorted(frozen['runs']) == sorted(expected)
        for run in frozen['runs'].values():
            assert sha(run['checkpoint']) == run['checkpoint_sha256']
        return
    while True:
        missing = [r for r in expected if not (RUNS/r/'complete.json').exists()]
        if not missing:
            break
        if not args.wait:
            raise RuntimeError(f'Training unfinished: {missing}')
        time.sleep(30)
    runs, selected = {}, {}
    for name in expected:
        folder = RUNS/name
        complete = read(folder/'complete.json')
        kind = family(name)
        if (folder/'checkpoint.pt').exists():
            checkpoint = folder/'checkpoint.pt'
            validation = complete['validation']
        else:
            checkpoint = folder/'best.pt'
            validation = {k:v['metrics'] for k,v in read(folder/'best_validation.json').items()}
        k = max(validation, key=lambda k:(validation[k]['macro']['R@1'],
                                         validation[k]['macro']['MRR'], -int(k)))
        metric = validation[k]['macro']
        regime = 'catalog' if name.startswith('medicine_catalog_') else 'strict'
        runs[name] = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                          training_signature=complete['signature'],regime=regime,family=kind,
                          candidate_k=int(k),validation_macro=metric)
        if kind != 'ablation':
            group = regime + '/' + kind
            previous = selected.get(group)
            if previous is None or (metric['R@1'],metric['MRR']) > (
                    runs[previous]['validation_macro']['R@1'],runs[previous]['validation_macro']['MRR']):
                selected[group] = name
    primary = {}
    for regime in ('strict','catalog'):
        candidates = [name for role,name in selected.items() if role.split('/')[0]==regime
                      and role.split('/')[1] in ('hidden','patch_hidden','embedding')]
        primary[regime] = max(candidates,key=lambda n:(runs[n]['validation_macro']['R@1'],runs[n]['validation_macro']['MRR']))
    result = dict(created=time.time(),policy='Medicine validation macro R@1 then macro MRR; ties prefer smaller K and declared run order.',
                  historical_exposure='Previously inspected baselines; no historically blind-test claim.',
                  transfer='Same checkpoint, normalization, and head; no adaptation or selection on external data.',
                  forb_candidate_budget=32,runs=runs,selected=selected,primary=primary,
                  experiment_protocol_sha256=sha(ROOT/'configs/method/protocol.json'),
                  data_bindings_sha256=sha(RESULTS/'data_bindings.json'),
                  baseline_selection_sha256=sha(RESULTS/'baseline_selection.json'))
    result['signature'] = signature(result)
    save(destination,result)
    print(selected,flush=True)


if __name__ == '__main__':
    main()
