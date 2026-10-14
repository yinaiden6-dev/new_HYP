from .common import ROOT, RESULTS, WORK, read, save, signature


def query_ranges(count, size=1024):
    return [(start,min(start+size,count)) for start in range(0,count,size)]


def main():
    selection = read(RESULTS/'selection.json')
    jobs = {'gpu':{},'cpu':{}}
    def add(dataset, split, run, k=None, intervention=None):
        device = 'cpu' if run in ('RAW','L0','M','weighted_L','S','U','V') or 'external_' in run else 'gpu'
        selected_k = selection['runs'].get(run,{}).get('candidate_k',128)
        k = 32 if dataset == 'forb32' else k or selected_k
        spec = dict(dataset=dataset,split=split,run=run,k=k,intervention=intervention)
        token = 'eval_' + signature(spec)[:20]
        args = ['-m','drugpack.method.evaluate','--dataset',dataset,'--split',split,'--run',run,'--k',str(k),
                '--device','cpu' if device=='cpu' else 'cuda:0']
        if intervention:
            args += ['--intervention',intervention]
        if dataset == 'forb32' and device == 'gpu':
            count = sum(r['split']==split for r in read(WORK/dataset/'manifest.json')['rows'])
            name = f'{run}_k{k}' + ('_'+intervention if intervention else '')
            shards = RESULTS/'evaluation'/dataset/split/name/'shards'
            dependencies = []
            for start,stop in query_ranges(count):
                part = token + f'_q{start}_{stop}'
                jobs['gpu'][part] = dict(id=part,requires={dataset:[split]},query_range=[start,stop],
                    arguments=args+['--query-start',str(start),'--query-stop',str(stop)],
                    description=dict(spec,query_range=[start,stop]))
                dependencies.append(str((shards/f'{start}_{stop}.json').relative_to(ROOT)))
            aggregate = args.copy()
            aggregate[aggregate.index('--device')+1] = 'cpu'
            jobs['cpu'][token+'_aggregate'] = dict(id=token+'_aggregate',requires_files=dependencies,
                arguments=aggregate+['--aggregate-only'],description=dict(spec,aggregate=True))
        else:
            jobs[device][token] = dict(id=token,requires={} if run=='RAW' else {dataset:[split]},
                                      arguments=args,description=spec)
    for run in selection['runs']:
        dataset = 'medicine_catalog' if run.startswith('medicine_catalog_') else 'medicine'
        add(dataset,'test',run)
        if 'reader_SET_' in run or 'reader_SPATIAL_' in run:
            for intervention in ('shuffle','rotate'):
                add(dataset,'test',run,intervention=intervention)
    for dataset in ('medicine','grozi','milk','forb32'):
        split = 'test' if dataset=='medicine' else 'transfer'
        for name in ('RAW','L0','M','weighted_L','S','U','V'):
            add(dataset,split,name)
    for role,run in selection['selected'].items():
        regime,family = role.split('/')
        medicine = 'medicine_catalog' if regime=='catalog' else 'medicine'
        for dataset in ('grozi','milk','forb32'):
            add(dataset,'transfer',run)
        for k in (8,16,32,64,128):
            add(medicine,'test',run,k)
        if family in ('hidden','patch_hidden','embedding','external'):
            for intervention in ('constant','shuffled'):
                add(medicine,'test',run,intervention=intervention)
        elif family=='reader' and 'reader_COMPRESSED_' not in run:
            for intervention in ('shuffle','rotate'):
                add(medicine,'test',run,intervention=intervention)
    for device,values in jobs.items():
        save(ROOT/'configs/method'/f'evaluation_{device}_jobs.json',list(values.values()))
        print(device,len(values),flush=True)


if __name__=='__main__':
    main()
