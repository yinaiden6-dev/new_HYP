#!/usr/bin/env python3
"""Frozen image-image reranking of natural H593 C128, then grouped calibration.

GPU workers never read identities. All 128 logits, original sigmoid outputs,
image grids and timings are retained. Fold fits read only their TRAIN labels.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[2]
OUT = ROOT / 'results/rc_h593_qwen3_rerank_v1'
AUTH = ROOT / 'registry/rc_h593_qwen3_rerank_authority_v1_20260923.json'
MODEL = WORK / 'models/downloaded_models/Qwen3-VL-Reranker-2B'
INSTRUCTION = ('Retrieve the exact same medicine product as shown in the query image. '
               'Distinguish visually similar but different products, strengths, and package variants.')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bind(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))


def checked(b):
    assert sha(b['path']) == b['sha256'], ('SHA_DRIFT', b['path'])
    return Path(b['path'])


def write(p, obj, immutable=True):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if p.exists() and immutable:
        assert p.read_text() == value, ('IMMUTABLE', str(p))
        return
    tmp = p.with_name('.' + p.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def prepare():
    assert not AUTH.exists(), 'NEW_AUTHORITY_REQUIRED'
    parent_path = ROOT / 'registry/rc_h593_learned_colnomic_only_authority_v1_20260920.json'
    parent = read(parent_path)
    images_path = ROOT / 'results/rc_colpali_h593_query_tokens_v4_legacy/manifest.json'
    images = {r['query_id']: r for r in read(images_path)['records']}
    rows, sources = [], []
    # Reuse independently verified original RAW scores and axes, no new encoder pass.
    for shard in range(75):
        d = ROOT / f'results/rc_h593_learned_colnomic_only_v1/features/shard{shard:02d}'
        v = read(d / 'validation.json')
        assert v['status'] == 'CONTENT_ALL_PAIRS_NUMPY_PASS'
        p = read(checked(v['payload']))
        assert p['authority'] == v['authority'] == bind(parent_path)
        sources.append(dict(payload=v['payload'], validation=bind(d / 'validation.json')))
        for r in p['records']:
            im = images[r['query_id']]
            assert r['execution_ordinal'] == im['ordinal']
            assert r['source_image_sha256'] == im['image_sha256'] == sha(im['image_path'])
            axis = r['candidate_physical_rows']
            raw = r['raw_scores']
            assert len(axis) == len(set(axis)) == len(raw) == 128 and axis == sorted(axis)
            assert all(math.isfinite(x) for x in raw)
            winner = max(range(128), key=lambda i: (raw[i], -axis[i]))
            assert winner == r['winner'] and axis[winner] == r['raw_ranked_physical_rows'][0]
            assert set(axis) == set(r['raw_ranked_physical_rows'][:128])
            rows.append(dict(query_id=r['query_id'], execution_ordinal=r['execution_ordinal'],
                             image_path=im['image_path'], image_sha256=im['image_sha256'],
                             candidate_physical_rows=axis, raw_scores=raw, winner=winner,
                             challenger_positions=r['challenger_positions']))
    rows.sort(key=lambda r: r['execution_ordinal'])
    assert len(rows) == len({r['query_id'] for r in rows}) == 593
    assert [r['execution_ordinal'] for r in rows] == list(range(593))
    used = {i for r in rows for i in r['candidate_physical_rows']}
    gallery = read(checked(parent['gallery']))
    refs = {str(r['physical_row']): dict(image_path=r['image_path'], image_sha256=sha(r['image_path']))
            for r in gallery['records'] if r['physical_row'] in used}
    assert len(refs) == len(used)
    write(OUT / 'workers.json', dict(records=rows, references=refs, labels_included=False,
                                    image_manifest=bind(images_path), raw_sources=sources))
    models = [MODEL / 'model.safetensors'] + sorted(MODEL.glob('*.json'))
    models += [MODEL / 'chat_template.jinja', MODEL / 'merges.txt',
               MODEL / 'scripts/qwen3_vl_reranker.py']
    model_files = [dict(**bind(p), bytes=p.stat().st_size) for p in models]
    sources = [Path(__file__), ROOT / 'slurm/rc_h593_qwen3_rerank_v1.sbatch',
               ROOT / 'slurm/rc_h593_qwen3_rerank_cpu_v1.sbatch',
               ROOT / 'plan/RC_H593_QWEN3_RERANK_V1_20260923.md']
    write(AUTH, dict(status='H593_QWEN3_RERANK_AUTHORIZED', workers=bind(OUT / 'workers.json'),
                     sources=[bind(p) for p in sources], parent=bind(parent_path),
                     model=str(MODEL), model_files=model_files, instruction=INSTRUCTION,
                     query_count=593, candidates=128, gpu_pairs=593*128, shards=50,
                     dtype='bfloat16', attention='sdpa', max_length=8192,
                     min_pixels=4*32*32, max_pixels=1280*32*32,
                     image_decode='PIL RGB, same decoded raw frame; no additional EXIF transform',
                     ranking='Descending pre-sigmoid yes-minus-no logit; ties use RAW rank then physical row',
                     sigmoid_control='Official BF16 sigmoid score; same deterministic tie break',
                     OCR=False, task_finetuning=False, target_injection=False,
                     split=parent['split'], folds=parent['folds'], gallery=parent['gallery'],
                     curator=parent['curator'], old_result=parent['old_result'],
                     calibration=dict(features=['RAW standardized challenger-minus-winner',
                                                'reranker logit standardized challenger-minus-winner'],
                                      parameters=3, losses=['COST1','CE'], steps=2000, seed=17,
                                      dtype='float64', optimizer='AdamW', lr=.03, weight_decay=.001,
                                      train_order='Original recall-present TRAIN order, exact original folds'),
                     evidence='Opened H593 grouped OOF5 comparison, not external confirmation'))
    write(OUT / 'preflight.json', dict(status='MANIFEST_AXIS_IMAGE_SHA_PASS', authority=bind(AUTH),
                                     queries=593, candidates_per_query=128, image_pairs=75904,
                                     heldout_label_reads=0, source_shards=75))
    print(dict(status='PREPARED', authority=str(AUTH), pairs=75904), flush=True)


def guard(stage, fold=None):
    a = read(AUTH)
    for b in a['sources']:
        checked(b)
    assert read(OUT / 'preflight.json')['authority'] == bind(AUTH)
    if stage != 'validate-inputs':
        assert os.environ.get('SLURM_JOB_ID'), 'SLURM_REQUIRED'
    # Prevent accidental query-identity access during GPU inference and held-label access in fit.
    allowed_results = {Path(a['workers']['path']).resolve()}
    if stage == 'fit':
        allowed_results.update(Path(a[k]['path']).resolve() for k in ('gallery','split'))
        allowed_results.add(Path(a['folds'][str(fold)]['train_roles']['path']).resolve())
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str,bytes,os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if stage in ('score','fit') and p.suffix == '.json' and ROOT/'results' in p.parents:
            assert OUT in p.parents or p in allowed_results, ('UNAUTHORIZED_RESULT_READ', str(p))
    sys.addaudithook(audit)
    return a, read(checked(a['workers']))


def validate_query(a, row):
    d = OUT / f"queries/query{row['execution_ordinal']:03d}"
    p = read(d / 'scores.json')
    assert p['authority'] == bind(AUTH) and p['query_id'] == row['query_id']
    assert p['candidate_physical_rows'] == row['candidate_physical_rows']
    assert len(p['pairs']) == 128
    for i, v in enumerate(p['pairs']):
        assert v['physical_row'] == row['candidate_physical_rows'][i] and v['position'] == i
        assert math.isfinite(v['logit']) and math.isfinite(v['sigmoid']) and 0 <= v['sigmoid'] <= 1
        assert len(v['image_grid_thw']) == 2 and all(min(g) > 0 for g in v['image_grid_thw'])
        assert v['image_tokens'] == sum(math.prod(g)//4 for g in v['image_grid_thw'])
        assert 0 < v['image_tokens'] < v['sequence_tokens'] <= a['max_length']
    v = dict(status='QWEN3_ALL128_SCORES_PASS', authority=bind(AUTH), scores=bind(d/'scores.json'),
             query_id=row['query_id'], pairs=128, labels_read=0)
    write(d / 'validation.json', v)
    return p


def load_model(a):
    import torch
    import transformers
    sys.path.insert(0, str(MODEL/'scripts'))
    import qwen3_vl_reranker as official
    for b in a['model_files']:
        assert Path(b['path']).stat().st_size == b['bytes']
        if not b['path'].endswith('.safetensors'):
            checked(b)
    # Hash weights once in the pilot; later workers use the sealed digest and file size.
    if not (OUT/'pilot_validation.json').exists():
        for b in a['model_files']:
            if b['path'].endswith('.safetensors'):
                checked(b)
    assert torch.cuda.is_available(), 'GPU_REQUIRED'
    torch.set_num_threads(8)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    loader = official.Qwen3VLForConditionalGeneration.from_pretrained
    loading = {}
    def strict_load(*args, **kwargs):
        model, info = loader(*args, **kwargs, output_loading_info=True, local_files_only=True)
        for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            assert not info.get(key), (key, info.get(key))
        # Transformers 5 returns sets for some loader diagnostics.
        loading.update(json.loads(json.dumps(info, default=lambda v: sorted(v) if isinstance(v,set) else str(v))))
        return model
    official.Qwen3VLForConditionalGeneration.from_pretrained = staticmethod(strict_load)
    try:
        model = official.Qwen3VLReranker(model_name_or_path=str(MODEL), torch_dtype=torch.bfloat16,
                                        attn_implementation=a['attention'], max_length=a['max_length'],
                                        min_pixels=a['min_pixels'], max_pixels=a['max_pixels'])
    finally:
        official.Qwen3VLForConditionalGeneration.from_pretrained = loader
    return model, dict(torch=torch.__version__, transformers=transformers.__version__,
                       device=torch.cuda.get_device_name(), loading=loading,
                       dtype=str(model.model.dtype), official_wrapper=bind(MODEL/'scripts/qwen3_vl_reranker.py'))


def score(a, inputs, shard, pilot, budget):
    start = time.monotonic()
    d = OUT/'execution'/('pilot' if pilot else f'shard{shard:02d}')
    d.mkdir(parents=True, exist_ok=True)
    lock = (d/'worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = inputs['records'][:1] if pilot else [r for r in inputs['records'] if r['execution_ordinal'] % 50 == shard]
    if not pilot:
        pv = read(OUT/'pilot_validation.json')
        assert pv['authority'] == bind(AUTH) and pv['status'] == 'QWEN3_PILOT_ALL128_PASS'
    for r in rows:
        if (OUT/f"queries/query{r['execution_ordinal']:03d}/scores.json").exists():
            validate_query(a, r)
    pending = [r for r in rows if not (OUT/f"queries/query{r['execution_ordinal']:03d}/validation.json").exists()]
    if pending:
        import torch
        from PIL import Image
        model, runtime = load_model(a)
        write(d/f"load_{os.environ['SLURM_JOB_ID']}_{os.environ.get('SLURM_RESTART_COUNT','0')}.json", runtime)
        stop = [False]
        signal.signal(signal.SIGTERM, lambda *args: stop.__setitem__(0, True))
        checked_images = set()
        for row in pending:
            qd = OUT/f"queries/query{row['execution_ordinal']:03d}"
            partial = qd/'partial.json'
            obj = read(partial) if partial.exists() else dict(authority=bind(AUTH), query_id=row['query_id'],
                         candidate_physical_rows=row['candidate_physical_rows'], pairs=[], label_reads=0)
            assert obj['authority'] == bind(AUTH) and obj['query_id'] == row['query_id']
            assert obj['candidate_physical_rows'] == row['candidate_physical_rows']
            assert sha(row['image_path']) == row['image_sha256']
            with Image.open(row['image_path']) as im:
                query = im.convert('RGB')
            for pos in range(len(obj['pairs']), 128):
                if stop[0] or time.monotonic()-start > budget:
                    write(partial, obj, immutable=False)
                    print(dict(event='CHECKPOINT', query=row['query_id'], pairs=len(obj['pairs'])), flush=True)
                    return 75
                physical = row['candidate_physical_rows'][pos]
                ref = inputs['references'][str(physical)]
                if physical not in checked_images:
                    assert sha(ref['image_path']) == ref['image_sha256']
                    checked_images.add(physical)
                with Image.open(ref['image_path']) as im:
                    document = im.convert('RGB')
                pair = model.format_mm_instruction(None, query, None, None, document, None,
                                                   instruction=a['instruction'])
                # Official tokenizer can catch image errors and substitute NULL; forbid that fallback.
                ts = time.monotonic()
                batch = model.tokenize([pair])
                for key, value in list(batch.items()):
                    if not isinstance(value, torch.Tensor):
                        batch[key] = torch.as_tensor(value)
                grid = batch['image_grid_thw'].tolist()
                assert len(grid) == 2 and 'pixel_values' in batch, 'TWO_ACTUAL_IMAGES_REQUIRED'
                assert batch['input_ids'].shape[1] <= a['max_length'], 'NO_TRUNCATION_ALLOWED'
                nimage = int((batch['input_ids'] == model.model.config.image_token_id).sum())
                assert nimage == sum(math.prod(g)//4 for g in grid), 'NO_IMAGE_TOKEN_TRUNCATION'
                batch = batch.to(model.model.device)
                with torch.inference_mode():
                    hidden = model.model(**batch).last_hidden_state[:, -1]
                    logit = model.score_linear(hidden).squeeze(-1)
                    probability = torch.sigmoid(logit)
                    assert torch.isfinite(logit).all()
                    # Compare against official inference once, before releasing the array.
                    if pilot and pos == 0:
                        official_scores = model.compute_scores(batch)
                        assert official_scores == probability.cpu().tolist(), 'OFFICIAL_SCORE_PARITY'
                        parity = dict(authority=bind(AUTH), status='OFFICIAL_SCORE_EXACT_PARITY',
                                      logit=float(logit[0]), sigmoid=float(probability[0]), grid=grid)
                        write(OUT/'official_score_parity.json', parity)
                obj['pairs'].append(dict(position=pos, physical_row=physical, logit=float(logit[0]),
                    sigmoid=float(probability[0]), image_grid_thw=grid, image_tokens=nimage,
                    sequence_tokens=int(batch['input_ids'].shape[1]), seconds=time.monotonic()-ts,
                    job_id=os.environ['SLURM_JOB_ID']))
                if len(obj['pairs']) % 4 == 0:
                    write(partial, obj, immutable=False)
                if len(obj['pairs']) % 32 == 0:
                    print(dict(event='PAIR_PROGRESS', query=row['query_id'], pairs=len(obj['pairs']),
                               elapsed=time.monotonic()-start), flush=True)
            write(qd/'scores.json', obj)
            validate_query(a, row)
            print(dict(event='QUERY128_COMPLETE', query=row['query_id']), flush=True)
    validations = [bind(OUT/f"queries/query{r['execution_ordinal']:03d}/validation.json") for r in rows]
    dest = OUT/'pilot_validation.json' if pilot else OUT/f'shards/{shard:02d}/validation.json'
    write(dest, dict(status='QWEN3_PILOT_ALL128_PASS' if pilot else 'QWEN3_SHARD_PASS',
                     authority=bind(AUTH), validations=validations, queries=len(rows)))
    return 0


def all_scores(a, inputs):
    for shard in range(50):
        v = read(OUT/f'shards/{shard:02d}/validation.json')
        assert v['status'] == 'QWEN3_SHARD_PASS' and v['authority'] == bind(AUTH)
        expected = [r for r in inputs['records'] if r['execution_ordinal'] % 50 == shard]
        assert len(v['validations']) == len(expected)
        for b in v['validations']:
            checked(b)
    scores = {}
    for row in inputs['records']:
        d = OUT/f"queries/query{row['execution_ordinal']:03d}"
        v = read(d/'validation.json')
        assert v['status'] == 'QWEN3_ALL128_SCORES_PASS' and v['authority'] == bind(AUTH)
        p = read(checked(v['scores']))
        assert p['query_id'] == row['query_id'] and p['candidate_physical_rows'] == row['candidate_physical_rows']
        scores[row['query_id']] = p
    return scores


def make_x(row, scores):
    import numpy as np
    columns = []
    for values in (row['raw_scores'], [p['logit'] for p in scores['pairs']]):
        v = np.asarray(values, dtype=np.float64)
        columns.append((v[row['challenger_positions']]-v[row['winner']])/max(float(v.std()), 1e-12))
    return np.stack(columns, axis=-1)


def loss(z, y, name):
    import torch
    from torch.nn import functional as F
    if name == 'CE':
        a = torch.cat((z.new_zeros((len(z), 1)), z), 1)
        return (torch.logsumexp(a, 1)-a[torch.arange(len(y)), y+1]).mean()
    raw = y == -1
    ii = torch.arange(len(y))
    index = y.clamp_min(0)
    wrong = z.clone()
    wrong[ii[~raw], index[~raw]] = -torch.inf
    return torch.where(raw, F.softplus(z.amax(1)),
                       F.softplus(-z[ii,index])+F.softplus(wrong.amax(1))).mean()


def self_test():
    import numpy as np
    import torch
    sys.path.insert(0, str(ROOT/'programs'))
    import run_rc_six_cause_loss_binding_v1 as legacy
    z = torch.tensor([[0.,0.,0.],[.2,-.1,.3],[-.4,.3,-.1]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([-1,0,2])
    for name in ('COST1','CE'):
        v = loss(z,y,name)
        old = legacy.unit(z,y) if name == 'COST1' else legacy.N.R.objective(z,y+1)
        assert torch.equal(v,old)
        assert torch.equal(torch.autograd.grad(v,z,retain_graph=True)[0],
                           torch.autograd.grad(old,z,retain_graph=True)[0])
    row = dict(raw_scores=[1.,2.,3.], winner=2, challenger_positions=[0,1])
    p = dict(pairs=[dict(logit=x) for x in (-2.,0.,2.)])
    x = make_x(row,p)
    assert np.isfinite(x).all() and x.shape == (2,2)
    np.testing.assert_allclose(x[:,0], x[:,1], atol=1e-14)
    assert np.array_equal(make_x(row,dict(pairs=[dict(logit=0.)]*3))[:,1], [0.,0.])
    print(dict(status='LOSS_VALUE_GRADIENT_AND_FEATURE_PREFLIGHT_PASS'),flush=True)


def fit(a, inputs, fold, budget):
    import numpy as np
    import torch
    torch.set_num_threads(8)
    start = time.monotonic()
    d = OUT/f'fold{fold}'
    d.mkdir(parents=True, exist_ok=True)
    lock = (d/'fit.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    self_test()
    scores = all_scores(a,inputs)
    b = a['folds'][str(fold)]
    roles = {r['query_id']: r for r in read(checked(b['train_roles']))['records']}
    fm = read(checked(a['split']))['folds'][fold]
    labels = {r['physical_row']: r['identity'] for r in read(checked(a['gallery']))['records']}
    train = [r for r in inputs['records'] if r['query_id'] in roles]
    held = [r for r in inputs['records'] if r['query_id'] in set(fm['heldout_query_ids'])]
    assert [r['query_id'] for r in train] == b['train_query_ids']
    assert set(roles) == set(fm['train_query_ids']) and not set(roles) & set(fm['heldout_query_ids'])
    kept, ys = [], []
    for r in train:
        positions = [i for i,p in enumerate(r['candidate_physical_rows']) if labels[p] == roles[r['query_id']]['identity']]
        assert len(positions) <= 1
        if positions:
            kept.append(r)
            ys.append(-1 if positions[0] == r['winner'] else r['challenger_positions'].index(positions[0]))
    assert [r['query_id'] for r in kept] == b['effective_train_query_ids']
    x = torch.tensor(np.stack([make_x(r,scores[r['query_id']]) for r in kept]), dtype=torch.float64)
    h = np.stack([make_x(r,scores[r['query_id']]) for r in held])
    y = torch.tensor(ys)
    params = {}
    for name in ('COST1','CE'):
        pp = d/f'{name}_parameters.json'
        if pp.exists():
            v = read(pp)
            assert v['authority'] == bind(AUTH) and v['steps'] == 2000
            params[name] = v['theta_hex']
            continue
        cp = d/f'{name}_checkpoint.pt'
        theta = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        opt = torch.optim.AdamW([theta],lr=.03,weight_decay=.001)
        step = 0
        if cp.exists():
            state = torch.load(cp,weights_only=True,map_location='cpu')
            assert state['authority'] == bind(AUTH)
            with torch.no_grad(): theta.copy_(state['theta'])
            opt.load_state_dict(state['optimizer'])
            step = state['step']
        while step < 2000:
            opt.zero_grad()
            v = loss(x@theta[:2]+theta[2],y,name)
            assert torch.isfinite(v)
            v.backward()
            opt.step()
            step += 1
            if step % 100 == 0 or step == 2000:
                tmp = cp.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    torch.save(dict(authority=bind(AUTH),theta=theta.detach(),optimizer=opt.state_dict(),step=step),f)
                    f.flush(); os.fsync(f.fileno())
                os.replace(tmp,cp)
                print(dict(event='FIT_PROGRESS',fold=fold,loss=name,step=step),flush=True)
                if time.monotonic()-start > budget and step < 2000: return 75
        params[name] = [float(v).hex() for v in theta.detach()]
        write(pp,dict(authority=bind(AUTH),theta_hex=params[name],steps=2000))
    predictions = []
    maximum = 0.
    for i,r in enumerate(held):
        models = {}
        for name,theta in params.items():
            t = np.array([float.fromhex(v) for v in theta])
            z = h[i]@t[:2]+t[2]
            independent = (torch.from_numpy(h[i])@torch.from_numpy(t[:2])+t[2]).numpy()
            error = float(np.max(np.abs(z-independent)))
            assert error < 2e-10
            maximum = max(maximum,error)
            j = int(z.argmax())
            pos = r['challenger_positions'][j] if z[j] > 0 else r['winner']
            assert (int(independent.argmax()), bool(independent.max()>0)) == (j,bool(z.max()>0))
            models[name] = dict(selected=r['candidate_physical_rows'][pos],logits_hex=[float(v).hex() for v in z])
        predictions.append(dict(query_id=r['query_id'],models=models))
    write(d/'payload.json',dict(authority=bind(AUTH),fold=fold,parameters=params,predictions=predictions,
                               train_query_ids=b['train_query_ids'],effective_train_query_ids=b['effective_train_query_ids'],
                               heldout_label_reads=0))
    write(d/'validation.json',dict(status='QWEN3_CALIBRATION_NUMPY_TORCH_PASS',authority=bind(AUTH),
                                  payload=bind(d/'payload.json'),max_error=maximum,fold=fold))
    return 0


def join(a, inputs):
    import numpy as np
    scores = all_scores(a,inputs)
    preds, old, seals = {}, {}, []
    for fold in range(5):
        v = read(OUT/f'fold{fold}/validation.json')
        assert v['status'] == 'QWEN3_CALIBRATION_NUMPY_TORCH_PASS' and v['authority'] == bind(AUTH)
        p = read(checked(v['payload']))
        assert p['fold'] == fold and p['authority'] == bind(AUTH)
        fm = read(checked(a['split']))['folds'][fold]
        assert {r['query_id'] for r in p['predictions']} == set(fm['heldout_query_ids'])
        assert not set(preds) & set(fm['heldout_query_ids'])
        preds.update({r['query_id']: dict(fold=fold, **r) for r in p['predictions']})
        b = a['folds'][str(fold)]
        ov = read(checked(b['old_validation']))
        assert ov['payload'] == b['old_payload']
        old.update({r['query_id']: r for r in read(checked(b['old_payload']))['predictions']})
        seals.append(bind(OUT/f'fold{fold}/validation.json'))
    write(OUT/'predictions_prelabel_seal.json',dict(authority=bind(AUTH),fold_validations=seals,
                                                  shards=[bind(OUT/f'shards/{s:02d}/validation.json') for s in range(50)]))
    roles = {r['query_id']: r for r in read(checked(a['curator']))['records']}
    labels = {r['physical_row']:r['identity'] for r in read(checked(a['gallery']))['records']}
    rows = []
    for r in inputs['records']:
        qid = r['query_id']; role = roles[qid]; axis = r['candidate_physical_rows']
        assert preds[qid]['fold'] == role['outer_fold']
        train = read(checked(a['folds'][str(role['outer_fold'])]['train_roles']))['records']
        assert role['identity'] not in {t['identity'] for t in train}
        assert role['component'] not in {t['component'] for t in train}
        raw_order = sorted(range(128),key=lambda i:(-r['raw_scores'][i],axis[i]))
        ranks = {i:j for j,i in enumerate(raw_order)}
        rerank_orders = {name:sorted(range(128),key=lambda i:(-scores[qid]['pairs'][i][key],ranks[i],axis[i]))
                         for name,key in [('QWEN3_DIRECT','logit'),('QWEN3_OFFICIAL_SIGMOID','sigmoid')]}
        orders = dict(RAW=raw_order,**rerank_orders)
        for name in ('COST1','ALL_CE'):
            entry = old[qid]['models'][name]
            z = [0.] * 128
            for pos,value in zip(r['challenger_positions'],entry['logits_hex']):z[pos]=float.fromhex(value)
            order = sorted(range(128),key=lambda i:(-z[i],i!=r['winner'],i))
            assert axis[order[0]] == entry['selected']
            orders['ORIGINAL_'+name] = order
        for name in ('COST1','CE'):
            entry = preds[qid]['models'][name]
            z = [0.] * 128
            for pos,value in zip(r['challenger_positions'],entry['logits_hex']):z[pos]=float.fromhex(value)
            order = sorted(range(128),key=lambda i:(-z[i],i!=r['winner'],i))
            assert axis[order[0]] == entry['selected']
            orders['QWEN3_CAL_'+name] = order
        target_positions = {i for i,p in enumerate(axis) if labels[p] == role['identity']}
        assert len(target_positions) <= 1
        rr = {m:next((1./(j+1) for j,i in enumerate(order) if i in target_positions),0.) for m,order in orders.items()}
        rows.append(dict(query_id=qid,original_query_id=role['original_query_id'],component=role['component'],
                         fold=role['outer_fold'],target_in_C128=bool(target_positions),
                         selected={m:axis[v[0]] for m,v in orders.items()},
                         correct={m:v==1 for m,v in rr.items()}, reciprocal_rank_C128=rr))
    counts = {m:sum(r['correct'][m] for r in rows) for m in rows[0]['correct']}
    assert len(rows)==593 and counts['RAW']==426 and counts['ORIGINAL_COST1']==481 and counts['ORIGINAL_ALL_CE']==486
    assert sum(r['target_in_C128'] for r in rows)==570
    comparisons = {}
    for base in ('RAW','ORIGINAL_COST1'):
        for model in counts:
            if model == base:continue
            groups = defaultdict(list)
            for r in rows:groups[r['component']].append(int(r['correct'][model])-int(r['correct'][base]))
            values = np.array([np.mean(v) for _,v in sorted(groups.items())])
            rng = np.random.default_rng(20260923)
            boot = values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
            comparisons[base+'__to__'+model] = dict(
                rescue=sum(r['correct'][model] and not r['correct'][base] for r in rows),
                breaks=sum(r['correct'][base] and not r['correct'][model] for r in rows),
                changed=sum(r['selected'][base]!=r['selected'][model] for r in rows),
                component_balanced_delta=float(values.mean()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))
    result = dict(status='QWEN3_H593_MATCHED_RERANK_COMPLETE',authority=bind(AUTH),counts=counts,denominator=593,
                  target_in_C128=570,comparisons=comparisons,
                  MRR_C128={m:sum(r['reciprocal_rank_C128'][m] for r in rows)/593 for m in counts},
                  pair_seconds=sum(p['seconds'] for v in scores.values() for p in v['pairs']),
                  rows=rows,evidence=a['evidence'],no_new_hyp_or_external_go_claim=True)
    write(OUT/'result.json',result)
    write(OUT/'validation.json',dict(status='QWEN3_FULL593_AXES_BASELINES_FOLDS_COUNTS_PASS',
                                   authority=bind(AUTH),result=bind(OUT/'result.json'),fold_validations=seals))
    report = '# H593 同候选 Qwen3-VL 重排对照\n\n'
    report += '开发面板；自然 RAW C128；五折身份与来源组隔离；目标缺席的23张仍计入593。\n\n'
    report += '| 方法 | 正确/593 | 对RAW救回/误伤 | MRR@C128 |\n|---|---:|---:|---:|\n'
    for m,n in counts.items():
        c = comparisons.get('RAW__to__'+m,dict(rescue=0,breaks=0))
        report += f"| {m} | {n}/593 | {c['rescue']}/{c['breaks']} | {result['MRR_C128'][m]:.6f} |\n"
    report += '\n直接重排使用冻结预训练模型，不读取OCR。校准仅训练三个参数，使用原COST1/CE损失和TRAIN次序。\n'
    report += '这检验常规图像重排能否复现原收益；不单凭一个baseline证明机制新颖性。全部候选logit与官方sigmoid均已保留。\n'
    (OUT/'report_zh.md').write_text(report)
    print(dict(status=result['status'],counts=counts),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=('prepare','score','fit','join','self-test'))
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--fold',type=int)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--budget',type=int,default=720)
    args = p.parse_args()
    if args.stage == 'prepare':prepare()
    elif args.stage == 'self-test':self_test()
    else:
        a,inputs = guard(args.stage,args.fold)
        if args.stage == 'score':
            assert 0 <= args.shard < 50
            sys.exit(score(a,inputs,args.shard,args.pilot,args.budget))
        elif args.stage == 'fit':
            assert args.fold in range(5)
            sys.exit(fit(a,inputs,args.fold,args.budget))
        else:join(a,inputs)
