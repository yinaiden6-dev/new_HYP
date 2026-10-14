import sys

import torch

from drugpack.method.common import ROOT
from drugpack.method.heads import cost, features, summary


def test_cost_matches_original_loss_and_gradients():
    sys.path.insert(0, str(ROOT/'drugpack/method/legacy'))
    from run_rc_internal_m_condition_scale_v4 import cost_from_logits
    for target in (0, 1, 17, 127):
        for tied in (False, True):
            torch.manual_seed(17)
            z = torch.zeros(127, dtype=torch.float64) if tied else torch.randn(127, dtype=torch.float64)
            x, y = z.clone().requires_grad_(), z.clone().requires_grad_()
            row = dict(target_positions=[target], winner_index=0, challenger_positions=list(range(1,128)))
            a, b = cost(x, target), cost_from_logits(row, y)
            assert torch.equal(a, b)
            assert torch.equal(torch.autograd.grad(a,x)[0], torch.autograd.grad(b,y)[0])


def test_features_match_original():
    from run_rc_internal_m_condition_scale_v4 import features as original
    torch.manual_seed(31)
    raw = torch.randn(128, dtype=torch.float64)
    x = torch.rand(128, 8, dtype=torch.float64)
    row = dict(raw_scores=raw.tolist(), M=x[:,1].tolist(), winner_index=0, challenger_positions=list(range(1,128)))
    for kind in ('INTERNAL3','ADDITIVE4','PRODUCT5'):
        assert torch.equal(features(raw,x,kind), original(row,x[:,0],kind))


def test_missing_target_is_counted_as_failure():
    rows = [dict(row=dict(id='a',group='g',identity='a',candidates=[7,5]),positions=[1]),
            dict(row=dict(id='b',group='g',identity='b',candidates=[9,2]),positions=[])]
    report, predictions = summary(rows, [[1,0],[0,1]])
    assert report['n'] == 2 and report['micro']['R@1'] == .5
    assert predictions[1]['rank'] is None and report['candidate_recall'] == .5


def test_full_gallery_tail_and_multiple_positives():
    import numpy as np
    from drugpack.method.ranking import full_rank, restored_rank
    scores = np.array([.8,.7,.7,.6,.4,.1])
    gallery = [0,1,2,3,4,5]
    assert full_rank(scores,gallery,{2}) == 3
    assert full_rank(scores,gallery,{2,5}) == 3
    assert full_rank(scores,[0,2,3,4,5],{2}) == 2
    assert restored_rank(5,[0,1,2],{4},[2,1,0]) == 5
    assert restored_rank(3,[0,1,2],{2,5},[2,1,0]) == 1
    assert restored_rank(2,[0,1,2],{1,2},[2,0,1]) == 1


def test_adapter_replay_preserves_forward_batch_numerics_and_dense_gradient(monkeypatch):
    from types import SimpleNamespace
    from drugpack.method import post
    def score_positions(model, inputs, item, positions, intervention=None):
        values = torch.arange(16,dtype=torch.float32)[positions]/16
        return model.adapter.weight.flatten()[0]*values + len(positions)*.001
    monkeypatch.setattr(post,'score_positions',score_positions)
    models = [SimpleNamespace(adapter=torch.nn.Linear(1,1,bias=False)) for _ in range(2)]
    for model in models:
        model.adapter.weight.data.fill_(.1)
    heads = [torch.nn.Parameter(torch.tensor([1.,.2,.1],dtype=torch.float64)) for _ in models]
    optimizers = [torch.optim.SGD([*m.adapter.parameters(),h],lr=.01) for m,h in zip(models,heads)]
    item = dict(row=dict(candidates=list(range(16))),target=3,
                features=dict(raw=torch.linspace(1.,0.,16),values=torch.zeros(16,8)))
    inputs = SimpleNamespace(device='cpu')
    content = torch.cat([score_positions(models[0],inputs,item,list(range(i,i+8))) for i in (0,8)]).double()
    objective = cost(features(item['features']['raw'],item['features']['values'],content=content)@heads[0],3)
    objective.backward()
    torch.nn.utils.clip_grad_norm_(models[0].adapter.parameters(),1.)
    torch.nn.utils.clip_grad_norm_([heads[0]],1.)
    optimizers[0].step()
    result = post.update(models[1],inputs,item,heads[1],optimizers[1],'cost',8)
    assert result['replay_error']==0.
    assert torch.allclose(models[0].adapter.weight,models[1].adapter.weight,atol=1e-7,rtol=0)
    assert torch.allclose(heads[0],heads[1],atol=1e-12,rtol=0)


def test_declared_training_jobs_have_unique_output_names():
    from drugpack.method.common import read
    from drugpack.method.select import run_name
    jobs = read(ROOT/'configs/method/jobs.json') + read(ROOT/'configs/method/head_jobs.json')
    names = [run_name(j) for j in jobs]
    assert len(set(names)) == len(names)
    assert 'post_real_cost_s17_b100_d16' in names
    assert 'medicine_catalog_embedding_post_real_cost_s17_b100_d16' in names
    assert 'patch_post_real_cost_s17_b100_d16' in names
    assert 'medicine_catalog_patch_post_real_cost_s17_b100_d16' in names


def test_reader_sparse_replay_matches_dense_objective_gradient():
    from copy import deepcopy
    from types import SimpleNamespace
    from drugpack.method import reader
    from drugpack.method.common import key
    torch.manual_seed(17)
    query = torch.nn.functional.normalize(torch.randn(4,128),dim=-1)
    refs = [torch.nn.functional.normalize(torch.randn(n,128),dim=-1) for n in (4,6,6,4)]
    row = dict(id='gradient-check',candidates=list(range(4)))
    hidden = dict(native=query,mask=torch.ones(4,dtype=torch.bool),grid=torch.tensor([[1,4,4]]))
    inputs = SimpleNamespace(device='cpu',hidden={key(row):hidden},reference=lambda i:refs[i],
        bank=SimpleNamespace(layout={(True,i):dict(height=2,width=len(r)//2) for i,r in enumerate(refs)}))
    values = torch.zeros(4,8)
    values[:,0] = torch.stack([(query@r.T).max(1).values.mean() for r in refs])
    item = dict(row=row,target=2,features=dict(raw=torch.tensor([3.,2.,1.,.5]),values=values))
    for arm in ('COMPRESSED','SET','SPATIAL'):
        dense = reader.Reader(arm,17)
        with torch.no_grad():
            dense.out.weight.normal_(std=.1)
        sparse = deepcopy(dense)
        heads = [torch.nn.Parameter(torch.tensor([.4,.2,.1],dtype=torch.float64)) for _ in range(2)]
        optimizers = [torch.optim.SGD([*m.parameters(),h],lr=.01) for m,h in zip((dense,sparse),heads)]
        scores = torch.stack([reader.relation_score(dense,inputs,item,i) for i in range(4)]).double()
        cost(reader.logits(item,scores,heads[0]),item['target']).backward()
        torch.nn.utils.clip_grad_norm_(dense.parameters(),1.)
        torch.nn.utils.clip_grad_norm_([heads[0]],1.)
        optimizers[0].step()
        result = reader.update(sparse,inputs,item,heads[1],optimizers[1])
        assert result['replay_error'] < 2e-6
        assert all(torch.allclose(a,b,atol=1e-7,rtol=0) for a,b in zip(dense.parameters(),sparse.parameters()))
        assert torch.allclose(heads[0],heads[1],atol=1e-12,rtol=0)


def test_reader_replay_failure_does_not_update_parameters(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from drugpack.method import reader
    model = torch.nn.Linear(1,1,bias=False)
    model.arm = 'SET'
    head = torch.nn.Parameter(torch.tensor([.4,.2,.1],dtype=torch.float64))
    optimizer = torch.optim.SGD([*model.parameters(),head],lr=.01)
    item = dict(row=dict(id='replay-failure',candidates=[0,1,2]),target=1,
                features=dict(raw=torch.tensor([3.,2.,1.]),values=torch.ones(3,8)))
    def relation(model,inputs,item,index,intervention='native'):
        return model.weight.flatten()[0]*index + (.01 if torch.is_grad_enabled() else 0.)
    monkeypatch.setattr(reader,'relation_score',relation)
    before = [p.detach().clone() for p in (*model.parameters(),head)]
    with pytest.raises(ValueError,match='Reader gradient replay differs'):
        reader.update(model,SimpleNamespace(),item,head,optimizer)
    assert all(torch.equal(a,b) for a,b in zip(before,(*model.parameters(),head)))


def test_official_forb_ap_denominator():
    from drugpack.method.ranking import average_precisions
    result = average_precisions({'a':1,'b':3})
    assert result['mAP@1'] == .5
    assert result['mAP@5'] == (1 + 2/3) / 2


def test_deduplication_uses_best_duplicate_then_original_row_ties():
    from drugpack.method.ranking import deduplicated_rank
    assert deduplicated_rank([.1,.9,.9,.8],[0,1,0,3],{0}) == 2
    assert deduplicated_rank([1.,.9,.9,.8],[0,1,0,3],{0}) == 1


def test_online_maxsim_matches_existing_masked_scorer():
    from drugpack.method.benchmark import maxsim
    from evaluation.scoring import multivector_scores
    torch.manual_seed(17)
    query = torch.randn(9,8)
    references = [torch.randn(n,8) for n in (3,5,11)]
    references[0] = -query.abs().mean(0)[None].expand(3,-1).clone()
    actual = maxsim(query,references,'cpu')
    expected,_ = multivector_scores([query],references,'cpu',reference_batch=3)
    assert torch.allclose(actual,expected[0],atol=1e-6,rtol=1e-6)


def test_roma_cache_is_unchanged_when_matching_head_replaces_feature():
    from drugpack.method.roma import CachedModule
    class Descriptor(torch.nn.Module):
        def forward(self,image):
            return [image*2,image*3]
    cached=CachedModule(Descriptor(),4096)
    image=torch.ones(1,3,2,2)
    image._cache_id='image'
    for _ in range(4):
        result=cached(image)
        assert torch.equal(result[-1],image*3)
        result[-1]=result[-1]+100


def test_mass_intervention_updates_dependent_product_features():
    from drugpack.method.heads import intervene_mass
    values = torch.rand(3, 8)
    values[:, 1] = torch.tensor([.1, .2, .3])
    values[:, 3] = values[:, 1] * values[:, 2]
    original = values.clone()
    changed = intervene_mass(values, [2, 0, 1], 'shuffled')
    assert torch.equal(changed[:, 1], torch.tensor([.2, .3, .1]))
    for mode in ('constant', 'shuffled'):
        changed = intervene_mass(values, [2, 0, 1], mode)
        assert torch.equal(changed[:, 3], changed[:, 1] * changed[:, 2])
        assert torch.equal(changed[:, 2], values[:, 2])
    assert torch.equal(values, original)


def test_pairwise_comparison_joins_ids_and_uses_the_named_reference():
    from drugpack.method.ranking import paired_comparison
    rows = [dict(id='a', group='g1', identity='a', rank=1, raw_rank=9),
            dict(id='b', group='g2', identity='b', rank=3, raw_rank=9)]
    reference = [dict(rows[1], rank=1), dict(rows[0], rank=2)]
    result = paired_comparison(rows, reference, 'external head')
    assert result['rescued'] == result['harmed'] == 1
    assert result['delta']['R@1'] == 0
    assert result['paired_intervals']['direction'] == 'method minus external head'
    assert rows[0]['raw_rank'] == 9


def test_scalar_head_starts_at_linear_baseline_and_matches_capacity():
    from drugpack.method.scalar_head import ScalarHead
    torch.manual_seed(17)
    raw, values, theta = torch.rand(8), torch.rand(8, 8), torch.randn(4, dtype=torch.float64)
    for width, expected in ((61, 4210), (340, 118321)):
        model = ScalarHead(width, 17, dict(mean=-2., std=.7))
        assert sum(p.numel() for p in model.parameters()) == expected
        actual = model(raw, values, theta)
        assert torch.equal(actual, features(raw, values, 'ADDITIVE4') @ theta)
        cost(actual, 3).backward()
        assert model.net[-1].weight.grad.norm() > 0
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_scalar_head_evaluation_matches_training_for_each_prefix(tmp_path):
    from drugpack.method.common import sha
    from drugpack.method.evaluate import Predictor
    from drugpack.method.scalar_head import ScalarHead
    config = dict(width=61, seed=17, kind='SCALAR_MLP')
    norm = dict(mean=-2., std=.7)
    model = ScalarHead(61, 17, norm)
    with torch.no_grad():
        model.net[-1].weight.normal_(std=.1)
    theta = torch.randn(4, dtype=torch.float64)
    path = tmp_path/'head.pt'
    torch.save(dict(config=config, normalization=norm, mlp=model.state_dict(), theta=theta), path)
    predictor = Predictor({}, dict(checkpoint=str(path), checkpoint_sha256=sha(path)), 'cpu')
    raw, values = torch.rand(8), torch.rand(8, 8)
    item = dict(row=dict(candidates=list(range(8))), features=dict(raw=raw, values=values))
    for k in (3, 8):
        actual = predictor.predict(item, k, None)
        expected = torch.cat((torch.zeros(1), model(raw[:k], values[:k], theta)))
        assert torch.equal(actual['scores'], expected)
        assert actual['permutation'] == expected.argsort(descending=True, stable=True).tolist()


def test_query_shards_cover_all_forb_queries_once():
    from drugpack.method.dispatch import query_ranges
    ranges = query_ranges(8995)
    assert ranges[-1] == (8192, 8995)
    assert [q for start, stop in ranges for q in range(start, stop)] == list(range(8995))


def test_queue_waits_only_for_its_declared_query_range(tmp_path, monkeypatch):
    from drugpack.method import queue
    from drugpack.method.common import key, save
    monkeypatch.setattr(queue, 'WORK', tmp_path)
    rows = [dict(id=str(i), split='transfer') for i in range(3)]
    save(tmp_path/'sample/manifest.json', dict(rows=rows))
    for row in rows[:2]:
        save(tmp_path/'sample/features'/(key(row)+'.pt'), {})
    base = dict(requires=dict(sample=['transfer']))
    assert not queue.ready(base)
    assert queue.ready(dict(base, query_range=[0, 2]))
    assert not queue.ready(dict(base, query_range=[2, 3]))
    save(tmp_path/'sample/features'/(key(rows[2])+'.pt'), {})
    assert queue.ready(base)


def test_directional_head_uses_each_side_and_intervenes_consistently():
    from drugpack.method.heads import intervene_mass, mass_columns
    raw = torch.tensor([3., 2., 1.])
    values = torch.ones(3, 8)
    values[:, 6] = torch.tensor([.2, .4, .6])
    values[:, 7] = torch.tensor([.8, .4, .2])
    assert (features(raw, values, 'RAW_U')[:, 1] > 0).all()
    assert (features(raw, values, 'RAW_V')[:, 1] < 0).all()
    changed = intervene_mass(values, [0, 1, 2], 'constant', mass_columns('ADDITIVE_UV'))
    assert torch.equal(features(raw, changed, 'ADDITIVE_UV')[:, 1:3], torch.zeros(2, 2, dtype=torch.float64))
    assert torch.equal(changed[:, 1], (changed[:, 6] * changed[:, 7]).sqrt())
