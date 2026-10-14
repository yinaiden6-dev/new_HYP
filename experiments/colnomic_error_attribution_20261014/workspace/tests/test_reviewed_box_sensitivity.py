"""Scientific safeguards for box-dependent refitting and fixed-support controls."""
from collections import defaultdict

import numpy as np
import torch

from scripts.colnomic_mechanism.geometry import anchors
from scripts.colnomic_mechanism.reviewed_box_sensitivity import box_changes, recompute_query


def synthetic_inputs():
    lines=[]
    for i,(x,y) in enumerate([(.1,.1),(.55,.2),(.1,.65),(.55,.75)]):
        polygon=np.array([[x,y],[x+.23,y],[x+.23,y+.05],[x,y+.05]])*100
        lines.append(dict(text='unique anchor '+str(i),polygon=polygon.tolist()))
    ocr=dict(width=100,height=100,lines=lines)
    spec=dict(layout=dict(start=1,count=4,width=2,height=2,total=5),exif_orientation=1)
    q=torch.tensor([[.2,.2],[1.,0.],[0.,1.],[.9,.1],[.1,.9]])
    r=torch.tensor([[.2,.2],[0.,1.],[1.,0.],[.1,.9],[.9,.1]])
    base=(q@r.T).max(1).values.mean().reshape(1,1)
    d=dict(qb=dict(ids=['query'],embeddings=[q]),rb=dict(ids=['reference'],embeddings=[r]),
           image=dict(scores=base),metadata=dict(photos=[spec],references=[spec]),qo=dict(query=ocr))
    ra=[anchors(ocr)];postings=defaultdict(list)
    for key in ra[0]:postings[key].append(0)
    record=dict(width=100,height=100,bbox=[0.,0.,100.,100.],status='confirmed')
    return d,ra,postings,record


def test_box_changes_ignore_metadata_but_detect_geometry_and_status():
    old={'q':dict(bbox=[0,0,10,10],width=10,height=10,status='confirmed',updated_at='old')}
    assert box_changes(old,{'q':{**old['q'],'updated_at':'new'}})==[]
    assert box_changes(old,{'q':{**old['q'],'bbox':[0,0,9,10]}})==['q']
    assert box_changes(old,{'q':{**old['q'],'status':'unconfirmed'}})==['q']


def test_revised_box_reproposes_and_removes_previous_homography():
    d,ra,postings,record=synthetic_inputs()
    old=recompute_query(d,0,record,ra,postings,[0.,.2])
    assert old['accepted_alignment'].tolist()==[True]
    assert len(old['fits'])==1 and old['subject_tokens']==4
    revised={**record,'bbox':[80.,80.,100.,100.]}
    new=recompute_query(d,0,revised,ra,postings,[0.,.2])
    assert new['fits']==[] and new['proposed_indices']==[]
    assert not new['accepted_alignment'].any()
    assert new['subject_fraction'].tolist()==[0.]
    np.testing.assert_array_equal(new['penalty'],np.repeat(d['image']['scores'].numpy(),2,axis=0))
    np.testing.assert_array_equal(new['shuffle'],new['penalty'])


def test_expanding_box_adds_alignment_and_shuffle_preserves_new_support():
    d,ra,postings,record=synthetic_inputs()
    small=recompute_query(d,0,{**record,'bbox':[80.,80.,100.,100.]},ra,postings,[0.,.2])
    full=recompute_query(d,0,record,ra,postings,[0.,.2])
    assert not small['accepted_alignment'].any() and full['accepted_alignment'].all()
    assert full['shuffle_audit'][0]['subject_tokens']==full['fits'][0]['subject_tokens']==4
    assert full['shuffle_audit'][0]['query_valid_tokens']==5
    assert full['shuffle_audit'][0]['nonidentity_permutation']
    assert full['subject_fraction'][0]==np.float32(4/5)
    np.testing.assert_array_equal(full['penalty'][0],d['image']['scores'][0].numpy())
    np.testing.assert_array_equal(full['shuffle'][0],full['penalty'][0])


def test_weighted_loader_replays_observed_historical_mean_bits(tmp_path,monkeypatch):
    # Real failure from photo_0913.../reference_001060: 384 subject / 731 valid.
    # Rounding a float32 fraction first gives one ULP below the weighted input.
    import scripts.colnomic_mechanism.reviewed_box_sensitivity as module
    from scripts.colnomic_mechanism.common import write,sha
    monkeypatch.setattr(module,'ROOT',tmp_path)
    parameters=dict(gamma=.2,lambda_value=.1,lambda_index=0)
    penalty=np.array([[[.5472301840782166]]],np.float32)
    fraction=np.array([[384/731]],np.float32)
    np.savez(tmp_path/'scores.npz',geo_penalty=penalty,subject_fraction=fraction,
        query_ids=['q'],reference_ids=['r'],reference_sha256=['reference_sha'])
    write(tmp_path/'fits.json',[dict(photo_id='q',reference_id='r',subject_tokens=384,homography=np.eye(3).tolist())])
    write(tmp_path/'boxes.json',dict(records=dict(q=dict(bbox=[0,0,100,100],width=100,height=100,status='confirmed'))))
    write(tmp_path/'selection.json',dict(selections=dict(B03=dict(combined=parameters))))
    write(tmp_path/'axes.json',dict(query_ids=['q'],reference_ids=['r']))
    paths=dict(scores='scores.npz',alignments='fits.json',boxes='boxes.json',selection='selection.json',score_axes='axes.json')
    manifest={**paths,**{key+'_sha256':sha(tmp_path/value) for key,value in paths.items()},'parameters':parameters}
    write(tmp_path/'manifest.json',manifest)
    d=dict(qb=dict(ids=['q'],embeddings=[torch.ones((731,1))]),rb=dict(ids=['r'],sha256=['reference_sha']))
    means,_,_,_=module.load_spatial_inputs(tmp_path/'manifest.json',d)
    assert int(means[0,0].view(np.uint32))==1059519640
    vector=penalty[0]+.2*fraction
    assert int(vector[0,0].view(np.uint32))==1059519639
    np.testing.assert_array_equal(np.load(tmp_path/'scores.npz')['geo_penalty'],penalty)
