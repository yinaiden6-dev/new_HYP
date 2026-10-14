"""No-encoding contracts: natural candidates, original X11, and frozen decisions."""
import importlib.util
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_content_box_ce_polish18 import DescriptorCache,construct_x11,math_functions,natural_candidates,save_npz,PACKAGE,read,relocate_manifest,recipe_input_hash,validate_reuse_contract,accepted_pair_bindings,digest


class Polish18PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core,cls.content,cls.gate=math_functions()
        spec=importlib.util.spec_from_file_location('polish18_infer_test',PACKAGE/'infer.py')
        cls.infer=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.infer)
        cls.model=read(PACKAGE/'models/fold0.json')

    def test_natural_c128_identity_dedup_sort_and_no_target_insert(self):
        refs=[dict(sha256=str(i)) for i in range(130)];refs[1]['sha256']=refs[0]['sha256']
        scores=np.arange(130,dtype=np.float32)[::-1].copy()
        axis,ranked,winner=natural_candidates(scores,refs)
        self.assertEqual(axis,sorted(ranked));self.assertEqual(len(axis),128)
        self.assertNotIn(1,axis);self.assertNotIn(129,axis)  # Lowest-scoring identity stays absent.
        self.assertEqual(axis[winner],0)
        tied=np.ones(130);axis,ranked,winner=natural_candidates(tied,refs)
        self.assertEqual(ranked[:2],[0,2]);self.assertEqual(winner,0)

    def test_content_stats_preserve_magnitudes_remove_only_zero_tokens(self):
        q=torch.tensor([[2.,0.],[0.,3.],[0.,0.]],dtype=torch.float16)
        r=torch.tensor([[4.,0.],[0.,5.]],dtype=torch.float16)
        actual=self.content['statistics'](self.content['active'](q),self.content['active'](r)).numpy()
        independent=self.content['numpy_statistics'](q.numpy(),r.numpy())
        np.testing.assert_array_equal(actual,independent)
        self.assertEqual(actual[0],11.5);self.assertEqual(actual[4],5.75)
        self.assertFalse(np.allclose(actual,[1,1,1,1,.5]))

    def test_complete_x11_column_order_population_std_and_symmetric_features(self):
        raw=[float(i) for i in range(128)];winner=7;positions=[i for i in range(128) if i!=winner]
        q=dict(raw_scores=raw,winner=winner,challenger_positions=positions)
        c4=np.array([[i+1.,.5,i+.8,i+.7] for i in range(128)])
        stats=np.array([[i+1.,i+2.,i+3.,i+4.,i+5.] for i in range(128)])
        x=construct_x11(q,c4,stats,self.content,self.gate)
        self.assertEqual(x.shape,(127,11))
        self.assertAlmostEqual(x[0,0],-7/np.std(raw,ddof=0),places=14)
        np.testing.assert_allclose(x[:,6:],(stats[positions]-stats[winner])/(np.abs(stats[positions])+np.abs(stats[winner])+1e-12))
        with self.assertRaises(Exception):construct_x11(q,c4[:,1:2],stats,self.content,self.gate)

    def test_model_example_exact_and_original_infer_rejects_six_only(self):
        record=read(PACKAGE/'example_input.json')
        self.assertEqual(self.infer.predict(record,self.model),read(PACKAGE/'example_expected.json'))
        bad=dict(record,X11=np.asarray(record['X11'])[:,:6].tolist())
        with self.assertRaises(ValueError):self.infer.predict(bad,self.model)

    def test_zero_threshold_and_challenger_ties(self):
        record=dict(feature_schema=self.infer.FEATURE_SCHEMA,X11=np.zeros((127,11)).tolist(),
            candidate_physical_rows=list(range(128)),winner=19,challenger_positions=[i for i in range(128) if i!=19])
        model=dict(self.model,parameters_hex=[0.0.hex()]*18)
        self.assertEqual(self.infer.predict(record,model)['action'],'HOLD')
        model['parameters_hex'][-1]=1.0.hex()
        prediction=self.infer.predict(record,model)
        self.assertEqual(prediction['selected'],0);self.assertEqual(prediction['action'],'SWITCH')

    def test_pair_storage_is_lossless_for_original_float_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pair.npz'
            values=dict(overlap_AB=np.random.default_rng(17).random((7,8),dtype=np.float32),u=np.array([.1,.2],dtype=np.float64))
            save_npz(path,values)
            with np.load(path,allow_pickle=False) as result:
                for k,v in values.items():np.testing.assert_array_equal(result[k],v);self.assertEqual(result[k].dtype,v.dtype)

    def test_cpu_descriptor_cache_preserves_query_reference_binding_and_containers(self):
        class Frozen(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,x):self.calls+=1;return [x.sum().reshape(1)]
        module=Frozen();cache=DescriptorCache(module,0,1000)
        a=torch.ones(1,3,2,2);b=a*2;c=a*3
        cache.start('query','reference_a');q=cache(a);r=cache(b)
        q[0]=torch.tensor([-100.])  # A consumer replaces a list item.
        cache.start('query','reference_b');self.assertEqual(float(cache(a)[0]),12.);self.assertEqual(float(cache(c)[0]),36.)
        cache.start('query','reference_a');self.assertEqual(float(cache(a)[0]),12.);self.assertEqual(float(cache(b)[0]),24.)
        self.assertEqual(module.calls,3);self.assertEqual(cache.hits['cpu'],3)

    def test_transfer_relocates_only_paths_preserving_recipe_and_pair_identity(self):
        source='/old/mount';relative='weights/colnomic7b/adapter/preprocessor_config.json'
        original=dict(signature='unchanged-evidence-binding',transfer=dict(source_root=source),
            recipe=dict(inputs={source+'/'+relative:'processor-hash'}),query_list=source+'/dataset/lists/test.csv',
            queries=[dict(id='photo_a',path=source+'/dataset/photos/a.jpg',candidate_physical_rows=[2,7],winner=1)],
            references={'7':dict(id='reference_b',path=source+'/dataset/references/b.jpg',sha256='same-image')})
        relocated=relocate_manifest(original,Path('/new/location'))
        self.assertEqual(relocated['queries'][0]['path'],'/new/location/dataset/photos/a.jpg')
        self.assertEqual(relocated['references']['7']['path'],'/new/location/dataset/references/b.jpg')
        self.assertEqual(relocated['recipe'],original['recipe']);self.assertEqual(relocated['signature'],original['signature'])
        self.assertEqual(relocated['queries'][0]['candidate_physical_rows'],[2,7]);self.assertEqual(relocated['queries'][0]['winner'],1)
        self.assertEqual(recipe_input_hash(relocated,relative),'processor-hash')
        self.assertEqual(original['queries'][0]['path'],source+'/dataset/photos/a.jpg')
        bad=dict(original,queries=[dict(original['queries'][0],path='/outside/a.jpg')])
        with self.assertRaises(ValueError):relocate_manifest(bad,Path('/new/location'))

    def test_population_reuse_rejects_changed_raw_candidates_geometry_or_inputs(self):
        query=dict(id='photo_a',sha256='image',source_sha256='source',package_id='package_a',bank_index=3,
            raw_height=100,raw_width=100,exif_orientation=1,layout={'height':2,'width':2},processor_input_frame='RAW',
            image_tokens_sha256='qt',all_tokens_sha256='qall',candidate_physical_rows=[2,7],ranked_physical_rows=[7,2],
            winner=1,challenger_positions=[0],raw_scores=[.1,.3],ordinal=0)
        references={str(i):dict(id='reference_'+str(i),physical_row=i,bank_index=i,sha256='image'+str(i),design_id='d'+str(i),
            raw_height=100,raw_width=100,exif_orientation=1,layout={'height':2,'width':2},processor_input_frame='RAW',
            image_tokens_sha256='rt'+str(i),all_tokens_sha256='rall'+str(i)) for i in [2,7]}
        src=dict(query_list=str(ROOT/'dataset/lists/test.csv'),recipe=dict(model={'fold':0},threshold=0,
            inputs={str(ROOT/'dataset/lists/test.csv'):'test',str(ROOT/'dataset/lists/photos.csv'):'photos',str(ROOT/'cache/tokens.pt'):'tokens'}),
            queries=[query],references=references)
        dst=copy.deepcopy(src);dst['query_list']=str(ROOT/'dataset/lists/photos.csv');dst['recipe']['inputs'].pop(str(ROOT/'dataset/lists/test.csv'));dst['queries'][0]['ordinal']=73
        for m in [src,dst]:m['signature']=digest(m['recipe'])
        self.assertEqual(validate_reuse_contract(src,dst),['photo_a'])
        for key,value in [('raw_scores',[.2,.3]),('candidate_physical_rows',[7,2]),('exif_orientation',2)]:
            bad=copy.deepcopy(dst);bad['queries'][0][key]=value
            with self.assertRaises(ValueError):validate_reuse_contract(src,bad)
        bad=copy.deepcopy(dst);bad['recipe']['inputs'][str(ROOT/'cache/tokens.pt')]='new-tokens';bad['signature']=digest(bad['recipe'])
        with self.assertRaises(ValueError):validate_reuse_contract(src,bad)

    def test_imported_pair_bindings_are_scoped_to_validated_queries_and_tokens(self):
        q=dict(id='photo_a',image_tokens_sha256='q');r=dict(image_tokens_sha256='r')
        m=dict(signature='new',evidence_reuse=[dict(source_signature='old',query_ids=['photo_a'],validated_recipe_equivalence=True)])
        legacy=digest(['old','photo_a','q','r']);self.assertIn(legacy,accepted_pair_bindings(m,q,r))
        self.assertNotIn(legacy,accepted_pair_bindings(m,dict(q,id='photo_b'),r))
        self.assertNotIn(legacy,accepted_pair_bindings(m,q,dict(image_tokens_sha256='changed')))
        m['evidence_reuse'][0]['validated_recipe_equivalence']=False
        with self.assertRaises(ValueError):accepted_pair_bindings(m,q,r)


if __name__=='__main__':unittest.main()
