"""Preserve correct/wrong candidate semantics, raw matrix axes and image decoding."""
import numpy as np
from PIL import Image
import torch

from scripts.supplement_colnomic_archives import rank_candidates, dense_similarity
from scripts import encode_colnomic_images as encoding
from evaluation.scoring import multivector_scores


def test_correct_top1_is_not_reported_as_strongest_wrong():
    result = rank_candidates([.8,.95,.7,.6], ['positive','positive_alias','wrong','wrong_alias'],
        ['p','p','n','n'], ['positive'])
    assert result['top1']=='positive_alias' and result['top1_correct']
    assert result['strongest_wrong']=='wrong' and result['strongest_wrong_rank']==2
    assert result['positive_rank']==1


def test_stable_sort_precedes_sha_deduplication():
    result = rank_candidates([.7,.9,.9,.8], ['positive','wrong_later_row','wrong_tied','wrong_low'],
        ['p','n','m','n'], ['positive'])
    assert result['top1']=='wrong_later_row'
    assert result['strongest_wrong']=='wrong_later_row'
    assert result['positive_rank']==3  # Lower scoring duplicate is removed after sorting.


def test_complete_matrix_keeps_zero_rows_but_replay_masks_them():
    q=torch.tensor([[0.,0.],[1.,0.],[0.,1.]],dtype=torch.float16)
    r=torch.tensor([[0.,0.],[-1.,-.2],[-.5,-.6]],dtype=torch.float16)
    matrix, qi, matches, maximum=dense_similarity(q,r)
    assert matrix.shape==(3,3) and matrix.dtype==np.float32
    assert np.all(matrix[0]==0) and np.all(matrix[:,0]==0)
    assert qi.tolist()==[1,2] and matches.tolist()==[2,1]
    baseline,_=multivector_scores([q],[r],device='cpu',token_block=1)
    np.testing.assert_allclose(maximum.mean(),baseline[0,0].item(),atol=1e-7)
    assert maximum.mean()<0  # Unmasked zero references would incorrectly make this zero.


def test_encoding_preserves_raw_exif_frame(tmp_path,monkeypatch):
    path=tmp_path/'raw.jpg'
    image=Image.new('RGB',(4,3),(24,55,99))
    exif=image.getexif();exif[274]=6
    image.save(path,exif=exif)
    class Tokenizer:
        def convert_tokens_to_ids(self,_):return 100
    class Processor:
        tokenizer=Tokenizer();image_token='image'
        def process_images(self,images):
            assert images[0].size==(4,3) and images[0].mode=='RGB'
            return dict(input_ids=torch.tensor([[1,100,100,2]]),
                attention_mask=torch.ones((1,4),dtype=torch.long),image_grid_thw=torch.tensor([[1,2,4]]))
    monkeypatch.setattr(encoding,'token_layout',lambda *args:dict(start=1,count=2,total=4,height=1,width=2))
    record=encoding.prepare_record(Processor(),path,'edited_image')
    assert record['raw_width']==4 and record['raw_height']==3
    assert record['exif_orientation']==6
    assert record['attended_input_ids']==[1,100,100,2]
    assert len(record['image_sha256'])==64
