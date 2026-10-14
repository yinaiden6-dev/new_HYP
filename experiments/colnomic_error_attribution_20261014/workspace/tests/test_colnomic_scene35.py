import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

_spec = importlib.util.spec_from_file_location("scene35", Path(__file__).parents[1] / "scripts/colnomic_mechanism/scene35.py")
s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s)


class Scene35Tests(unittest.TestCase):
    def prediction(self, correct=False, rank=12):
        return dict(correct=correct, rank=rank, data_issue=False,
            positive_design_ids=["right"], selected_design_id="right" if correct else "wrong")

    def test_E4_priority_does_not_replace_literal_recall_miss(self):
        pred={"B03":{"q":self.prediction(True,1)},"B05":{"q":self.prediction(False,500)},
            "F03":{"q":self.prediction(False,999)}}
        self.assertEqual(s.automatic_main("F03","q",pred),"E4")
        self.assertEqual(s.rank_bucket(pred["F03"]["q"]["rank"]),">10")

    def test_E2_not_candidate_generation_evidence(self):
        self.assertEqual(s.automatic_main("B03","q",{"B03":{"q":self.prediction(False,11)}}),"E2")
        self.assertEqual(s.automatic_main("B03","missing",{"B03":{}}),None)

    def test_four_cell_rescue_harm_net_with_missing(self):
        qids=["a","b","c","d","e"];photos={q:dict(package_id=q) for q in qids}
        groups={q:dict(group_id=q) for q in qids}
        result=s.paired_statistics(np.array([1,0,1,0,-1]),np.array([1,1,0,0,1]),
            np.ones(5,dtype=bool),qids,photos,groups,repetitions=20)
        self.assertEqual([result[k] for k in ["both_correct","rescues","harms","both_wrong"]],[1,1,1,1])
        self.assertEqual(result["missing_predictions"],1)
        self.assertEqual(result["net"],0)

    def test_scene_dimensions_count_each_photo_once_per_dimension(self):
        labels={"a":dict(background="single_object",placement="handheld",occlusion="none"),
            "b":dict(background="multi_object",placement="tabletop",occlusion="foreground")}
        masks=s.scene_masks(labels,["a","b"])
        for dimension in s.DIMENSIONS:
            self.assertEqual(sum(int(m.sum()) for d,l,m in masks if d==dimension),2)
        self.assertEqual(sum(int(m.sum()) for d,l,m in masks if d=="joint"),2)

    def test_unaudited_scene_does_not_become_reviewed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reviews=root/"reviews";reviews.mkdir()
            labels={"a":dict(background="single_object",placement="handheld",occlusion="none",scene_reviewed=False)}
            s.write_json(reviews/"a.json",dict(photo_id="a",scene_labels=dict(background="unclear",placement="handheld",occlusion="foreground"),scene_labels_reviewed=False))
            revised,events,audit=s.load_revisions(reviews,labels,{},root)
            self.assertEqual(revised["a"],labels["a"])
            self.assertEqual(events,[])
            self.assertIn("a",audit)

    def test_consolidated_uncertainty_replaces_old_label_per_dimension(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'actual_review.json';source.write_text('{}')
            image=root/'actual_image.jpg';image.write_bytes(b'image-evidence')
            labels={'a':dict(background='single_object',placement='tabletop',occlusion='none',scene_reviewed=True)}
            dimensions={d:dict(value=value,status='reviewed_uncertain' if value=='unclear' else 'reviewed_known',
                source_path=source.name,source_sha256=s.file_hash(source),
                evidence=[dict(path=image.name,actual_sha256=s.file_hash(image),declared_sha256=s.file_hash(image))])
                for d,value in dict(background='unclear',placement='handheld',occlusion='foreground').items()}
            revisions=root/'revisions.json';s.write_json(revisions,dict(records=[dict(photo_id='a',scene_dimensions=dimensions)]))
            revised,events,audit=s.load_revisions(revisions,labels,{},root)
            self.assertEqual(revised['a']['background'],'unclear')
            self.assertEqual(revised['a']['placement'],'handheld')
            self.assertEqual(revised['a']['occlusion'],'foreground')
            self.assertFalse(revised['a']['scene_reviewed'])
            self.assertTrue(events[0]['scene_audit_complete'])

    def test_consolidated_evidence_hash_is_actually_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'actual_review.json';source.write_text('{}')
            image=root/'image.jpg';image.write_bytes(b'image-evidence')
            revisions=root/'revisions.json';s.write_json(revisions,dict(records=[dict(photo_id='a',scene_dimensions={
                'background':dict(value='multi_object',status='reviewed_known',source_path=source.name,
                    source_sha256=s.file_hash(source),evidence=[dict(path=image.name,actual_sha256='wrong')])})]))
            with self.assertRaisesRegex(ValueError,'evidence hash mismatch'):
                s.load_revisions(revisions,{'a':dict(background='single_object',scene_reviewed=False)}, {},root)


if __name__ == "__main__":
    unittest.main()
