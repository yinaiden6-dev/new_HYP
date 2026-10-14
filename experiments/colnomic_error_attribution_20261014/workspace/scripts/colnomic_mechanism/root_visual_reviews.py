"""Persist notes ONLY after root actually views the referenced audit sheet."""
from datetime import datetime, timezone
from scripts.colnomic_mechanism.common import ROOT, read, write, sha

FOLDER = ROOT / 'results/colnomic_full_audit/20261013_resume'


def save(batch, records):
    index = next(r for r in read(FOLDER / 'visual_queue/index.json') if r['batch'] == batch)
    assert len(records) == len(index['queries'])
    image = FOLDER / 'visual_queue' / index['path']
    assert sha(image) == index['sha256']
    target = FOLDER / 'visual_reviews/root_resume' / ('batch_%03d.json' % batch)
    assert not target.exists(), 'Preserve existing review: use an explicit revision file'
    result = []
    for qid, note in zip(index['queries'], records):
        for key in ['object_count', 'capture', 'foreground_occlusion', 'identity_key_region_occluded',
                    'visible_face', 'direct_observation', 'observable_differences', 'unknown']:
            assert key in note, key
        result.append(dict(photo_id=qid, batch=batch, reviewer='root_resume',
            reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
            viewed_image_path=str(image.relative_to(ROOT)), viewed_image_SHA256=index['sha256'],
            review_status='reviewed_visual_overview', fine_detail_verified=False,
            evidence_level='direct_visual_observation', mechanism_claim='unresolved; observation alone does not identify a mechanism',
            **note))
    write(target, result)
    return result
