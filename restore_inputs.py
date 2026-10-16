from pathlib import Path
import numpy as np,gzip,hashlib,json
B=Path(__file__).resolve().parent
z=np.load(B/'inputs_masks_compact.npz');q={};r={}
for i,x in enumerate(z['query_ids']):
 for f in ['query_indices','visual','foreground_center','template','box_overlap_fraction']:q[str(x)+'__'+f]=z[f][z['query_offsets'][i]:z['query_offsets'][i+1]]
for i,x in enumerate(z['reference_ids']):r[str(x)]=z['reference_visual'][z['reference_offsets'][i]:z['reference_offsets'][i+1]]
np.savez_compressed(B/'masks.npz',**q);np.savez_compressed(B/'reference_visual_masks.npz',**r)
(B/'cohort.json').write_bytes(gzip.decompress((B/'inputs_cohort.json.gz').read_bytes()))
print('Restored1117 query masks,5405 reference masks and cohort; compare array identity if ZIP metadata changes.')
