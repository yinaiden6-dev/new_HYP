"""New OCR on five reviewed-subject crops; selected-case diagnostic only."""
from pathlib import Path
import os,sys,json,hashlib,time
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
D=Path(__file__).resolve().parent;V=D.parent;P=V.parent.parent;R=Path('/mnt/lsdf_zyin')
sys.path[:0]=[str(P),str(R)]
from evaluation.ocr import OCR
from evaluation.regions import open_photo,crop_region
from experiment_utils import read,write
protocol=read(D/'protocol.json');qo=read(R/'cache/ocr/photos.json');policy=read(P/'SUBJECT_REGION_POLICY.json');boxes=read(Path(policy['remote_source']))['records']
(D/'ocr_inputs').mkdir(exist_ok=True);result={};engine=OCR(R,'gpu:0')
for q in protocol['identifier_diagnostics']:
 rec=boxes[q];assert rec['source']=='human' and rec['status']=='confirmed'
 im=open_photo(R/qo[q]['image_path']);crop,bounds=crop_region(im,rec['bbox'],margin=.05);path=D/'ocr_inputs'/f'{q}.png';crop.save(path)
 entry,seconds=engine.recognize(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),path)
 result[q]=dict(ocr=entry,seconds=seconds,bounds=bounds,scope='Five selected-case human-subject crops; automatic OCR, not a population-wide deployment evaluation.')
 write(D/'rerecognized.json',result);print(q,entry['text'],seconds,flush=True)
