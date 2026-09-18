#!/usr/bin/env python3
"""Close READY sources after qualified TRAIN128 inputs, then run the frozen readout launcher."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];PROGRAM=Path(__file__).resolve()
READOUT=ROOT/'programs/run_rc_original7_train128_readout_v1.py'
LAUNCH=ROOT/'slurm/rc_original7_train128_readout_v1_dev_cpuonly_59m.sbatch'
SELF_LAUNCH=ROOT/'slurm/rc_original7_train128_readout_prepare_and_run_v1_dev_cpuonly_59m.sbatch'
PLAN=ROOT/'plan/RC_ORIGINAL7_TRAIN96_TRAIN128_FIXED_RECIPE_READOUT_V1_20260910.md'
INPUT_AUTH=ROOT/'registry/rc_original7_train128_execution_authority_v1_20260910.json'
AUTH=ROOT/'registry/rc_original7_train128_readout_authority_v1_20260910.json'
RESEARCH=ROOT/'registry/rc_retrieval_only_research_extension_authority_v1_20260909.json'
PINS={READOUT:'88b981c1394e2948c19d4ff2728eee7976654eb646be38bc6a862a962fcb7af7',
LAUNCH:'d36beedebe7bed416d3eb9b5bfbfdb69d345cd6aaab2a31a80646b19ed6f7420',
PLAN:'716e590d220799b8af12f0e0b104bf0a29b612b149f13d940b65021cfb3b055c',
INPUT_AUTH:'34cfddfe7840e67388d627b0e6a312621903030de6957226f56f7d68b1d76cbb',
RESEARCH:'b9bdcd4b1c847fc22755a2e5565f30a4e77f3bda4f92e4273769601caf10ceda'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def need(v,m):
 if not bool(v):raise RuntimeError(m)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-preflight',action='store_true');args=parser.parse_args()
 for p,h in PINS.items():need(sha(p)==h,'FROZEN_PREPARATION_SOURCE:'+p.name)
 need(datetime.now(timezone.utc)<datetime(2026,9,11,16,tzinfo=timezone.utc),'USER_RESEARCH_DEADLINE')
 if args.source_preflight:
  print(json.dumps({'status':'TRAIN128_READOUT_PREPARATION_SOURCE_PREFLIGHT_PASS','program':bind(PROGRAM),'pins':{p.name:bind(p) for p in PINS},'natural_training_or_feature_reads':0}),flush=True);return
 need(bool(os.environ.get('SLURM_JOB_ID')),'REAL_SLURM_ALLOCATION_REQUIRED')
 # The readout preflight hashes sources and validates metadata, but never decodes training features or EVAL labels.
 proc=subprocess.run([sys.executable,str(READOUT),'--phase','preflight'],capture_output=True,text=True,check=True)
 for line in proc.stdout.splitlines():print(line,flush=True)
 info=json.loads(proc.stdout.strip().splitlines()[-1]);need(info['status']=='TRAIN128_READOUT_PREFLIGHT_READY','QUALIFIED_TRAIN_INPUTS_NOT_READY')
 prepath=Path(info['preflight']['path']);need(bind(prepath)==info['preflight'],'READY_PREFLIGHT_BINDING');pre=json.loads(prepath.read_text())
 need(pre['natural_optimizer_updates']==pre['natural_feature_payload_reads']==pre['EVAL_role_outcome_reads']==pre['forbidden_read_attempts']==0,'PURE_SOURCE_CLOSURE')
 source=pre['sources'];need(source['program']==bind(READOUT) and source['plan']==bind(PLAN) and source['launcher']==bind(LAUNCH) and source['train_input_authority']==bind(INPUT_AUTH),'FROZEN_RECIPE_AND_INPUT_AUTHORITY')
 vp=Path(source['train_input_validation']['path']);need(bind(vp)==source['train_input_validation'],'INPUT_VALIDATION_BINDING');v=json.loads(vp.read_text())
 need(v['status']=='ORIGINAL7_TRAIN128_INPUTS_AND_FEATURES_REPLAY_PASS' and v['query_count']==128 and v['independent_C4_scalar_checks']==65536 and v['all32_original_input_bits_exact'] and v['original_FULL32_token_RAW_map_C4_parity_count']==32,'ALL128_AND_ORIGINAL32_QUALIFIED')
 need(v['authority']==bind(INPUT_AUTH) and v['feature_records']==source['train_input_features'] and v['curator_reads']==v['EVAL_result_reads']==v['training_updates']==v['forbidden_read_attempts']==0,'INPUT_SEAL_AND_BOUNDARY')
 value={'status':'ORIGINAL7_TRAIN128_READOUT_AUTHORIZED','source_bindings':pre['sources'],'postjoin_source_bindings':pre['postjoin_sources'],
  'preflight':bind(prepath),'contract':pre['contract'],'allowed_phases':['fit','validate-fit','predict','validate-predictions','join','validate-result'],
  'cutoff_UTC':'2026-09-11T16:00:00+00:00','parent_research_authority':bind(RESEARCH),'parent_TRAIN_input_authority':bind(INPUT_AUTH),
  'preparation_program':bind(PROGRAM),'preparation_launcher':bind(SELF_LAUNCH),'source_closure_policy':'Only predeclared source files from the frozen readout are bound after input qualification; no parameter/outcome selection',
  'new_task_spatial_supervision':False,'EVAL_used_for_training':False,'deployment_replacement_authorized':False}
 data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 if AUTH.exists():need(AUTH.read_bytes()==data,'IMMUTABLE_READOUT_AUTHORITY_DRIFT')
 else:
  with AUTH.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  AUTH.chmod(0o444)
 print(json.dumps({'status':'TRAIN128_READY_AUTHORITY_FROZEN','authority':bind(AUTH)}),flush=True)
 # Recheck the entrypoints immediately before running their fixed fit -> predict -> join sequence.
 for p,h in PINS.items():need(sha(p)==h,'SOURCE_CHANGED_AFTER_AUTHORITY:'+p.name)
 subprocess.run(['/bin/bash',str(LAUNCH)],check=True)
if __name__=='__main__':main()
