#!/usr/bin/env python3
"""Source-bound C moment inputs with a separate-process independent validator.

All original C scores and native six features must replay byte exactly. The sole
new statistic is a weighted second centered moment of the cached full-reference
visibility-weighted MaxSim vector. No target join or training occurs here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve()
OUT = ROOT / 'results/rc_evidence_dispersion_inputs_v1'
PLAN = ROOT / 'plan/RC_EVIDENCE_DISPERSION_VS_NONLINEAR_READOUT_V1_20260909.md'
SHARED = ROOT / 'results/rc_shared_query_target_prior_cache_v1'
FULL = ROOT / 'results/routea_matched_three_arm_fullnegative_features_v1'
PAIR = ROOT / 'results/routea_matched_three_arm_pair64_training_features_v2/payload.pt'
SCALARS = ('real_score', 'visibility_mass', 'query_control_score', 'reference_control_score')
PINS = {
 'full_feature_validation': (FULL / 'validation.json', '4f6c514b7e7d619d32cf312e06ee9a39494eb7ef4113aaebc45e10b38ce1c827'),
 'pair_payload': (PAIR, 'd7be701ac4629059d22301c17b0f0d44b69b4b67666072b8fab7c3c0cb7716e3'),
 'pair_validation': (PAIR.parent / 'independent_validation.json', '657b316a83b80f20e8779bf985b2777b1533126c9b8e7ec7b552ea245febbf1b'),
 'shared_manifest': (SHARED / 'manifest.json', 'f9f89522a7a3a67a52072e5a32739b4da9f238ff77d63d3b1faf9e342201fc7c'),
 'shared_validation': (SHARED / 'validation.json', '0258615f000d0d82f757f57298b803420395ebc3a35e3a46bfa53c3116b7488f'),
 'frozen_feature_core': (ROOT / 'src/rc_aslo_xf/romav2_colnomic_frozen_gate_v1.py', '96599a560a4066507ae57e641e983b3cf85a18b06ecda0960318174265c981a7'),
}
BOUNDARY = {'theory_name': 'new HYP', 'scope': 'DISPERSION_INPUTS_NO_ACCURACY_OR_SCIENTIFIC_GO',
 'task_training_supervision': 'ORIGINAL_PAIR_QUERY_REFERENCE_IDENTITY_SWITCH_LABELS_ONLY',
 'query_and_reference_tokens': 'ORIGINAL_FROZEN_COLNOMIC_IMAGE_TOKENS',
 'maps_and_local_vector': 'ORIGINAL_ROMA_VISIBILITY_AND_FULL_REFERENCE_WEIGHTED_MAXSIM',
 'dispersion': 'sum(wq*(b-(S/max(M,1e-12)))**2)/clamp_min(sum(wq),1e-12)',
 'extra_contrast': 'symmetric(V_challenger,V_winner)',
 'curvature_control': 'native_feature_column_3_times_its_absolute_value',
 'Q_R_shifts': 'EACH_ORIGINAL_SHIFT_PRESERVED_PAIR_V1_ROLL1_V2_AND_FULL_HALF_AXIS',
 'C_BIND': 'ORIGINAL_WHOLE_EVIDENCE_SOURCE_POSITIONS_INCLUDING_DISPERSION',
 'new_training_updates': 0, 'new_encoder_RoMa_SAM_forwards': 0,
 'evaluation_target_or_outcome_reads': 0, 'candidate_generation_or_insertion': 0,
 'accuracy_or_universal_theory_claimed': False}
BLOCKED = []
def need(value, message):
 if not bool(value): raise RuntimeError(message)
def barrier(event, args):
 if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
 path = Path(os.fsdecode(args[0])).resolve()
 if '/role_shards/' in str(path) or '/rc_opened_eval_strict_' in str(path) or (path.name == 'result.json' and '/results/' in str(path) and path.parent != OUT):
  BLOCKED.append(str(path)); raise RuntimeError('TARGET_OR_OUTCOME_READ_FORBIDDEN')
def sha(path):
 h = hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
 return h.hexdigest()
def bind(path):
 path = Path(path).resolve(); return {'path': str(path), 'sha256': sha(path)}
def path_of(value):
 p = Path(value['path']); p = p if p.is_absolute() else ROOT / p
 need(sha(p) == value['sha256'], 'SOURCE_BINDING_DRIFT:' + str(p)); return p
def read(path): return json.loads(Path(path).read_text())
def encode(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def write(path, value):
 with Path(path).open('xb') as f: f.write(encode(value) + b'\n'); f.flush(); os.fsync(f.fileno())
 Path(path).chmod(0o444)
def tsha(t):
 t = t.detach().cpu().contiguous()
 return hashlib.sha256(str(t.dtype).encode() + encode(list(t.shape)) + t.view(torch.uint8).numpy().tobytes()).hexdigest()
def rawsha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def hexes(v): return [float(x).hex() for x in v]
def hx(e): return {k: float(e[k]).hex() for k in SCALARS}
def sources():
 need(PLAN.is_file(), 'PLAN_REQUIRED')
 result = {'plan': bind(PLAN), 'program': bind(PROGRAM)}
 for name, (p, expected) in PINS.items():
  result[name] = bind(p); need(result[name]['sha256'] == expected, 'SOURCE_PIN:' + name)
 return result

def load_rows():
 validation = read(FULL / 'validation.json')
 need(all(v is True for v in validation['checks'].values()), 'FULL_VALIDATION')
 rows = {}; shards = []
 for entry in validation['shards']:
  p = FULL / ('shard%02d/payload.pt' % entry['shard'])
  need(sha(p) == entry['payload_sha256'], 'FULL_SHARD_SHA'); shards.append(bind(p))
  for row in torch.load(p, map_location='cpu', mmap=True, weights_only=True)['records']:
   key = ('FULL', row['query_id'], int(row['execution_ordinal']))
   need(key not in rows and 'target_position' not in row, 'FULL_TARGET_FREE_UNIQUE'); rows[key] = row
 pv = read(PAIR.parent / 'independent_validation.json')
 need(all(v is True for v in pv['checks'].values()) and pv['v2_payload_sha256'] == PINS['pair_payload'][1], 'PAIR_VALIDATION')
 for row in torch.load(PAIR, map_location='cpu', mmap=True, weights_only=True)['records']:
  key = ('PAIR', row['query_id'], int(row['execution_ordinal']))
  need(key not in rows, 'PAIR_UNIQUE'); rows[key] = row
 need(len(rows) == 128, 'SOURCE_POPULATION')
 return rows, shards

def scalar_replay(w, wm, wrm, b, br, qs):
 shifted = w.roll(qs)
 m = torch.sqrt(w.mean() * wm)
 qm = torch.sqrt(shifted.mean() * wm)
 rm = torch.sqrt(w.mean() * wrm)
 return dict(zip(SCALARS, map(float, (m * (w*b).sum() / w.sum().clamp_min(1e-12), m,
     qm * (shifted*b).sum() / shifted.sum().clamp_min(1e-12), rm * (w*br).sum() / w.sum().clamp_min(1e-12)))))
def independent_scalars(w, wm, wrm, b, br, qs):
 scores = []; masses = []
 for qw, reference_mean, local in ((w, wm, b), (torch.roll(w, shifts=qs), wm, b), (w, wrm, br)):
  m = torch.sqrt(torch.mean(qw) * reference_mean)
  scores.append(float((m * torch.sum(qw * local)) / torch.clamp(torch.sum(qw), min=1e-12)))
  masses.append(float(m))
 return {'real_score': scores[0], 'visibility_mass': masses[0], 'query_control_score': scores[1], 'reference_control_score': scores[2]}
def variance(w, b, evidence):
 center = float(evidence['real_score']) / max(float(evidence['visibility_mass']), 1e-12)
 return float((w * (b - center).square()).sum() / w.sum().clamp_min(1e-12))
def independent_variance(w, b, evidence):
 center = float(evidence['real_score']) / max(float(evidence['visibility_mass']), 1e-12)
 deviation = torch.sub(b, center)
 squared = torch.mul(deviation, deviation)
 return float(torch.sum(torch.mul(w, squared)) / torch.clamp(torch.sum(w), min=1e-12))
def independent_feature(raw, evidence, c, w):
 mu = sum(float(x) for x in raw) / len(raw)
 sd = (sum((float(x)-mu)**2 for x in raw)/len(raw))**.5
 def expanded(e):
  s,m,q,r = (float(e[k]) for k in SCALARS)
  return (s,m,s/max(m,1e-12),s-q,s-r)
 a,b = expanded(evidence[c]),expanded(evidence[w])
 return [(float(raw[c])-float(raw[w]))/max(sd,1e-12)] + [(x-y)/(abs(x)+abs(y)+1e-12) for x,y in zip(a,b)]
def toy(independent):
 fn = independent_variance if independent else variance
 w = torch.ones(2,dtype=torch.float64)
 ev = {'real_score': .5, 'visibility_mass': 1., 'query_control_score': .5, 'reference_control_score': .5}
 a,b = torch.tensor([.25,.75],dtype=torch.float64),torch.tensor([.5,.5],dtype=torch.float64)
 score = independent_scalars if independent else scalar_replay
 one = torch.tensor(1., dtype=torch.float64)
 first, second = score(w,one,one,a,a,1), score(w,one,one,b,b,1)
 need(hx(first) == hx(second) == hx(ev), 'E0_FOUR_ORIGINAL_SCALARS_IDENTICAL')
 va,vb = fn(w,a,first),fn(w,b,second)
 need(va == .0625 and vb == 0 and float(a.mean()) == float(b.mean()) == .5, 'E0_EQUAL_MEAN_DISTINCT_SECOND_MOMENT')
 return {'status': 'E0_INFORMATION_WITNESS_PASS_NOT_ACCURACY', 'original_four_scalars_binary64': hx(ev),
  'uniform_query_and_reference_weights': True, 'first_local_vector_binary64': hexes(a), 'second_local_vector_binary64': hexes(b),
  'first_dispersion_binary64': va.hex(), 'second_dispersion_binary64': vb.hex(),
  'same_S_M_and_Q_R_means_different_second_moment': True, 'accuracy_claimed': False}

def recompute(independent=False):
 from rc_aslo_xf.romav2_colnomic_frozen_gate_v1 import candidate_feature, symmetric
 feature = independent_feature if independent else candidate_feature
 var = independent_variance if independent else variance
 score = independent_scalars if independent else scalar_replay
 rows, shards = load_rows()
 manifest, validation = read(SHARED / 'manifest.json'), read(SHARED / 'validation.json')
 need(validation['manifest_sha256'] == PINS['shared_manifest'][1] and all(v is True for v in validation['checks'].values()), 'QUALIFIED_SHARED_CACHE')
 records = []; counts = Counter()
 for entry in manifest['records']:
  mp = path_of(entry); meta = read(mp)
  key = (meta['kind'], meta['query_id'], int(meta['execution_ordinal'])); row = rows[key]
  need(meta['axis'] == row['candidate_physical_rows'] and meta['winner'] == row['base_winner_position'], 'C128_AXIS_AND_WINNER')
  need(meta.get('challenger_positions', [i for i in range(128) if i != meta['winner']]) == row['challenger_positions'], 'CHALLENGER_ORDER')
  ap = path_of(meta['arrays'])
  with np.load(ap, allow_pickle=False) as pack:
   arrays = {k: torch.from_numpy(pack[k].copy()) for k in ('q_tokens','wq','wr_mean','wr_rolled_mean','b','br')}
   positions = pack['candidate_positions'].tolist(); raw = pack['raw_scores'].tolist()
   qshift,rshift = pack['qshift'].tolist(),pack['rshift'].tolist()
  q,weights,b_all,br_all = (arrays[k] for k in ('q_tokens','wq','b','br'))
  need(q.dtype == torch.float16 and q.shape[1] == 128 and tsha(q) == meta['q_tokens_sha256'], 'ORIGINAL_QUERY_TOKEN_SHA')
  need(weights.dtype == b_all.dtype == br_all.dtype == torch.float64 and weights.shape == b_all.shape == br_all.shape, 'FP64_LOCAL_VECTOR_AXIS')
  need(weights.shape[1] == len(q) and positions == [c['candidate_position'] for c in meta['candidates']], 'CANDIDATE_QUERY_AXES')
  need(hexes(raw) == hexes(row['base_scores']) and len(raw) == 128, 'RAW_SCORE_BITS')
  if key[0] == 'PAIR': need(meta['pair_cohort'] == row['pair_cohort'] and meta['switch_label'] == row['switch_label'], 'PAIR_TRAIN_SUPERVISION')
  ev,vs,candidates = {},{},[]
  for index,c in enumerate(meta['candidates']):
   pos,physical = int(c['candidate_position']),int(c['physical_row'])
   need(meta['axis'][pos] == physical, 'REFERENCE_PHYSICAL_AXIS')
   w,b,br = weights[index],b_all[index],br_all[index]
   need(bool(torch.isfinite(w).all()) and bool((w >= 0).all()) and bool(torch.isfinite(b).all()) and bool(torch.isfinite(br).all()), 'LOCAL_FINITE_NONNEGATIVE_WEIGHT')
   qs,rs = int(qshift[index]),int(rshift[index])
   expected = (1,1) if key[0] == 'PAIR' and row['pair_cohort'] == 'BALANCED32_V1' else (max(1,len(w)//2),max(1,c['reference_token_count']//2))
   need((qs,rs) == expected and rawsha(w) == c['query_map_sha256'], 'ORIGINAL_MAP_AND_CONTROL_SHIFT')
   if key[0] == 'PAIR':
    fixed = row['fixed_candidate_maps'][pos]
    need(rawsha(w) == fixed['query_map_sha256'] and rawsha(fixed['reference_map']) == c['reference_map_sha256'], 'PAIR_ORIGINAL_MAP_SHAS')
    need(c['reference_tokens_sha256'] == fixed['reference_tokens_sha256'] and tsha(q) == fixed['query_tokens_sha256'], 'PAIR_ORIGINAL_TOKEN_SHAS')
   e = score(w,arrays['wr_mean'][index],arrays['wr_rolled_mean'][index],b,br,qs)
   need(hx(e) == c['old_scalars_binary64'] == hx(row['evidence']['C_PAIRED'][pos]), 'ORIGINAL_C4_BINARY64_REPLAY:' + key[1] + ':' + str(pos))
   v = var(w,b,e); need(np.isfinite(v) and v >= 0, 'FINITE_NONNEGATIVE_DISPERSION')
   ev[pos],vs[pos] = e,v
   candidates.append({'candidate_position': pos, 'physical_row': physical,
    'original_query_control_shift': qs, 'original_reference_control_shift': rs,
    'reference_tokens_sha256': c['reference_tokens_sha256'], 'reference_token_count': c['reference_token_count'],
    'query_map_sha256': c['query_map_sha256'], 'reference_map_sha256': c['reference_map_sha256'],
    'b_sha256': tsha(b), 'br_sha256': tsha(br), 'wq_sha256': tsha(w),
    'center_L_binary64': (e['real_score']/max(e['visibility_mass'],1e-12)).hex(), 'dispersion_binary64': v.hex()})
   counts[key[0] + '_candidate_occurrences'] += 1; counts['original_C4_scalar_bit_exact_checks'] += 4
  features = {}
  for mode in (('REAL','CBIND') if key[0] == 'FULL' else ('REAL',)):
   mapping = {i: int(row['cbind_source_positions'][i]) for i in ev} if mode == 'CBIND' else {i:i for i in ev}
   need(set(mapping.values()) == set(ev), 'WHOLE_EVIDENCE_SOURCE_PERMUTATION')
   e = {i:ev[mapping[i]] for i in ev}; v = {i:vs[mapping[i]] for i in ev}
   native = [list(map(float,feature(raw,e,c,row['base_winner_position']))) for c in row['challenger_positions']]
   old = row[('real_' if mode == 'REAL' else 'cbind_') + 'native_features']['C_PAIRED']
   need([hexes(f) for f in native] == [hexes(f) for f in old], 'ORIGINAL_NATIVE6_BINARY64_REPLAY:' + mode)
   extra = []
   for c in row['challenger_positions']:
    vc,vw = v[c],v[row['base_winner_position']]
    extra.append((vc-vw)/(abs(vc)+abs(vw)+1e-12) if independent else symmetric(vc,vw))
   features[mode] = {'ORIGINAL_C': [hexes(f) for f in native],
    'DISPERSION': [hexes(f+[z]) for f,z in zip(native,extra)],
    'CURVE': [hexes(f+[f[3]*abs(f[3])]) for f in native]}
   counts['original_native6_bit_exact_checks'] += len(native)*6
   counts['new_seventh_feature_checks'] += len(native)*2
  out = {'kind':key[0], 'query_id':key[1], 'execution_ordinal':key[2],
   'candidate_physical_rows':meta['axis'], 'base_winner_position':row['base_winner_position'],
   'challenger_positions':row['challenger_positions'], 'base_scores_binary64':hexes(raw),
   'evidence_binary64':{str(i):hx(e) for i,e in ev.items()}, 'dispersion_binary64':{str(i):v.hex() for i,v in vs.items()},
   'feature_binary64':features, 'candidates':candidates, 'query_tokens_sha256':tsha(q),
   'source':{'metadata':bind(mp),'arrays':bind(ap)}}
  if key[0] == 'PAIR': out.update(pair_cohort=row['pair_cohort'], pair_row_ordinal=row['pair_row_ordinal'], switch_label=bool(row['switch_label']))
  else: out['cbind_source_positions'] = row['cbind_source_positions']
  records.append(out); counts[key[0]+'_query_count'] += 1
 need(counts['FULL_query_count'] == counts['PAIR_query_count'] == 64 and counts['FULL_candidate_occurrences'] == 8192 and counts['PAIR_candidate_occurrences'] == 128, 'FINAL_POPULATION')
 need(not BLOCKED, 'FORBIDDEN_READ_ATTEMPT')
 return {'records':records,'counts':dict(counts),'source_shards':shards,'E0':toy(independent)}

def validate_child(nonce):
 need(nonce == os.environ.get('RC_DISPERSION_VALIDATOR_NONCE') and str(os.getppid()) == os.environ.get('RC_DISPERSION_VALIDATOR_PARENT_PID'), 'EXPLICIT_SUBPROCESS_REQUIRED')
 need(not (OUT / 'validation.json').exists(), 'APPEND_ONLY_VALIDATION')
 sealed = sources(); manifest = read(OUT / 'manifest.json'); result = read(OUT / 'result.json')
 need(manifest['sources'] == result['sources'] == sealed and result['boundary'] == BOUNDARY, 'SOURCE_PROVENANCE')
 need(manifest['result'] == bind(OUT / 'result.json'), 'RESULT_SHA')
 replay = recompute(independent=True)
 for k in replay: need(replay[k] == result[k], 'INDEPENDENT_REPLAY:' + k)
 need(bind(PLAN) == sealed['plan'] and bind(PROGRAM) == sealed['program'], 'SOURCES_CHANGED_DURING_VALIDATION')
 value = {'status':'RC_EVIDENCE_DISPERSION_INPUTS_V1_INDEPENDENT_VALIDATION_PASS', 'sources':sealed,
  'manifest':bind(OUT / 'manifest.json'), 'result':bind(OUT / 'result.json'), 'counts':replay['counts'], 'boundary':BOUNDARY,
  'fresh_explicit_subprocess':True, 'validator_pid':os.getpid(), 'validator_parent_pid':os.getppid(),
  'independent_scalar_feature_and_centered_second_moment_implementations':True,
  'qualified_cache_original_token_and_map_vector_validation_reused':True, 'E0':replay['E0']}
 write(OUT / 'validation.json',value)
 print(json.dumps({'status':value['status'],'counts':value['counts'],'validation':bind(OUT / 'validation.json')}),flush=True)

def main():
 parser = argparse.ArgumentParser(description=__doc__); mode = parser.add_mutually_exclusive_group(required=True)
 mode.add_argument('--produce',action='store_true'); mode.add_argument('--validate',action='store_true'); mode.add_argument('--validator-child')
 args = parser.parse_args(); torch.set_num_threads(8); torch.set_num_interop_threads(1)
 sys.path.insert(0,str(ROOT / 'src')); sys.addaudithook(barrier)
 if args.validate:
  nonce = uuid.uuid4().hex; env = dict(os.environ, RC_DISPERSION_VALIDATOR_NONCE=nonce, RC_DISPERSION_VALIDATOR_PARENT_PID=str(os.getpid()))
  subprocess.run([sys.executable,str(PROGRAM),'--validator-child',nonce],check=True,env=env)
 elif args.validator_child:
  with torch.inference_mode(): validate_child(args.validator_child)
 else:
  need(not OUT.exists(), 'APPEND_ONLY_OUTPUT_EXISTS'); sealed = sources()
  with torch.inference_mode(): replay = recompute()
  need(bind(PLAN) == sealed['plan'] and bind(PROGRAM) == sealed['program'], 'SOURCES_CHANGED_DURING_PRODUCTION')
  OUT.mkdir(); result = {'status':'RC_EVIDENCE_DISPERSION_INPUTS_V1_REPLAY_PASS','sources':sealed,'boundary':BOUNDARY,**replay}
  write(OUT / 'result.json',result)
  write(OUT / 'manifest.json',{'status':'RC_EVIDENCE_DISPERSION_INPUTS_V1_PRODUCED','sources':sealed,'result':bind(OUT / 'result.json'),'counts':replay['counts'],'producer_pid':os.getpid(),'boundary':BOUNDARY})
  print(json.dumps({'status':result['status'],'counts':replay['counts'],'result':bind(OUT / 'result.json')}),flush=True)
if __name__ == '__main__': main()
