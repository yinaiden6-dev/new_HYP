#!/usr/bin/env python3
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"programs"));import materialize_romav2_colnomic_visibility_xf_full_c128_shard_v1 as base
base.SELECTION=(307,120,5,306,327,586,329,0,194,502,489,21,200,588,27,488,495,299,414,340,449,193,599,113,122,441,300,6,598,17,1,192);base.HELDOUT_FOLD=2;base.SELECTION_NAMESPACE="ROMA_COLXF_FULLNEG_EVAL_V1";base.OUTROOT=ROOT/"results/romav2_colnomic_visibility_xf_fullnegative_eval_prejoin_v1"
used={480,20,421,324,485,413}
for p in (ROOT/"results/rgh_v9b_balanced_fold2_oof_pilot_v1/result.json",ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v1/result.json",ROOT/"results/romav2_colnomic_visibility_xf_balanced32_v2/result.json",ROOT/"results/romav2_colnomic_visibility_xf_full_c128_gate_v1/result.json"):
 x=json.loads(p.read_text())
 for key in ("evaluation_executions","selection","rows"):
  for item in x.get(key,[]):used.add(int(item if isinstance(item,int) else item["execution_ordinal"]))
base.EXCLUDE_EXTRA=used
if __name__=="__main__":base.main()

