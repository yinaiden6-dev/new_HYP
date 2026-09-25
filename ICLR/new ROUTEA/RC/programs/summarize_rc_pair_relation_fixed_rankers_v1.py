#!/usr/bin/env python3
"""Fixed-argmax diagnostic of existing held119 score vectors; no head fit."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rc_pair_relation_localization_v1/appearance_controls'


def bind(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def main():
    source=OUT/'pair_trace_recount.json';r=json.loads(source.read_text())
    originalp=ROOT/'results/rc_pair_quality_fold0_analysis_v1/result.json'
    original=json.loads(originalp.read_text());old={x['query_id']:x for x in original['rows']}
    summary={};details=[]
    for name in r['summary']:
        count={k:0 for k in ['correct','rescues_vs_RAW','breaks_vs_RAW','strict_target_first',
            'target_beats_RAWwinner_on_inpool_RAW_errors','target_beats_RAWwinner_on_original9rescues',
            'target_strict_first_on_original9rescues','target_ARGMAX_first_on_original9rescues']}
        corr=[];gain=[];lost=[]
        for row in r['rows']:
            q=row['query_id'];s=np.asarray(row['scores'][name]);pos=int(s.argmax());t=row['target_position'];w=row['winner']
            correct=t is not None and pos==t;raw=old[q]['models']['RAW']['correct']
            strict=t is not None and s[t]>np.delete(s,t).max()
            count['correct']+=int(correct);count['rescues_vs_RAW']+=int(correct and not raw)
            count['breaks_vs_RAW']+=int(raw and not correct);count['strict_target_first']+=int(strict)
            if t is not None and not raw:count['target_beats_RAWwinner_on_inpool_RAW_errors']+=int(s[t]>s[w])
            if row['original_rescue']:
                count['target_beats_RAWwinner_on_original9rescues']+=int(s[t]>s[w])
                count['target_strict_first_on_original9rescues']+=int(strict)
                count['target_ARGMAX_first_on_original9rescues']+=int(correct)
            if correct:corr.append(q)
            if correct and not raw:gain.append(q)
            if raw and not correct:lost.append(q)
            details.append(dict(query_id=q,statistic=name,target_position=t,selected_position=pos,
                selected_physical_row=old[q]['candidate_physical_rows'][pos],raw_winner_position=w,
                correct=correct,raw_correct=raw,score_target=None if t is None else float(s[t]),
                score_RAWwinner=float(s[w]),score_maximum=float(s[pos]),
                max_tie_count=int((s==s.max()).sum()),original_rescue=row['original_rescue']))
        summary[name]={**count,'n':119,'inpool':113,'inpool_RAW_errors':15,'original_rescues':9,
                       'correct_query_ids':corr,'rescued_query_ids':gain,'broken_query_ids':lost}
    result=dict(status='FIXED_119_C128_RANKER_RECOUNT_COMPLETE',sources=[bind(__file__),bind(source),bind(originalp)],
        protocol='No fit, threshold or HOLD/SWITCH. Every candidate scored; choose argmax in original physical C128 order. All119 held retained including6 missing targets. Opened first-fold diagnostic.',
        original_RAW_correct=98,summary=summary,rows=details,
        caveat='Different direct rankers are diagnostic mappings, not learned action heads or verified replacements. No held-based parameter selection.')
    (OUT/'candidate_rank_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if not kk.endswith('_ids')} for k,v in summary.items()},indent=2))


if __name__=='__main__':main()
