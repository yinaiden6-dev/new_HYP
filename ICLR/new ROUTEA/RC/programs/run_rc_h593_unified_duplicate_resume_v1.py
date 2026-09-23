#!/usr/bin/env python3
"""Resume query70; recompute native matcher when internal hooks need execution."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'programs'),str(ROOT/'src')]
import run_rc_h593_unified_acquisition_v1 as R

AUTH=ROOT/'registry/rc_h593_unified_duplicate_resume_authority_v1_20260923.json'


def main():
    a=R.read(AUTH)
    for b in a['sources']:R.checked(b)
    index=int(os.environ['SLURM_ARRAY_TASK_ID']);assert index==70
    original=R.G.JointObserver.memo_matcher
    count=[0]
    def observed_matcher(self,*args,**kwargs):
        key=R.V.tags(self.arm)
        if self.want_inside and self.cache_identity==self.identity and key in self.pair_cache:
            cached=self.pair_cache[key]
            assert R.equal_tree((args,kwargs),cached['inputs']),'JOINT_MATCHER_INPUT_DRIFT'
            # Cached native outputs cannot replay internal head forward hooks.
            # Re-execute the unchanged matcher; keep descriptors and all outputs.
            del self.pair_cache[key]
            count[0]+=1
        return original(self,*args,**kwargs)
    R.G.JointObserver.memo_matcher=observed_matcher
    try:
        R.collect(index)
    finally:
        R.write(R.OUT/'query070/repair_execution'/f"{os.environ['SLURM_JOB_ID']}.json",dict(
            authority=R.bind(AUTH),original_acquisition_authority=R.bind(R.AUTH),
            cache_hits_recomputed_for_internal_hooks=count[0],index=70,
            science_unchanged=True,existing_capsules_preserved=True))


if __name__=='__main__':main()
