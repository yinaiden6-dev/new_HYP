import math
import unittest
import sys
from pathlib import Path
import numpy as np
from rc_aslo_xf.h593_gap_net2_v1 import arrays,slope_candidates,scan,fit_direct,row_counts,apply_head
from rc_aslo_xf.h593_s_bias_competition_v1 import calibrate_bias


def brute(rows,beta):
    rs=[r for r in rows if r['m']<=0]
    u=[(float(r['m'])+0.)+float(beta*r['d']) for r in rs]
    candidates={0.}
    for v in u:
        c=math.nextafter(-v,math.inf)
        if math.isfinite(c):candidates.update((c,math.nextafter(c,-math.inf)))
    best=(0,0,0.,0.);answer=[0.,0.,0.,0.,0.,0.]
    for b in candidates:
        delta=[r['delta'] for r,v in zip(rs,u) if v+b>0]
        key=(sum(delta),-len(delta),-abs(b),-b)
        if sum(delta)>0 and key>best:
            best=key;answer=[sum(delta),len(delta),delta.count(1),delta.count(-1),delta.count(0),b]
    return answer


class DirectNetTests(unittest.TestCase):
    def test_independent_full_search_table(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'programs'))
        from validate_rc_h593_gap_net2_v1 import independent_search
        for seed in range(5):
            rng=np.random.default_rng(110+seed)
            rows=[dict(m=float(m),d=float(d),h_s=0.,delta=int(y)) for m,d,y in
                  zip(-rng.uniform(0,4,17),rng.uniform(0,3,17),rng.integers(-1,2,17))]
            m,d,y=arrays(rows);bs=slope_candidates(m,d,.17);table=scan(m,d,y,bs)
            other_bs,other_table=independent_search(rows,.17)
            self.assertTrue(np.array_equal(bs,other_bs))
            self.assertTrue(np.array_equal(table,other_table))

    def test_all_candidates_scalar_threshold_replay(self):
        for seed in range(10):
            rng=np.random.default_rng(seed)
            rows=[dict(m=float(m),d=float(d),h_s=0.,delta=int(y)) for m,d,y in
                  zip(-rng.uniform(0,4,12),rng.uniform(0,3,12),rng.integers(-1,2,12))]
            m,d,y=arrays(rows);bs=slope_candidates(m,d,.25);table=scan(m,d,y,bs,batch_size=13)
            for beta,entry in zip(bs,table):self.assertEqual(entry.tolist(),brute(rows,float(beta)))

    def test_ties_neutral_and_no_positive_fallback(self):
        rows=[dict(m=-1.,d=1.,h_s=0.,delta=v) for v in (1,-1,0)]
        m,d,y=arrays(rows)
        self.assertTrue(np.array_equal(scan(m,d,y,np.array([0.,1.,2.])),np.zeros((3,6))))

    def test_incumbent_inclusion_and_tie_retention(self):
        rows=[dict(m=-1.,d=.5,h_s=0.,delta=1)]
        old=calibrate_bias(rows,0.,.25);head=fit_direct(rows,old)
        self.assertTrue(head['search']['incumbent_retained'])
        for key in ('alpha_hex','beta_hex','bias_hex','disabled'):self.assertEqual(head[key],old[key])

    def test_strict_training_improvement(self):
        rows=[dict(m=-1.,d=2.,h_s=0.,delta=1),dict(m=-.5,d=0.,h_s=0.,delta=-1)]
        old=calibrate_bias(rows,0.,0.);head=fit_direct(rows,old)
        self.assertEqual(old['training_net_gain'],0)
        self.assertEqual(head['training_net_gain'],1)
        self.assertFalse(head['search']['incumbent_retained'])

    def test_bitwise_action_locks(self):
        h=dict(alpha_hex=0.0.hex(),beta_hex=1.0.hex(),bias_hex=1.0.hex(),disabled=False)
        z=[-3.]*127;z[4]=.5
        self.assertEqual([x.hex() for x in apply_head(z,0.,1.,h)],[x.hex() for x in z])
        z[4]=-.5;out=apply_head(z,0.,1.,h)
        self.assertGreater(out[4],0.)
        self.assertTrue(all(out[i].hex()==z[i].hex() for i in range(127) if i!=4))

    def test_empty_and_duplicate_slopes(self):
        rows=[];old=calibrate_bias(rows,0.,0.)
        self.assertTrue(fit_direct(rows,old)['disabled'])
        rows=[dict(m=-1.,d=1.,h_s=0.,delta=1),dict(m=-2.,d=1.,h_s=0.,delta=-1)]
        m,d,y=arrays(rows);self.assertEqual(slope_candidates(m,d,.2).tolist(),[0.,.2,1.])

    def test_crossing_boundaries_fp64(self):
        tiny=math.nextafter(0.,math.inf)
        rows=[dict(m=-tiny,d=0.,h_s=0.,delta=1),dict(m=-1.,d=1.,h_s=0.,delta=-1)]
        m,d,y=arrays(rows);bs=slope_candidates(m,d,.1)
        for beta,entry in zip(bs,scan(m,d,y,bs)):
            self.assertEqual(entry.tolist(),brute(rows,float(beta)))

if __name__=='__main__':unittest.main()
