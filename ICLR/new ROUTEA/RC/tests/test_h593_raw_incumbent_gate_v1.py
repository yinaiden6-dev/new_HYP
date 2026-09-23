"""Signed-margin, matched-control, independent optimum and action contracts."""
import math
from pathlib import Path
import struct
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'programs')]
from rc_aslo_xf import h593_raw_incumbent_gate_v1 as C
from rc_aslo_xf import h593_s_bias_competition_v1 as OLD
from validate_rc_h593_raw_incumbent_gate_v1 import numerical_check
import validate_rc_h593_s_bias_competition_v1 as independent


def bits(x):
    return b''.join(struct.pack('>d', float(v)) for v in x)


class Contracts(unittest.TestCase):
    def test_matched_incumbent(self):
        rng=np.random.default_rng(63)
        rows=[dict(m=-float(rng.random()), h_raw=-float(rng.random()),
                   d=float(rng.random()), delta=int(rng.integers(-1,2))) for _ in range(90)]
        a=C.fit_head(rows,'GAP_BIAS2')
        b=OLD.fit_head([dict(r,h_s=0.) for r in rows],'GAP_BIAS2')
        for k in ('beta_hex','bias_hex','training_net_gain','disabled'):
            self.assertEqual(a[k],b[k])
        self.assertEqual(a['optimization']['theta_hex'],b['optimization']['theta_hex'])

    def test_signed_input_optimizer_independent(self):
        rows=[dict(m=-.5,h_raw=-.05,d=.4,delta=1),
              dict(m=-.5,h_raw=-2.,d=.4,delta=-1)]*12
        for mode in ('GAP_RAWPAIR3','GAP_RAWNEAR3'):
            h=C.fit_head(rows,mode)
            self.assertGreater(h['gamma'],0.)
            numerical_check(rows,'h_raw',h)
            self.assertEqual(h,C.fit_head(rows,mode))

    def test_exact_bias_independent_signed_and_neutral(self):
        rng=np.random.default_rng(57)
        for _ in range(15):
            rows=[dict(m=-float(rng.random()),h_raw=-float(rng.random()),
                       d=float(rng.random()*3),delta=int(rng.integers(-1,2))) for _ in range(25)]
            rows += [dict(rows[0],delta=0),dict(rows[0],delta=-1),dict(m=.1,h_raw=-2.,d=1.,delta=1)]
            h=C.calibrate_bias(rows,.8,.3)
            v=independent.independent_bias([dict(r,h_s=r['h_raw']) for r in rows],.8,.3)
            self.assertEqual(h['bias_hex'],v['bias_hex'])
            self.assertEqual(h['disabled'],v['disabled'])
            self.assertEqual(h['calibration_certificate']['certificates'],v['certificates'])
            for k,val in v['counts'].items():self.assertEqual(h['training_'+k],val)

    def test_original_switch_retained_hold_and_first_tie_bits(self):
        h=dict(gamma=1.,beta=.5,bias=.5,disabled=False)
        z=[-float(i+1) for i in range(127)];z[2]=z[60]=-.1
        out=C.apply_head(z,-.05,0.,h)
        self.assertGreater(out[2],0.)
        self.assertEqual(bits(z[:2]+z[3:]),bits(out[:2]+out[3:]))
        self.assertEqual(bits(z),bits(C.apply_head(z,-2.,0.,h)))
        z[25]=.2;z[26]=-0.
        self.assertEqual(bits(z),bits(C.apply_head(z,-100.,20.,h)))

    def test_raw_input_axis_and_ties(self):
        import torch
        from run_rc_h593_raw_incumbent_gate_v1 import raw_inputs
        x=torch.zeros((127,6),dtype=torch.float64);x[:,0]=-2.;x[80,0]=x[32,0]=-.1
        f=dict(modes=dict(REAL=dict(X=x)))
        v=raw_inputs(f,90)
        self.assertEqual(v,dict(r_pair=-2.,r_near=-.1,raw_nearest_index=32))
        self.assertEqual(raw_inputs(f,80)['r_pair'],v['r_near'])

    def test_gradient(self):
        x=np.array([[-.1,.2,1.],[-2.,.3,1.]])
        m=np.array([-.2,-.4]);y=np.array([1.,-1.]);theta=np.array([.5,.2,.1])
        loss,g=C.surrogate_loss_gradient(theta,m,x,y)
        numeric=[]
        for j in range(3):
            p,n=theta.copy(),theta.copy();p[j]+=1e-6;n[j]-=1e-6
            numeric.append((C.surrogate_loss_gradient(p,m,x,y)[0]-C.surrogate_loss_gradient(n,m,x,y)[0])/2e-6)
        np.testing.assert_allclose(g,numeric,rtol=1e-7,atol=1e-9)

    def test_disabled_and_invalid(self):
        for rows in [[],[dict(m=-1.,h_raw=-.1,d=0.,delta=-1)],
                     [dict(m=-1.,h_raw=-.1,d=0.,delta=0)]]:
            self.assertTrue(C.fit_head(rows,'GAP_RAWPAIR3')['disabled'])
        for raw in [.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):C.fit_head([dict(m=-1.,h_raw=raw,d=1.,delta=1)],'GAP_RAWPAIR3')
        tiny=math.nextafter(0.,1.)
        h=C.calibrate_bias([dict(m=-tiny,h_raw=-tiny,d=0.,delta=1)],1.,0.)
        self.assertGreater(C.gate_score(-tiny,-tiny,0.,h),0.)


if __name__=='__main__':unittest.main()
