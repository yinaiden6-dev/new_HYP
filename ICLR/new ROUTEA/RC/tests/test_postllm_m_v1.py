import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'programs'))
import torch
import run_rc_postllm_m_v1 as N
import run_rc_internal_m_condition_scale_v4 as V
from rc_prellm_m_scaled_adapter_v4 import ScaledQualityResidualAdapter


class PostTests(unittest.TestCase):
    def test_unchanged_training_and_resume_algorithm(self):
        # A source-level equality check guards accidental budget/loss/resume edits.
        for name in ('score_all', 'evaluate_endpoint', 'finish_pilot', 'train'):
            expected=inspect.getsource(getattr(V,name)).replace("'PRE_REAL'", "'POST_REAL'").replace("'PRE_CONSTANT'", "'POST_CONSTANT'").replace("'PRE_SHUFFLED'", "'POST_SHUFFLED'")
            expected=expected.replace("source=cache['merged']", "source=cache['hidden'][cache['image_mask']]").replace("'prellm_residual_max'", "'postllm_residual_max'")
            self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(getattr(N,name)))))

    def test_constant_and_real_initialization(self):
        torch.manual_seed(17)
        real=ScaledQualityResidualAdapter(hidden_size=12,bottleneck=3,condition_gain=12**.5)
        const=ScaledQualityResidualAdapter(hidden_size=12,bottleneck=3,condition_gain=12**.5,conditioning='constant')
        const.load_state_dict(real.state_dict())
        x=torch.randn(7,12).to(torch.bfloat16)
        self.assertTrue(torch.equal(real(x,.2),x));self.assertTrue(torch.equal(const(x,.8),x))
        self.assertTrue(torch.equal(const.standardized_mass(.1,x),const.standardized_mass(.9,x)))

    def test_no_head_M_and_full_C128_gradient(self):
        r=dict(raw_scores=list(range(128)),winner_index=127,challenger_positions=list(range(127)),target_positions=[3],M=[.5]*128)
        design=torch.randn(128,4,dtype=torch.float64);h=torch.randn(4,dtype=torch.float64,requires_grad=True)
        theta=torch.tensor([-.1,25.,-.5],dtype=torch.float64,requires_grad=True)
        content=(design@h).sigmoid();loss=V.loss(r,content,theta)
        dh,dt=torch.autograd.grad(loss,(h,theta),retain_graph=True)
        _,dl,dhead=V.joint_derivatives(r,content,theta)
        replay=sum(content[j]*dl[j] for j in dl.nonzero().flatten().tolist())
        self.assertTrue(torch.allclose(torch.autograd.grad(replay,h)[0],dh,atol=1e-11,rtol=1e-11))
        self.assertTrue(torch.allclose(dt,dhead,atol=1e-11,rtol=1e-11))
        x=V.features(r,content);r['M']=[.1]*128
        self.assertTrue(torch.equal(x,V.features(r,content)))

    def test_launcher_shell_success_failure_requeue(self):
        source=(ROOT/'slurm/rc_postllm_m_v1.sbatch').read_text()
        subprocess.run(['bash','-n'],input=source,text=True,check=True)
        block=source[source.index('stage="$1"'):]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);events=root/'events'
            # Execute the exact launcher block with mock external commands.
            prelude='''set -euo pipefail
timeout() { return "$MOCK_STATUS"; }
scontrol() { printf '%s\\n' "$*" >> "$EVENTS"; }
PY=mock_python
mock_python() { printf '%s\\n' "$*" >> "$EVENTS"; }
'''
            for stage,status,restart,expected in [('pilot',0,0,0),('worker',0,0,0),('pilot',75,0,0),('worker',124,0,0),('pilot',1,0,1),('worker',75,48,75)]:
                events.write_text('')
                env=dict(os.environ,MOCK_STATUS=str(status),SLURM_RESTART_COUNT=str(restart),SLURM_JOB_ID='123',EVENTS=str(events))
                p=subprocess.run(['bash','-c',prelude+block,'mock',stage,'POST_REAL'],env=env,capture_output=True,text=True)
                self.assertEqual(p.returncode,expected,p.stderr)
                output=events.read_text()
                self.assertEqual('advance' in output,stage=='pilot' and status==0)
                self.assertEqual('requeue 123' in output,status in (75,124) and restart<48)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
