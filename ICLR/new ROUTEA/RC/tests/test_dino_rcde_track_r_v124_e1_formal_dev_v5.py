from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
def load(p,n):
 s=importlib.util.spec_from_file_location(n,ROOT/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_formal_candidate():
 f=load("programs/freeze_dino_rcde_track_r_v124_e1_formal_dev_authority_v5.py","f5");v=f.build();f.A.validate_candidate(v);assert v["eligible_as_e1_result"] is True
def test_resources_and_checkpoint():
 p=(ROOT/"slurm/dino_rcde_track_r_v124_e1_formal_dev_producer_v5.sbatch").read_text();v=(ROOT/"slurm/dino_rcde_track_r_v124_e1_formal_dev_validator_v5.sbatch").read_text();assert all("#SBATCH --time=01:00:00" in x and "#SBATCH --partition=dev_accelerated" in x for x in (p,v));assert "family_commit.json" in v
def test_formal_statuses():
 p=load("programs/run_dino_rcde_track_r_v124_e1_formal_dev_producer_v5.py","p5");v=load("programs/validate_dino_rcde_track_r_v124_e1_formal_dev_independent_v5.py","v5");assert p.STATUS.endswith("PREJOIN_PASS");assert v.STATUS.endswith("INDEPENDENT_VALIDATION_PASS")
