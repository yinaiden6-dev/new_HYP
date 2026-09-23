from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_candidate_and_parent_science_are_fixed():
    freezer=load("programs/freeze_dino_rcde_track_r_v124_e1_dev_staged_authority_v4.py","freeze_v4_test")
    value=freezer.build(); freezer.A.validate_candidate(value)
    assert value["eligible_as_e1_result"] is False
    assert value["resource_contract"]["walltime_seconds_per_stage"]==3600

def test_launchers_are_one_hour_and_separated():
    p=(ROOT/"slurm/dino_rcde_track_r_v124_e1_dev_staged_producer_v4.sbatch").read_text()
    v=(ROOT/"slurm/dino_rcde_track_r_v124_e1_dev_staged_validator_v4.sbatch").read_text()
    assert all("#SBATCH --partition=dev_accelerated" in x and "#SBATCH --time=01:00:00" in x for x in (p,v))
    assert "run_dino_rcde_track_r_v124_e1_dev_staged_producer_v4.py" in p
    assert "validate_dino_rcde_track_r_v124_e1_dev_staged_independent_v4.py" in v
    assert "family_commit.json" in v

def test_wrappers_keep_nonpromotable_boundary():
    p=load("programs/run_dino_rcde_track_r_v124_e1_dev_staged_producer_v4.py","p_v4_test")
    v=load("programs/validate_dino_rcde_track_r_v124_e1_dev_staged_independent_v4.py","v_v4_test")
    assert "NONPROMOTABLE" in p.STATUS and "NONPROMOTABLE" in v.STATUS
    assert p.AUTH.PRODUCER_NAMESPACE != p.AUTH.PARENT.PRODUCER_NAMESPACE
