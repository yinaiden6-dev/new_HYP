#!/usr/bin/env python3
"""Bounded CPU-only regression for the failed query000/pair059 transform.

No DPT, Slurm, protocol, cache or v1 source is modified. The immutable v1 AST
is evaluated in an isolated namespace with exactly its final combined phase
assertion removed, solely to compare the pre-existing output equation.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "programs"))
OUT = ROOT / "results/rc_m_property_restoration_numeric_v2"


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def checked(record):
    assert binding(record["path"]) == record, "SOURCE_SHA_DRIFT"
    return record["path"]


def write_once(path, value):
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        assert path.read_text() == content, "VALIDATION_IMMUTABILITY"
    else:
        path.write_text(content)


def main():
    import torch
    import rc_roma_position_factor_v1 as original
    import rc_roma_property_restore_v1 as previous
    import rc_roma_property_restore_v2 as repaired

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_path = ROOT / "results/rc_m_property_restoration_factorial_v1/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for record in protocol["code_sources"]:
        checked(record)
    source = Path(original.__file__)
    tree = ast.parse(source.read_text())
    removed = []

    class IsolatedEquation(ast.NodeTransformer):
        def visit_Assert(self, node):
            if ast.unparse(node.test) == "float(relative) < 2e-06 and float(spectral) < 2e-06":
                removed.append(ast.unparse(node))
                return None
            return node

    tree = IsolatedEquation().visit(tree)
    assert len(removed) == 1, "REFERENCE_AST_UNEXPECTED"
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), "<isolated-v1-equation>", "exec"), namespace)
    head_binding = protocol["sources"]["source_head"]
    head = torch.load(checked(head_binding), map_location="cpu", weights_only=True, mmap=True)
    worker = next(row for row in protocol["workers"] if row["index"] == 0)
    source_validation = json.loads(Path(checked(worker["gpu_validation"])).read_text())
    pair = torch.load(checked(source_validation["pairs"][59]), map_location="cpu", weights_only=True, mmap=True)
    capture_binding = pair["capture"]
    capture = torch.load(checked(capture_binding), map_location="cpu", weights_only=True, mmap=True)
    rows = []
    for side in (0, 1):
        info = capture["sides"][side]["P"]
        p = info["tensor"].to(dtype=getattr(torch, info["original_dtype"]))
        radius = protocol["radius"][side]
        omega, scale = head["omega"], head["scale"]
        for arm in repaired.ARMS:
            value, diagnostics = repaired.transform(p, omega, scale, arm, radius, side)
            if "__" in arm:
                base, _ = original.transform(p, omega, scale, "AMP_PERMUTE", radius, side)
                expected, old_diag = namespace["transform"](base, omega, scale, arm.split("__", 1)[1], radius, side)
                old_assert_would_pass = old_diag["norm_relative_error"] < 2e-6 and old_diag["spectral_power_max_abs_error"] < 2e-6
                phase = diagnostics["phase"]
            else:
                expected, old_diag = previous.transform(p, omega, scale, arm, radius, side)
                old_assert_would_pass = True
                phase = diagnostics if diagnostics["kind"] == "phase" else None
            assert torch.equal(value, expected), "TRANSFORMATION_OUTPUT_CHANGED"
            rows.append(dict(side=side, arm=arm, bit_exact_to_v1_equation=True,
                             old_assert_would_pass=old_assert_would_pass,
                             old_diagnostics=old_diag, phase_diagnostics=phase))
    algebra = repaired.self_test()
    OUT.mkdir(parents=True, exist_ok=True)
    write_once(OUT / "algebra.json", algebra)
    joint = [row for row in rows if "__" in row["arm"]]
    result = dict(
        status="PROPERTY_RESTORE_V2_TRANSFORM_ROUNDING_VALIDATION_PASS",
        scope="query000_pair059_original_failed_capture; CPU_transform_only_no_DPT",
        index=0, position=59, side_arm_checks=len(rows), joint_side_arm_checks=len(joint),
        all_outputs_bit_exact_to_immutable_v1_equation=True,
        legacy_10_arms_unchanged=True,
        old_joint_assert_failures=sum(not row["old_assert_would_pass"] for row in joint),
        max_joint_norm_relative_error=max(row["phase_diagnostics"]["norm_relative_error"] for row in joint),
        max_joint_power_abs_error=max(row["phase_diagnostics"]["spectral_power_max_abs_error"] for row in joint),
        max_joint_power_error_over_derived_bound=max(row["phase_diagnostics"]["spectral_power_max_error_over_bound"] for row in joint),
        rounding_derivation="u=eps(dtype)/2; gamma=64*eps64; each Fourier pair uses [gamma+(2u+u^2)(1+gamma)]*Pin plus conservative subnormal terms. Pre-cast FP64 power is checked separately against gamma*Pin. Original norm relative <2e-6 remains.",
        removed_reference_assertion_only=removed,
        input_protocol=binding(protocol_path), capture=capture_binding, head=head_binding,
        code=[binding(Path(__file__)), binding(source), binding(previous.__file__), binding(repaired.__file__)],
        algebra=binding(OUT / "algebra.json"), checks=rows,
        scientific_transform_changed=False, full_dpt_results_not_claimed=True,
    )
    write_once(OUT / "validation.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("checks", "code")}, indent=2))


if __name__ == "__main__":
    main()
