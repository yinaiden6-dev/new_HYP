"""Frozen property interventions with a scale-aware joint-phase rounding check.

The first ten arms still call the immutable position-factor v1 transform.
Joint arms use the identical FP64 rotation followed by the identical cast.
Only the joint phase invariant check changes: FP32 power rounding error scales
with each sine/cosine pair's power after amplitude reassignment.
"""
from __future__ import annotations

import torch

import rc_roma_position_factor_v1 as original
import rc_roma_property_restore_v1 as previous

SEED, DELTA = previous.SEED, previous.DELTA
PANEL, PHASE_ARMS, ARMS = previous.PANEL, previous.PHASE_ARMS, previous.ARMS
FP64_POWER_RELATIVE_ALLOWANCE = 64 * torch.finfo(torch.float64).eps


def _phase_invariants(base, rotated64, output):
    """Check the FP64 rotation and the independently derived cast error bound.

    For an exact FP64 component z cast to the output format, round-to-nearest
    gives |fl(z)-z| <= u|z| + a, u=eps/2. We conservatively use one smallest
    subnormal for a (rather than half). For a pair with power P64 this gives
      |Pout-P64| <= (2u+u^2)P64
                    + 2a(1+u)(|z_s|+|z_c|) + 2a^2.
    The pre-cast rotation is checked separately against gamma*Pin, with
    gamma=64*eps64 covering FP64 trig, multiply/add and power evaluation.
    Since P64 <= (1+gamma)Pin, the triangle inequality gives the final bound
    [gamma+(2u+u^2)(1+gamma)]Pin plus the subnormal terms. No observed error
    or experiment outcome is used to set this bound.
    """
    assert base.dtype in (torch.float32, torch.float64), "UNQUALIFIED_PHASE_DTYPE"
    assert output.dtype == base.dtype and rotated64.dtype == torch.float64
    assert bool(torch.isfinite(base).all() and torch.isfinite(rotated64).all()
                and torch.isfinite(output).all()), "NONFINITE_PHASE_FIELD"
    x = base.double()
    sine, cosine = x.chunk(2, dim=-1)
    rsine, rcosine = rotated64.chunk(2, dim=-1)
    osine, ocosine = output.double().chunk(2, dim=-1)
    power = sine.square() + cosine.square()
    rotated_power = rsine.square() + rcosine.square()
    output_power = osine.square() + ocosine.square()
    gamma = FP64_POWER_RELATIVE_ALLOWANCE
    fp64_error = (rotated_power - power).abs()
    assert bool((fp64_error <= gamma * power).all()), "FP64_ROTATION_POWER_DRIFT"
    finfo = torch.finfo(output.dtype)
    unit_roundoff = finfo.eps / 2
    smallest_subnormal = finfo.tiny * finfo.eps
    cast_factor = 2 * unit_roundoff + unit_roundoff ** 2
    bound = (gamma + cast_factor * (1 + gamma)) * power
    bound = bound + 2 * smallest_subnormal * (1 + unit_roundoff) * (rsine.abs() + rcosine.abs())
    bound = bound + 2 * smallest_subnormal ** 2
    error = (output_power - power).abs()
    assert bool((error <= bound).all()), "PHASE_POWER_EXCEEDS_DTYPE_ROUNDING_BOUND"
    norm = x.norm(dim=-1, keepdim=True)
    assert float(norm.min()) > 0
    relative = (output.double().norm(dim=-1, keepdim=True) / norm - 1).abs().max()
    assert float(relative) < 2e-6, "PHASE_NORM_RELATIVE_DRIFT"
    ratio = torch.where(bound > 0, error / bound, torch.zeros_like(error))
    fp64_ratio = torch.where(power > 0, fp64_error / power, torch.zeros_like(error))
    return dict(
        kind="phase", norm_relative_error=float(relative),
        spectral_power_max_abs_error=float(error.max()),
        perturbation_rms=float((output.double() - x).square().mean().sqrt()),
        spectral_power_check="per_pair_dtype_rounding_bound_v2",
        spectral_power_max_error_over_bound=float(ratio.max()),
        spectral_power_max_abs_bound=float(bound.max()),
        input_pair_power_max=float(power.max()),
        fp64_rotation_power_max_relative_error=float(fp64_ratio.max()),
        fp64_rotation_power_relative_allowance=gamma,
        output_unit_roundoff=unit_roundoff,
        norm_relative_tolerance_unchanged=2e-6,
    )


def _joint_phase(base, omega, scale, phase_arm, side):
    """Keep the original expression and evaluation order, including final cast."""
    assert phase_arm in PHASE_ARMS and base.shape[-1] == 2 * len(omega)
    x = base.double()
    norm = x.norm(dim=-1, keepdim=True)
    assert float(norm.min()) > 0
    scope, axis, sign = phase_arm.split("_")
    axis = {"X": 0, "Y": 1}[axis]
    direction = 1. if sign == "PLUS" else -1.
    shift = torch.ones(base.shape[:-1] + (1,), dtype=torch.float64, device=base.device) * DELTA * direction
    if scope == "LOCAL":
        n = norm.numel()
        signs = torch.ones(n, dtype=torch.float64)
        signs[original.permutation(n, side)[:n // 2]] = -1.
        shift = shift * signs.reshape_as(shift).to(base.device)
    angle = shift * omega[:, axis].double().to(base.device) * float(scale)
    sin, cos = x.chunk(2, dim=-1)
    rotated64 = torch.cat((sin * angle.cos() + cos * angle.sin(),
                           cos * angle.cos() - sin * angle.sin()), dim=-1)
    output = rotated64.to(base.dtype)
    return output, _phase_invariants(base, rotated64, output)


def transform(p, omega, scale, arm, radius, side):
    if arm not in ARMS:
        raise ValueError(f"unknown property-restore arm: {arm}")
    if "__" not in arm:
        return original.transform(p, omega, scale, arm, radius, side)
    amplitude_arm, phase_arm = arm.split("__", 1)
    assert amplitude_arm == "AMP_PERMUTE" and phase_arm in PHASE_ARMS
    base, amplitude_diagnostics = original.transform(p, omega, scale, amplitude_arm, radius, side)
    output, phase_diagnostics = _joint_phase(base, omega, scale, phase_arm, side)
    return output, dict(
        kind="amplitude_then_phase", order=[amplitude_arm, phase_arm],
        amplitude=amplitude_diagnostics, phase=phase_diagnostics,
        phase_preservation_reference="AMP_PERMUTE_output",
        total_perturbation_rms_vs_native=float((output.double() - p.double()).square().mean().sqrt()),
        conditional_restoration_is_not_true_geometry_recovery=True,
    )


def self_test():
    """Retain the v1 algebra audit and exercise the replacement on scaled input."""
    import numpy as np

    legacy = previous.self_test()
    generator = torch.Generator().manual_seed(SEED + 392)
    field = torch.randn(1, 1, 13, 32, generator=generator, dtype=torch.float64) * 32
    omega64 = torch.randn(16, 2, generator=generator, dtype=torch.float64)
    rows = []
    for dtype in (torch.float32, torch.float64):
        p, omega = field.to(dtype), omega64.to(dtype)
        for side in (0, 1):
            base, _ = original.transform(p, omega, 1.7, "AMP_PERMUTE", 1., side)
            for phase in PHASE_ARMS:
                value, diagnostics = transform(p, omega, 1.7, "AMP_PERMUTE__" + phase, 1., side)
                # NumPy starts from the already-cast amplitude input, matching
                # the frozen joint order rather than fusing its two casts.
                expected = previous._numpy_expected(base.numpy(), omega.numpy(), 1.7, phase, side)
                error = float(np.abs(expected - value.double().numpy()).max())
                limit = (torch.finfo(dtype).eps * float(np.abs(expected).max())
                         + 128 * torch.finfo(torch.float64).eps * float(base.abs().max()))
                assert error <= limit, "INDEPENDENT_NUMPY_ROTATION_DISAGREEMENT"
                rows.append(dict(dtype=str(dtype), side=side, arm=phase,
                                 independent_numpy_max_abs_error=error, **diagnostics["phase"]))
    # A substantive perturbation must still fail the power gate.
    bad = field.float().clone()
    bad[..., 0] *= 1.001
    try:
        _phase_invariants(field.float(), field.float().double(), bad)
    except AssertionError as error:
        assert str(error) == "PHASE_POWER_EXCEEDS_DTYPE_ROUNDING_BOUND"
    else:
        raise AssertionError("CORRUPTION_NOT_REJECTED")
    return dict(status="PROPERTY_RESTORE_FACTORIAL_ALGEBRA_V2_PASS",
                arms=list(ARMS), phase_arms=list(PHASE_ARMS), panel=PANEL,
                seed=SEED, delta=DELTA, legacy_algebra=legacy, scaled_checks=rows,
                corruption_rejected=True, joint_order="AMP_PERMUTE_then_phase",
                scientific_transform_changed=False,
                phase_reference="amplitude-conditioned field")


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
