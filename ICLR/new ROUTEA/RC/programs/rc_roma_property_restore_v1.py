"""Norm-assignment x phase interventions using the frozen position-factor rules.

The joint arm first permutes patch norms and then applies the original phase
intervention. Restoring a factor means restoring its computational value; it
does not establish recovery of true image geometry or a target segmentation.
In particular, phase norm/power preservation is relative to that arm's
amplitude-conditioned input, not necessarily the original NATIVE field.
"""
from __future__ import annotations

import torch

import rc_roma_position_factor_v1 as original

SEED = original.SEED
DELTA = original.DELTA
PANEL = [0, 1, 2, 4, 5, 6, 7, 8]
PHASE_ARMS = tuple(arm for arm in original.ARMS if arm.startswith(("GLOBAL_", "LOCAL_")))
ARMS = ("NATIVE", "AMP_PERMUTE") + PHASE_ARMS + tuple("AMP_PERMUTE__" + arm for arm in PHASE_ARMS)


def transform(p, omega, scale, arm, radius, side):
    """Return transformed field and diagnostics without changing old semantics."""
    if arm not in ARMS:
        raise ValueError(f"unknown property-restore arm: {arm}")
    if "__" not in arm:
        return original.transform(p, omega, scale, arm, radius, side)
    amplitude_arm, phase_arm = arm.split("__", 1)
    assert amplitude_arm == "AMP_PERMUTE" and phase_arm in PHASE_ARMS
    base, amplitude_diagnostics = original.transform(p, omega, scale, amplitude_arm, radius, side)
    output, phase_diagnostics = original.transform(base, omega, scale, phase_arm, radius, side)
    return output, {
        "kind": "amplitude_then_phase", "order": [amplitude_arm, phase_arm],
        "amplitude": amplitude_diagnostics, "phase": phase_diagnostics,
        "phase_preservation_reference": "AMP_PERMUTE_output",
        "total_perturbation_rms_vs_native": float((output.double() - p.double()).square().mean().sqrt()),
        "conditional_restoration_is_not_true_geometry_recovery": True,
    }


def _numpy_expected(p, omega, scale, arm, side):
    """Independent double-precision equation, not the torch transform kernel.

    The frozen permutation indices are shared as intervention inputs. NumPy
    independently computes normalization, rotation and all joint outputs.
    """
    import numpy as np

    value = np.asarray(p, dtype=np.float64).copy()
    omega = np.asarray(omega, dtype=np.float64)
    norms = np.sqrt(np.sum(value * value, axis=-1, keepdims=True))
    assert float(norms.min()) > 0
    indices = original.permutation(norms.size, side).numpy()
    if arm.startswith("AMP_PERMUTE"):
        desired = norms.reshape(-1)[indices].reshape(norms.shape)
        value = value * desired / norms
    phase_arm = arm.split("__", 1)[-1]
    if phase_arm in PHASE_ARMS:
        scope, axis, sign = phase_arm.split("_")
        delta = DELTA * (1.0 if sign == "PLUS" else -1.0)
        shifts = np.full(value.shape[:-1] + (1,), delta, dtype=np.float64)
        if scope == "LOCAL":
            local_signs = np.ones(norms.size, dtype=np.float64)
            local_signs[indices[:norms.size // 2]] = -1.0
            shifts = shifts * local_signs.reshape(shifts.shape)
        angle = shifts * omega[:, {"X": 0, "Y": 1}[axis]] * float(scale)
        sine, cosine = np.split(value, 2, axis=-1)
        value = np.concatenate((sine * np.cos(angle) + cosine * np.sin(angle),
                                cosine * np.cos(angle) - sine * np.sin(angle)), axis=-1)
    return value


def _power(value):
    sine, cosine = value.double().chunk(2, dim=-1)
    return sine.square() + cosine.square()


def self_test():
    """Check the factorial controls, including odd patch counts and both sides.

    Tests use nonconstant norms and soft distributions rather than unit-norm
    Fourier points, so norm reassignment cannot accidentally be an identity.
    """
    import numpy as np

    assert len(ARMS) == len(set(ARMS)) == 18
    assert len(PHASE_ARMS) == 8 and SEED == 20260924 and DELTA == 0.125
    generator = torch.Generator().manual_seed(SEED + 391)
    records = []
    for patch_count in (7, 12):
        omega64 = torch.randn(16, 2, generator=generator, dtype=torch.float64)
        coords = torch.randn(13, 2, generator=generator, dtype=torch.float64)
        attention = torch.softmax(torch.randn(patch_count, 13, generator=generator, dtype=torch.float64), -1)
        scale = 1.7
        phase = (coords @ omega64.T) * scale
        field64 = (attention @ torch.cat((phase.sin(), phase.cos()), -1)).reshape(1, 1, patch_count, 32)
        assert float(field64.norm(dim=-1).std()) > 0.01
        for dtype in (torch.float64, torch.float32):
            p = field64.to(dtype=dtype)
            omega = omega64.to(dtype=dtype)
            limit = 1e-12 if dtype == torch.float64 else 1e-6
            radius = float(p.double().norm(dim=-1).mean())
            for side in (0, 1):
                outputs = {arm: transform(p, omega, scale, arm, radius, side)[0] for arm in ARMS}
                norms = p.double().norm(dim=-1, keepdim=True)
                indices = original.permutation(norms.numel(), side).to(p.device)
                reordered = norms.reshape(-1)[indices].reshape_as(norms)
                reassigned = outputs["AMP_PERMUTE"]
                assert not torch.equal(reassigned, p), "fixture must exercise norm reassignment"
                histogram_error = float((reassigned.double().norm(dim=-1).flatten().sort().values
                                         - norms.flatten().sort().values).abs().max())
                amplitude_power_error = float((_power(reassigned) - _power(p) * (reordered / norms).square()).abs().max())
                assert histogram_error < limit and amplitude_power_error < limit
                phase_norm_error = phase_power_error = dose_error = commute_error = 0.0
                numpy_error = legacy_error = 0.0
                for amplitude in ("NATIVE", "AMP_PERMUTE"):
                    base = outputs[amplitude]
                    base_norm = base.double().norm(dim=-1)
                    base_power = _power(base)
                    prefix = "" if amplitude == "NATIVE" else "AMP_PERMUTE__"
                    for phase_arm in PHASE_ARMS:
                        value = outputs[prefix + phase_arm]
                        phase_norm_error = max(phase_norm_error, float((value.double().norm(dim=-1) - base_norm).abs().max()))
                        phase_power_error = max(phase_power_error, float((_power(value) - base_power).abs().max()))
                    for axis in ("X", "Y"):
                        global_dose = sum((outputs[prefix + f"GLOBAL_{axis}_{sign}"].double() - base.double()).square()
                                          for sign in ("PLUS", "MINUS"))
                        local_dose = sum((outputs[prefix + f"LOCAL_{axis}_{sign}"].double() - base.double()).square()
                                         for sign in ("PLUS", "MINUS"))
                        dose_error = max(dose_error, float((global_dose - local_dose).abs().max()))
                for phase_arm in PHASE_ARMS:
                    reversed_order = original.transform(outputs[phase_arm], omega, scale, "AMP_PERMUTE", radius, side)[0]
                    commute_error = max(commute_error, float((reversed_order.double() - outputs["AMP_PERMUTE__" + phase_arm].double()).abs().max()))
                for arm, value in outputs.items():
                    expected = _numpy_expected(p.double().numpy(), omega.double().numpy(), scale, arm, side)
                    numpy_error = max(numpy_error, float(np.abs(expected - value.double().numpy()).max()))
                    if "__" not in arm:
                        old_value = original.transform(p, omega, scale, arm, radius, side)[0]
                        legacy_error = max(legacy_error, float((value.double() - old_value.double()).abs().max()))
                assert phase_norm_error < limit and phase_power_error < limit
                assert dose_error < limit and commute_error < limit and numpy_error < limit
                assert legacy_error == 0.0
                records.append({
                    "patches": patch_count, "side": side, "dtype": str(dtype), "absolute_tolerance": limit,
                    "norm_histogram_max_error": histogram_error,
                    "amplitude_power_scaling_max_error": amplitude_power_error,
                    "phase_norm_error_vs_amplitude_base": phase_norm_error,
                    "phase_power_error_vs_amplitude_base": phase_power_error,
                    "antithetic_global_local_elementwise_dose_max_error": dose_error,
                    "amplitude_phase_commutation_max_error": commute_error,
                    "independent_numpy_max_error": numpy_error,
                    "legacy_arm_max_error": legacy_error,
                })
    return {
        "status": "PROPERTY_RESTORE_FACTORIAL_ALGEBRA_PASS", "arms": list(ARMS),
        "phase_arms": list(PHASE_ARMS), "panel": PANEL, "seed": SEED, "delta": DELTA,
        "checks": records, "joint_order": "AMP_PERMUTE_then_phase",
        "phase_reference": "amplitude-conditioned field",
        "inference_boundary": "Conditional factor restoration is not true geometry recovery.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=2))
