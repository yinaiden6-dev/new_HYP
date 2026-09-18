"""Synthetic engineering checks only; runnable without pytest or torch."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


CORE_PATH = Path(__file__).resolve().parents[1] / "src/rc_aslo_xf/rc_cycle_closed_region_generation_v1.py"
SPEC = importlib.util.spec_from_file_location("rc_cycle_closed_region_generation_v1_e0", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def bank(shape=(4, 4), reference_shape=None):
    reference_shape = reference_shape or shape
    h, w = shape
    rh, rw = reference_shape
    count = h * w
    refs = [index % (rh * rw) for index in range(count)]
    return SimpleNamespace(
        query_grid_shape=shape, reference_grid_shape=reference_shape,
        valid_mask=[True] * count,
        source_query_indices=list(range(count)), source_reference_indices=refs,
        query_indices=list(range(count)), reference_indices=refs.copy(),
        query_coordinate_permutation=list(range(count)),
        reference_coordinate_permutation=list(range(rh * rw)),
        query_rc=[[index // w, index % w] for index in range(count)],
        reference_rc=[[index // rw, index % rw] for index in refs],
        source_reverse_query_xy=[[(index % w + 0.5) * 2 / w - 1,
                                  (index // w + 0.5) * 2 / h - 1] for index in range(count)],
        query_resource_key="opaque-query", candidate_resource_key="opaque-candidate",
        reference_resource_key="opaque-reference",
    )


def reference_assignment(value, indices):
    width = value.reference_grid_shape[1]
    value.source_reference_indices = list(indices)
    value.reference_indices = list(indices)
    value.reference_rc = [[index // width, index % width] for index in indices]


class CycleClosedRegionE0(unittest.TestCase):
    def test_identity_warp_returns_one_maximal_sixteen_atom_region(self):
        value = bank()
        before = copy.deepcopy(vars(value))
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertFalse(out["structural_h0"])
        self.assertIsNone(out["h0_reason"])
        self.assertEqual(out["counts"]["cycle_closed_atoms"], 16)
        self.assertEqual([item["atom_indices"] for item in out["components"]], [list(range(16))])
        self.assertEqual(vars(value), before)

    def test_constant_return_false_region_is_structural_h0(self):
        value = bank()
        value.source_reverse_query_xy = [value.source_reverse_query_xy[0].copy() for _ in range(16)]
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertEqual(out["cycle_closed_atom_indices"], [0])
        self.assertTrue(out["structural_h0"])
        self.assertEqual(out["h0_reason"], "CYCLE_CLOSED_REGION_FAMILY_EMPTY")
        self.assertEqual(out["h0_detail"], "NO_QUALIFYING_CONNECTED_COMPONENTS")

    def test_half_open_boundaries_floor_xy_without_clamping(self):
        value = bank((2, 2))
        value.source_reverse_query_xy = [[-1.0, -1.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 0.0]]
        self.assertEqual(CORE.enumerate_cycle_closed_regions(value)["cycle_closed_atom_indices"], [0, 1, 2, 3])
        value.source_reverse_query_xy = [[1.0, -0.5], [-1.01, -0.5], [-0.5, 1.0], [0.0, 0.0]]
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertEqual(out["source_returned_query_indices"], [None, None, None, 3])
        self.assertEqual(out["cycle_closed_atom_indices"], [3])
        value.source_reverse_query_xy[3] = [math.nextafter(1.0, 0.0)] * 2
        self.assertEqual(CORE.enumerate_cycle_closed_regions(value)["source_returned_query_indices"][3], 3)

    def test_three_atoms_cannot_form_a_hypothesis(self):
        value = bank((2, 2))
        value.valid_mask[-1] = False
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertTrue(out["structural_h0"])
        self.assertEqual(out["counts"]["cycle_closed_atoms"], 3)

    def test_four_atoms_require_two_distinct_assigned_reference_atoms(self):
        value = bank((2, 2))
        reference_assignment(value, [0, 0, 0, 0])
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertTrue(out["structural_h0"])
        self.assertEqual(out["counts"]["rejected_fewer_than_two_reference_atoms"], 1)
        reference_assignment(value, [0, 0, 0, 1])
        self.assertFalse(CORE.enumerate_cycle_closed_regions(value)["structural_h0"])

    def test_all_disconnected_qualifying_components_are_returned(self):
        value = bank((2, 5))
        value.valid_mask[2] = value.valid_mask[7] = False
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertEqual([item["query_indices"] for item in out["components"]], [[0, 1, 5, 6], [3, 4, 8, 9]])
        self.assertEqual(out["counts"]["qualifying_components"], 2)

    def test_reference_distance_splits_query_connected_support(self):
        value = bank((2, 4), (2, 10))
        reference_assignment(value, [0, 1, 8, 9, 10, 11, 18, 19])
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertEqual([item["query_indices"] for item in out["components"]], [[0, 1, 4, 5], [2, 3, 6, 7]])

    def test_features_sign_labels_and_identity_names_do_not_drive_geometry(self):
        value = bank()
        baseline = CORE.enumerate_cycle_closed_regions(value)
        value.features = [[float("nan"), -1e12]] * 16
        value.signed_evidence = [-1e12] * 16
        value.rank = value.slot = value.target = "FORBIDDEN_NOT_READ"
        self.assertEqual(CORE.enumerate_cycle_closed_regions(value), baseline)
        value.query_resource_key = "renamed-query"
        value.candidate_resource_key = "renamed-candidate"
        value.reference_resource_key = "renamed-reference"
        renamed = CORE.enumerate_cycle_closed_regions(value)
        for key in ("query_resource_key", "candidate_resource_key", "reference_resource_key"):
            baseline.pop(key)
            renamed.pop(key)
        self.assertEqual(renamed, baseline)

    def test_no_hidden_feature_or_score_attribute_access(self):
        class GeometryOnly(SimpleNamespace):
            def __getattribute__(self, name):
                if name in {"features", "signed_evidence", "rank", "slot", "target", "cycle_error", "precision_quality"}:
                    raise AssertionError(f"forbidden read: {name}")
                return super().__getattribute__(name)
        value = GeometryOnly(**vars(bank()))
        self.assertEqual(CORE.enumerate_cycle_closed_regions(value)["counts"]["qualifying_components"], 1)

    def test_deterministic_complete_serialization_replay(self):
        value = bank()
        out = CORE.enumerate_cycle_closed_regions(value)
        serialized = CORE.serialize_cycle_closed_regions(out)
        self.assertEqual(CORE.replay_cycle_closed_regions(value, serialized), out)
        self.assertEqual(CORE.serialize_cycle_closed_regions(CORE.enumerate_cycle_closed_regions(value)), serialized)
        broken = json.loads(serialized)
        broken["components"][0]["reference_indices"][0] = 15
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            CORE.replay_cycle_closed_regions(value, CORE.serialize_cycle_closed_regions(broken))

    def test_coordinate_control_changes_connectivity_but_preserves_cycle_axis(self):
        value = bank()
        value.valid_mask = [index in {0, 1, 4, 5} for index in range(16)]
        real = CORE.enumerate_cycle_closed_regions(value)
        self.assertEqual(real["components"][0]["query_indices"], [0, 1, 4, 5])
        controlled = copy.deepcopy(value)
        permutation = list(range(16))
        for first, second in ((1, 3), (4, 12), (5, 15)):
            permutation[first], permutation[second] = permutation[second], permutation[first]
        controlled.query_coordinate_permutation = permutation
        controlled.query_indices = permutation.copy()
        controlled.query_rc = [[index // 4, index % 4] for index in permutation]
        changed = CORE.enumerate_cycle_closed_regions(controlled)
        self.assertEqual(changed["cycle_closed_atom_indices"], real["cycle_closed_atom_indices"])
        self.assertEqual(changed["source_returned_query_indices"], real["source_returned_query_indices"])
        self.assertTrue(changed["structural_h0"])
        self.assertEqual(changed["counts"]["maximal_components"], 4)

    def test_component_provenance_is_canonical_in_current_not_source_coordinates(self):
        value = bank()
        permutation = list(reversed(range(16)))
        value.query_coordinate_permutation = permutation.copy()
        value.reference_coordinate_permutation = permutation.copy()
        value.query_indices = permutation.copy()
        value.reference_indices = permutation.copy()
        value.query_rc = [[index // 4, index % 4] for index in permutation]
        value.reference_rc = copy.deepcopy(value.query_rc)
        out = CORE.enumerate_cycle_closed_regions(value)
        self.assertEqual(out["cycle_closed_atom_indices"], list(range(16)))
        self.assertEqual(out["components"][0]["query_indices"], list(range(16)))
        self.assertEqual(out["components"][0]["atom_indices"], permutation)
        self.assertEqual(out["components"][0]["source_returned_query_indices"], permutation)

    def test_malformed_geometry_aborts_instead_of_scientific_h0(self):
        for field, invalid in (
            ("query_coordinate_permutation", [0] * 16),
            ("query_rc", [[0, 0]] * 16),
            ("source_query_indices", list(reversed(range(16)))),
            ("source_reference_indices", [16] * 16),
            ("source_reverse_query_xy", [[float("nan"), 0.0]] * 16),
            ("valid_mask", [1] * 16),
        ):
            with self.subTest(field=field):
                value = bank()
                setattr(value, field, invalid)
                with self.assertRaises(ValueError):
                    CORE.enumerate_cycle_closed_regions(value)

    def test_empty_valid_axis_and_empty_cycle_have_explicit_distinct_h0_reasons(self):
        value = bank()
        value.valid_mask = [False] * 16
        self.assertEqual(CORE.enumerate_cycle_closed_regions(value)["h0_detail"], "NO_VALID_ATOMS")
        value.valid_mask = [True] * 16
        value.source_reverse_query_xy = [[1.0, 1.0]] * 16
        self.assertEqual(CORE.enumerate_cycle_closed_regions(value)["h0_detail"], "NO_TOKEN_CYCLE_CLOSED_ATOMS")


def run_synthetic_e0():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CycleClosedRegionE0)
    names = [test._testMethodName for test in suite]
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {"status": "SYNTHETIC_ENGINEERING_E0_PASS" if result.wasSuccessful() else "ABORT_SYNTHETIC_E0_FAILURE",
            "scientific_go": False, "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors),
            "checks": names, "runner_output": stream.getvalue()}


if __name__ == "__main__":
    receipt = run_synthetic_e0()
    print(json.dumps(receipt, sort_keys=True, indent=2))
    raise SystemExit(0 if receipt["status"] == "SYNTHETIC_ENGINEERING_E0_PASS" else 1)
