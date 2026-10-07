"""CPD/N4T ring dihedral: CH2 carbon - CPD termini midpoint - C=C midpoint - imide N."""
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCollection, CpdN4tCriteria
from MartiniFuncs.chemistry.cpd_n4t.selection import _dihedral
from MartiniFuncs.io.lammps import read_lammps_frame
from MartiniFuncs.pairs.assessment import screen_frame
from MartiniFuncs.pairs.candidates import candidate_mask, concatenate, frame_candidates
from MartiniFuncs.pairs.model import get_pair_model
from MartiniFuncs.pairs.structures import get_pair_positions_and_forces


def projected_dihedral(p1, p2, p3, p4):
    """Independent formula: angle from proj(p1-p2) to proj(p4-p3) about the p2->p3 axis."""
    axis = (p3 - p2) / np.linalg.norm(p3 - p2)
    v = (p1 - p2) - np.dot(p1 - p2, axis) * axis
    w = (p4 - p3) - np.dot(p4 - p3, axis) * axis
    return np.degrees(np.arctan2(np.dot(axis, np.cross(v, w)), np.dot(v, w)))


def neighbors(xyz, symbols, row, element, cutoff):
    d = np.linalg.norm(xyz - xyz[row], axis=1)
    return [k for k in np.flatnonzero((d < cutoff) & (d > 0)) if symbols[k] == element]


class DihedralFunctionTests(unittest.TestCase):
    def test_known_values_and_sign(self):
        p2, p3 = np.zeros(3), np.array([0., 0, 1])
        p1 = np.array([1., 0, 0])
        self.assertAlmostEqual(_dihedral(p1, p2, p3, [1., 0, 1]), 0.)          # cis: exo-like
        self.assertAlmostEqual(abs(_dihedral(p1, p2, p3, [-1., 0, 1])), 180.)  # trans: endo-like
        # IUPAC: viewed from p2 towards p3, p2-p1 turns clockwise onto p3-p4 -> positive.
        self.assertAlmostEqual(_dihedral(p1, p2, p3, [0., 1, 1]), 90.)
        self.assertAlmostEqual(_dihedral(p1, p2, p3, [0., -1, 1]), -90.)       # mirror image
        with self.assertRaisesRegex(ValueError, "collinear"):
            _dihedral([0., 0, -1], p2, p3, [0., 1, 1])

    def test_matches_projection_formula_for_random_points(self):
        rng = np.random.default_rng(7)
        for points in rng.normal(size=(200, 4, 3)):
            assert_allclose(_dihedral(*points), projected_dihedral(*points), atol=1e-9)


class RingDihedralGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = read_lammps_frame(Path(__file__).parent / "lammpsio_tests/multiple_frames.lammpstrj", units="metal")
        cls.frame = frame
        cls.pairs, cls.shifts, cls.candidates = screen_frame(frame.atoms, "cpd_n4t", collection=CpdN4tCollection(reactive_radius=12.0),
                                                             return_candidates=True)

    def test_dihedral_from_independently_identified_atoms(self):
        """CH2, termini, C=C and N found by distances alone in the jointly placed pair."""
        selected = list(range(0, len(self.pairs), max(1, len(self.pairs) // 25)))
        records = get_pair_positions_and_forces(self.frame.atoms, self.frame, [self.pairs[k] for k in selected],
                                                [self.shifts[k] for k in selected])
        checked = 0
        for record, k in zip(records.values(), selected):
            geometry = self.candidates[k][3]
            if geometry is None:
                continue
            xyz, symbols = record["positions"], list(record["symbols"])
            cpd, mal = record["first_local_indices"], record["second_local_indices"]
            ch2 = [r for r in cpd if symbols[r] == "C" and len(neighbors(xyz, symbols, r, "H", 1.25)) == 2]
            self.assertEqual(len(ch2), 1)
            termini = neighbors(xyz, symbols, ch2[0], "C", 1.7)
            carbonyl = [r for r in mal if symbols[r] == "C" and neighbors(xyz, symbols, r, "O", 1.4)]
            vinyl = [r for r in mal if symbols[r] == "C" and neighbors(xyz, symbols, r, "H", 1.25)
                     and set(neighbors(xyz, symbols, r, "C", 1.7)) & set(carbonyl)]
            nitrogen = [r for r in mal if symbols[r] == "N"]
            self.assertEqual((len(termini), len(vinyl), len(nitrogen)), (2, 2, 1))
            expected = projected_dihedral(xyz[ch2[0]], xyz[termini].mean(0), xyz[vinyl].mean(0), xyz[nitrogen[0]])
            assert_allclose(geometry.ring_dihedral, expected, atol=1e-9)
            magnitude = abs(expected)
            assert_allclose(geometry.ring_dihedral_deviation, min(magnitude, 180 - magnitude), atol=1e-9)
            self.assertEqual(geometry.approach_type, "exo" if magnitude < 90 else "endo")
            checked += 1
        self.assertGreater(checked, 10)

    def test_criterion_accepts_near_zero_or_180_only(self):
        with self.assertRaisesRegex(ValueError, "at most 90"):
            CpdN4tCriteria(max_ring_dihedral_deviation=91)
        criteria = CpdN4tCriteria(max_ring_dihedral_deviation=30)
        geometries = [c[3] for c in self.candidates if c[3] is not None]
        model = get_pair_model("cpd_n4t")
        for geometry in geometries:
            phi = abs(geometry.ring_dihedral)
            accepted = not criteria.rejection_reasons(model.metrics(geometry))
            self.assertEqual(accepted, phi <= 30 or phi >= 150)
        self.assertTrue(any(abs(g.ring_dihedral) <= 30 or abs(g.ring_dihedral) >= 150 for g in geometries))
        self.assertTrue(any(30 < abs(g.ring_dihedral) < 150 for g in geometries))

    def test_cached_candidates_select_by_ring_dihedral(self):
        model = get_pair_model("cpd_n4t")
        rows = [(0, 0, shift, model.metric_row(distance, geometry))
                for _, shift, distance, geometry in self.candidates]
        candidates = concatenate([frame_candidates(rows, model)], 12.0, model)
        self.assertIn("ring_dihedral", candidates["metric_names"])
        mask = candidate_mask(candidates, criteria=CpdN4tCriteria(max_ring_dihedral_deviation=30))
        expected = [g is not None and g.ring_dihedral_deviation <= 30 for *_, g in self.candidates]
        np.testing.assert_array_equal(mask, expected)


if __name__ == "__main__":
    unittest.main()
