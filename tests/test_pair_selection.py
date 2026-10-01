from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs.io.lammps import read_lammps_frame
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_all_prereactive_pairs
from MartiniFuncs.chemistry.cpd_n4t.selection import (PairCriteria, assess_prereactive_pairs,
                                                      measure_pair_geometry)


class PairSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atoms = read_lammps_frame(
            Path(__file__).parent / "lammpsio_tests/frame_0.lammpstrj", units="metal"
        ).atoms
        cls.pairs, cls.shifts = get_all_prereactive_pairs(cls.atoms, scale=1.3)

    def test_no_default_orientation_rejection_and_opt_in_reasons(self):
        reports = assess_prereactive_pairs(self.atoms, self.pairs, self.shifts)
        self.assertTrue(all(r.accepted for r in reports))
        criteria = PairCriteria(max_axis_angle=45)
        filtered = assess_prereactive_pairs(self.atoms, self.pairs, self.shifts, criteria=criteria)
        self.assertEqual([r.accepted for r in filtered], [False, True])
        self.assertIn("max_axis_angle", filtered[0].reasons[0])
        pairs, shifts = get_all_prereactive_pairs(self.atoms, criteria=criteria)
        self.assertEqual(pairs, [self.pairs[1]])
        self.assertEqual(shifts, [self.shifts[1]])

    def test_distances_match_jointly_placed_cartesian_coordinates(self):
        from MartiniFuncs.structure import graph
        bonded = graph.build_bond_graph(self.atoms, scale=1.3)
        for pair, shift in zip(self.pairs, self.shifts):
            geometry = measure_pair_geometry(self.atoms, pair, shift, graph=bonded)
            mapped_ids = tuple(tuple(int(self.atoms.arrays["global_index"][i]) for i in bond)
                               for bond in geometry.forming_bonds_frame_local)
            self.assertEqual(mapped_ids, geometry.forming_bonds)
            ci, cx, _ = graph.place_component(self.atoms, bonded, pair[0], image_shift=shift)
            mi, mx, _ = graph.place_component(self.atoms, bonded, pair[1])
            pair_ids = np.concatenate((ci, mi))
            self.assertEqual(tuple(tuple(int(pair_ids[i]) for i in bond)
                                   for bond in geometry.forming_bonds_local), geometry.forming_bonds)
            self.assertTrue(all(0 <= i < 11 and 11 <= j < 34
                                for i, j in geometry.forming_bonds_local))
            positions = dict(zip(np.concatenate((ci, mi)), np.vstack((cx, mx))))
            expected = [np.linalg.norm(positions[i] - positions[j]) for i, j in geometry.forming_bonds]
            assert_allclose(geometry.forming_distances, expected)
        # Tight distance screening rejects both despite passing the COM cutoff.
        reports = assess_prereactive_pairs(self.atoms, self.pairs, self.shifts,
                                           criteria=PairCriteria(max_forming_distance=3.25))
        self.assertTrue(all(not r.accepted for r in reports))

    def test_reordering_rotation_and_lattice_translation_invariance(self):
        reference = assess_prereactive_pairs(self.atoms, self.pairs, self.shifts)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transformed = self.atoms[::-1]
        transformed.positions = (transformed.positions + self.atoms.cell[0]) @ rotation
        transformed.set_cell(self.atoms.cell.array @ rotation)
        reports = assess_prereactive_pairs(transformed, self.pairs, self.shifts)
        for original, changed in zip(reference, reports):
            self.assertEqual(original.geometry.forming_bonds, changed.geometry.forming_bonds)
            expected_local = tuple(tuple(len(self.atoms)-1-i for i in bond)
                                   for bond in original.geometry.forming_bonds_frame_local)
            self.assertEqual(changed.geometry.forming_bonds_frame_local, expected_local)
            self.assertEqual(changed.geometry.forming_bonds_local, original.geometry.forming_bonds_local)
            assert_allclose(original.geometry.forming_distances, changed.geometry.forming_distances)
            assert_allclose([original.geometry.axis_angle, original.geometry.plane_angle,
                             original.geometry.approach_angle],
                            [changed.geometry.axis_angle, changed.geometry.plane_angle,
                             changed.geometry.approach_angle], atol=1e-10)

    def test_ambiguous_connectivity_is_reported(self):
        reports = assess_prereactive_pairs(self.atoms, self.pairs, self.shifts, scale=1.5)
        self.assertIsNone(reports[1].geometry)
        self.assertIn("bridge", reports[1].reasons[0])

    def test_invalid_thresholds_and_lengths(self):
        for kwargs in ({"max_axis_angle": 91}, {"max_forming_distance": -1},
                       {"max_plane_angle": float("nan")}):
            with self.assertRaises(ValueError):
                PairCriteria(**kwargs)
        with self.assertRaises(ValueError):
            assess_prereactive_pairs(self.atoms, self.pairs, [])


if __name__ == "__main__":
    unittest.main()
