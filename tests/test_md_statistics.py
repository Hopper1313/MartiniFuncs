import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms, units

from MartiniFuncs import md_statistics as md


class MDStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms("HHe", positions=[[0, 0, 0], [2, 0, 0]],
                           masses=[1, 3], cell=[10, 10, 10])
        self.atoms.new_array("global_index", np.array([90, 7]))
        self.atoms.set_velocities([[1, 0, 0], [0, 2, 0]])
        self.atoms.new_array("lammps_forces", np.array([[1., 0, 0], [-1., 3, 0]]))

    def test_molecule_selection_and_ase_velocity_units(self):
        assert_allclose(md.get_molecule_center_of_mass(self.atoms[::-1], [90, 7]), [1.5, 0, 0])
        assert_allclose(md.get_molecule_com_velocity(self.atoms, [90, 7]),
                        np.array([.25, 1.5, 0]) * 1000 * units.fs)
        assert_allclose(md.get_molecule_net_force(self.atoms, [7, 90]), [0, 3, 0])
        self.atoms.pbc = True
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            md.get_molecule_center_of_mass(self.atoms, [90, 7])
        with self.assertRaisesRegex(ValueError, "Missing atom"):
            md.get_molecule_com_velocity(self.atoms, [999])

    def test_frame_thermodynamics(self):
        self.assertAlmostEqual(md.get_frame_kinetic_energy(self.atoms), 6.5)
        self.assertAlmostEqual(md.get_frame_temperature(self.atoms, degrees_of_freedom=6),
                               13 / (6 * units.kB))
        self.assertAlmostEqual(md.get_frame_density(self.atoms), 4 * 1.66053906660 / 1000, places=8)
        stats = md.get_frame_force_statistics(self.atoms)
        assert_allclose(stats["net_force"], [0, 3, 0])
        self.atoms.set_velocities([[1, 2, 3], [1, 2, 3]])
        self.assertAlmostEqual(md.get_frame_temperature(self.atoms, degrees_of_freedom=3, remove_com=True), 0)
        self.atoms.arrays.pop("momenta")
        with self.assertRaisesRegex(ValueError, "required"):
            md.get_frame_temperature(self.atoms, degrees_of_freedom=6)

    def test_rmsd_identity_and_displacement(self):
        moved = self.atoms[::-1]
        moved.positions += [3, 0, 0]
        self.assertAlmostEqual(md.get_molecule_rmsd(self.atoms, moved, [90, 7]), 0)
        self.assertAlmostEqual(md.get_frame_rmsd(self.atoms, moved), 3)
        assert_allclose(md.get_molecule_displacement(self.atoms, moved, [90, 7]), [3, 0, 0])

    def test_ballistic_msd_and_explicit_diffusion_fit(self):
        times = np.arange(6.)
        positions = np.zeros((6, 2, 3))
        positions[:, :, 0] = times[:, None] * [1, 2]
        result = md.get_molecular_msd(positions, times, unwrapped=True, selection="CPD")
        assert_allclose(result["msd_angstrom2"], 2.5 * times ** 2)
        # A constructed linear MSD checks the estimator, not physical diffusion.
        result["msd_angstrom2"] = 12 * times + 0.5
        fit = md.get_molecular_diffusion_coefficient(result, fit_start_ps=1, fit_end_ps=4)
        self.assertAlmostEqual(fit["diffusion_angstrom2_per_ps"], 2)
        self.assertAlmostEqual(fit["diffusion_cm2_per_s"], 2e-4)
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            md.get_atomic_msd(positions, times, unwrapped=False, selection="C")
        with self.assertRaisesRegex(ValueError, "uniform"):
            md.get_atomic_msd(positions, [0, 1, 2, 3, 4, 6], unwrapped=True, selection="C")

    def test_correlations_known_alternating_signal(self):
        vectors = np.zeros((4, 1, 3))
        vectors[:, 0, 0] = [2, -2, 2, -2]
        result = md.get_atomic_force_autocorrelation(vectors, np.arange(4), selection="C")
        assert_allclose(result["correlation"], [4, -4, 4, -4])
        normalized = md.get_molecular_velocity_autocorrelation(
            vectors, np.arange(4), selection="CPD", normalize=True)
        assert_allclose(normalized["correlation"], [1, -1, 1, -1])
        with self.assertRaisesRegex(ValueError, "zero"):
            md.get_atomic_velocity_autocorrelation(np.ones((4, 1, 3)), np.arange(4),
                                                  selection="C", subtract_mean=True, normalize=True)

    def test_toroidal_com_unwrapping_uses_destination_box(self):
        wrapped = np.zeros((3, 1, 3))
        wrapped[:, 0, 0] = [4.8, -4.8, 5.7]
        lengths = np.array([[10., 10, 10], [10., 10, 10], [12., 12, 12]])
        result = md.unwrap_orthorhombic_com(wrapped, lengths)
        assert_allclose(result[:, 0, 0], [4.8, 5.2, 3.7])
        # Constant-cell crossings reduce to ordinary minimum-image increments.
        wrapped[:, 0, 0] = [4.8, -4.8, -4.4]
        assert_allclose(md.unwrap_orthorhombic_com(wrapped, np.full((3,3),10.))[:,0,0],
                        [4.8, 5.2, 5.6])
        with self.assertRaisesRegex(ValueError, "origin-centered"):
            md.unwrap_orthorhombic_com(wrapped + 20, lengths)

    def test_pair_lifetimes_order_gaps_and_censoring(self):
        a = ([90, 7], [33, 4])
        reordered = ([4, 33], [7, 90])
        pairs = [[a], [reordered, a], [], [a], [], []]
        episodes = md.get_pair_lifetime(iter(pairs), np.arange(6) * .5)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].detected_frames, 2)
        self.assertEqual(episodes[0].observed_span_ps, .5)
        self.assertTrue(episodes[0].left_censored)
        self.assertFalse(episodes[0].right_censored)
        self.assertEqual(episodes[1].observed_span_ps, 0)
        joined = md.get_pair_lifetime(pairs, np.arange(6) * .5, max_gap_frames=1)
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0].observed_span_ps, 1.5)
        self.assertEqual(joined[0].detected_frames, 3)
        self.assertFalse(joined[0].right_censored)
        cropped = md.get_pair_lifetime(pairs[:4], np.arange(4) * .5, max_gap_frames=1)
        self.assertTrue(cropped[0].right_censored)
        self.assertEqual(md.get_pair_lifetime([[], []], [0, 1]), [])

    def test_pair_validation_and_irregular_times(self):
        pair = ([1], [2])
        result = md.get_pair_lifetime([[pair], [pair], []], [0, .5, 2])
        self.assertEqual(result[0].observed_span_ps, .5)
        with self.assertRaisesRegex(ValueError, "every timestamp"):
            md.get_pair_lifetime([[pair]], [0, 1])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            md.get_pair_lifetime([[([1], [1])], []], [0, 1])
        with self.assertRaisesRegex(ValueError, "increasing"):
            md.get_pair_lifetime([[], []], [1, 1])

    def test_axes_force_projections_and_cell_map(self):
        assert_allclose(md.get_molecule_orientation(self.atoms, [90, 7]), [1, 0, 0])
        projections = md.get_pair_force_projections(self.atoms, [90], [7], direction=[2, 0, 0])
        self.assertEqual(projections["difference"], -2)
        second = self.atoms.copy()
        deformation = np.array([[1, .2, 0], [0, 2, 0], [0, 0, 1]])
        second.set_cell(self.atoms.cell.array @ deformation)
        assert_allclose(md.get_cell_deformation(self.atoms, second), deformation)

    def test_fixture_pair_tracking_uses_persistent_ids(self):
        from MartiniFuncs.lammpsio import iter_lammps_frames
        from MartiniFuncs.geom_utils import get_all_prereactive_pairs
        frames = list(iter_lammps_frames(Path(__file__).parent / "lammpsio_tests/multiple_frames.lammpstrj",
                                        units="metal"))
        candidates = [get_all_prereactive_pairs(f.atoms, scale=1.3) for f in frames]
        times = [f.timestep * .0005 for f in frames]
        episodes = md.get_pair_lifetime([p for p, _ in candidates], times)
        self.assertGreater(len(episodes), 0)
        self.assertTrue(all(e.last_frame < len(frames) for e in episodes))
        tracked = md.get_pair_geometry_timeseries(
            [f.atoms for f in frames], [p for p, _ in candidates],
            [s for _, s in candidates], times, pair=candidates[0][0][0])
        self.assertTrue(tracked[0]["present"])
        self.assertIsNotNone(tracked[0]["geometry"])
        self.assertIsNone(tracked[0]["error"])
        self.assertEqual(len(tracked), len(frames))


if __name__ == "__main__":
    unittest.main()
