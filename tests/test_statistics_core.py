"""Shared statistics core: FFT lag estimators, general-cell COM unwrapping, validators."""
import tempfile
import unittest
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from MartiniFuncs import md_statistics as md
from MartiniFuncs.stability import (_frame_distances, get_molecular_com_rdf, get_series_statistics,
                                    scalar_autocorrelation)
from MartiniFuncs.thermo import attach_thermo

FIXTURES = Path(__file__).parent / "lammpsio_tests"


def loop_msd(x, nlag):
    return np.array([0.] + [np.mean(np.sum((x[k:] - x[:-k])**2, axis=-1)) for k in range(1, nlag)])


def loop_correlation(x, nlag):
    return np.array([np.mean(np.sum(x[:len(x)-k] * x[k:], axis=-1)) for k in range(nlag)])


class LagEstimatorTests(unittest.TestCase):
    def test_fft_estimators_match_explicit_lag_loops(self):
        rng = np.random.default_rng(3)
        # Large absolute offsets check the mean shift that limits FFT cancellation.
        positions = 500 + np.cumsum(rng.normal(size=(257, 7, 3)), axis=0)
        for nlag in (1, 2, 40, 257):
            assert_allclose(md.lag_msd(positions, nlag), loop_msd(positions, nlag), rtol=1e-10, atol=1e-10)
        vectors = rng.normal(size=(300, 5, 3)) + 2
        assert_allclose(md.lag_correlation(vectors, 50), loop_correlation(vectors, 50), rtol=1e-10, atol=1e-12)
        centered = vectors - vectors.mean(axis=0)
        assert_allclose(md.lag_correlation(vectors, 50, subtract_mean=True), loop_correlation(centered, 50),
                        rtol=1e-10, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "nlag"):
            md.lag_msd(positions, 0)

    def test_lag_count_keeps_representable_endpoint(self):
        self.assertEqual(md.lag_count(0.3, 0.1, 100), 4)
        self.assertEqual(md.lag_count(0.29, 0.1, 100), 3)
        self.assertEqual(md.lag_count(50, 0.1, 10), 10)

    def test_shared_timestamp_validator(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            md.validate_times([0, 1], minimum=3)
        with self.assertRaisesRegex(ValueError, "uniform"):
            md.validate_times([0, 1, 3], uniform=True)
        with self.assertRaisesRegex(ValueError, "increasing"):
            md.validate_times([0, 1, 1])

    def test_scalar_acf_matches_direct_estimator(self):
        x = np.random.default_rng(9).normal(size=500).cumsum()
        c = x - x.mean()
        direct = np.correlate(c, c, "full")[len(x)-1:] / (np.arange(len(x), 0, -1) * np.mean(c**2))
        assert_allclose(scalar_autocorrelation(x), direct, rtol=1e-9, atol=1e-12)
        self.assertIsNone(scalar_autocorrelation(np.ones(10)))
        self.assertGreaterEqual(get_series_statistics(x, sample_interval_ps=1.)["statistical_inefficiency_first_zero"], 1)


class CellGeometryTests(unittest.TestCase):
    def setUp(self):
        self.cell = np.array([[10., 0, 0], [3., 9, 0], [1., 2., 11]])

    def test_minimum_image_orthorhombic_matches_componentwise_tor(self):
        rng = np.random.default_rng(1)
        d = rng.uniform(-30, 30, (50, 3))
        L = np.array([10., 12., 9.])
        assert_allclose(md.minimum_image(d, np.diag(L)), d - L*np.floor(d/L + .5))

    def test_triclinic_trajectory_crossing_every_face_is_recovered(self):
        rng = np.random.default_rng(2)
        steps = rng.normal(scale=.4, size=(400, 6, 3))
        cells = np.array([self.cell + [[0, 0, 0], [0.01*t, 0, 0], [0, 0, 0]] for t in range(400)])  # growing tilt
        # As in LAMMPS (fix deform remap v): step inside the current cell, then
        # wrap by that cell's lattice vectors when a face is crossed.
        wrapped = [md.minimum_image(rng.uniform(-5, 5, (6, 3)), cells[0])]
        for step, cell in zip(steps[1:], cells[1:]):
            wrapped.append(md.minimum_image(wrapped[-1] + step, cell))
        unwrapped = md.unwrap_com_trajectory(np.asarray(wrapped), cells)
        assert_allclose(unwrapped[1:] - unwrapped[:-1], steps[1:], atol=1e-9)
        self.assertGreater(np.abs(unwrapped - unwrapped[0]).max(), 10)   # really crossed boundaries

    def test_orthorhombic_wrapper_unchanged(self):
        wrapped = np.zeros((3, 1, 3))
        wrapped[:, 0, 0] = [4.8, -4.8, 5.7]
        lengths = np.array([[10., 10, 10], [10., 10, 10], [12., 12, 12]])
        assert_allclose(md.unwrap_orthorhombic_com(wrapped, lengths)[:, 0, 0], [4.8, 5.2, 3.7])

    def test_triclinic_rdf_distances_are_true_minimum_images(self):
        rng = np.random.default_rng(4)
        delta = rng.uniform(-25, 25, (200, 3))
        shifts = np.array([[i, j, k] for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)]) @ self.cell
        brute = np.linalg.norm(delta[:, None, :] + shifts[None], axis=-1).min(axis=1)
        limit = md.cell_plane_spacings(self.cell).min() / 2
        found = _frame_distances(delta, self.cell)
        assert_allclose(found[brute < limit], brute[brute < limit])
        self.assertTrue(np.all(found >= brute - 1e-12))

    def test_rdf_accepts_cells_and_matches_lengths_for_orthorhombic(self):
        rng = np.random.default_rng(6)
        positions = rng.uniform(-10, 10, (20, 30, 3))
        lengths = np.full((20, 3), 20.)
        cells = np.stack([np.diag(L) for L in lengths])
        edges = np.linspace(0, 9.9, 12)
        labels = ["A"]*15 + ["B"]*15
        a = get_molecular_com_rdf(positions, lengths, labels, species_a="A", species_b="B", edges=edges)
        b = get_molecular_com_rdf(positions, cells, labels, species_a="A", species_b="B", edges=edges)
        assert_allclose(a["g_r"], b["g_r"])
        with self.assertRaisesRegex(ValueError, "half"):
            get_molecular_com_rdf(positions, np.stack([self.cell]*20), labels, species_a="A", species_b="B",
                                  edges=np.linspace(0, 6, 4))


class AnalysisFixesTests(unittest.TestCase):
    def test_sheared_cell_com_tracking_matches_image_flags(self):
        from MartiniFuncs.lammpsio import iter_lammps_frames
        from MartiniFuncs.trajectory_analysis import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(FIXTURES / "sheared_frames.lammpstrj", tmp)
            result = analyze_run(tmp, units="metal", timestep_ps=.0005)
        self.assertIsNotNone(result["com_angstrom"])
        self.assertEqual(result["cell_angstrom"].shape, (10, 3, 3))
        self.assertTrue(any("shear flow" in notice for notice in result["notices"]))
        groups = np.empty(1360, dtype=int)
        for index, ids in enumerate(result["molecule_atom_ids"]):
            groups[ids] = index
        roots = [ids[0] for ids in result["molecule_atom_ids"]]
        image, root_flags = [], []
        for frame in iter_lammps_frames(FIXTURES / "sheared_frames.lammpstrj", units="metal"):
            atoms = frame.atoms
            r = atoms.positions + atoms.arrays["image_flags"] @ atoms.cell.array
            m = atoms.get_masses()
            image.append(np.column_stack([np.bincount(groups, weights=m*r[:, a]) / np.bincount(groups, weights=m)
                                          for a in range(3)]))
            root_flags.append(atoms.arrays["image_flags"][roots])
        image, root_flags, cells = np.asarray(image), np.asarray(root_flags), result["cell_angstrom"]
        # Image-flag unwrapping q = r + n H_t also moves each image row with the
        # deforming cell; TOR omits that term: du_t = dq_t - n_(t-1) (H_t - H_(t-1)).
        # Image flags themselves record crossings made at earlier tilts, so the
        # identity holds to within one per-step cell change (4.5e-4 A here), plus
        # one more where a COM lies just outside the centered cell (TOR wraps it in).
        tor_steps = np.diff(result["com_angstrom"], axis=0)
        expected = np.diff(image, axis=0) - np.einsum("tma,tab->tmb", root_flags[:-1], np.diff(cells, axis=0))
        bound = 3*np.abs(np.diff(cells, axis=0)).max()
        self.assertLess(np.abs(tor_steps - expected).max(), bound)
        self.assertLess(bound, 2e-3)
        self.assertGreater(np.abs(root_flags).max(), 0)   # molecules have crossed faces
        self.assertLess(result["max_com_step_cell_fraction"], .05)

    def test_pair_lifetime_censoring_uses_gap_and_rejects_bool(self):
        pair = ([1], [2])
        episodes = md.get_pair_lifetime([[], [pair], [pair], [], [], []], np.arange(6.), max_gap_frames=1)
        self.assertTrue(episodes[0].left_censored)   # could bridge to an unseen detection before frame 0
        self.assertFalse(md.get_pair_lifetime([[], [pair], [], []], np.arange(4.))[0].left_censored)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            md.get_pair_lifetime([[pair]], [0.], max_gap_frames=True)

    def test_thermo_join_prefixes_colliding_columns(self):
        frames = pd.DataFrame(dict(timestep=[0, 10], time_ps=[0., .005], temperature_K=[399., 401.]))
        thermo = pd.DataFrame(dict(timestep=[0, 10], temperature_K=[400., 402.], pressure_bar=[1., 2.],
                                   thermo_block=[0, 0]))
        joined, coverage = attach_thermo(frames, thermo)
        assert_allclose(joined["temperature_K"], [399., 401.])
        assert_allclose(joined["thermo_temperature_K"], [400., 402.])
        self.assertEqual(coverage["renamed"], {"temperature_K": "thermo_temperature_K"})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            attach_thermo(frames, thermo, on_conflict="error")


if __name__ == "__main__":
    unittest.main()
