import unittest

import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs.statistics.stability import get_molecular_com_rdf, get_series_statistics


class StabilityTests(unittest.TestCase):
    def test_periodic_distance_and_npt_normalization(self):
        positions = np.array([[[0, 0, 0], [9, 0, 0]], [[0, 0, 0], [19, 0, 0]]])
        lengths = np.array([[10, 10, 10], [20, 20, 20]])
        edges = np.array([0, .5, 1.5, 4.])
        shell = 4*np.pi/3*(1.5**3-.5**3)
        for labels, b in [(["A", "A"], "A"), (["A", "B"], "B")]:
            result = get_molecular_com_rdf(positions, lengths, labels, species_a="A", species_b=b, edges=edges)
            assert_allclose(result["nearest_distances_angstrom"], 1.)
            assert_allclose(result["g_r"], [0, 4500/shell, 0])
        with self.assertRaises(ValueError):
            get_molecular_com_rdf(positions, lengths, ["A", "B"], species_a="A", species_b="B", edges=[0, 6])

    def test_ideal_gas_rdf_baseline_and_self_exclusion(self):
        rng = np.random.default_rng(18)
        positions = rng.uniform(0, 20, (100, 60, 3))
        result = get_molecular_com_rdf(positions, np.full((100, 3), 20.), ["A"]*60,
                                      species_a="A", species_b="A", edges=[2,4,6,8,10])
        # Finite random sample: generous tolerance, still detects pair-factor errors.
        assert_allclose(result["g_r"], 1, atol=.06)
        self.assertTrue(np.all(result["nearest_distances_angstrom"] > 0))

    def test_batch_mean_uncertainty_and_trailing_samples(self):
        values = np.array([0,0,2,2,4,4,6,6,100.])
        result = get_series_statistics(values, sample_interval_ps=.5, batch_sizes=(2,))
        batch = result["batch_mean_errors"][0]
        assert_allclose(batch["standard_error"], np.std([0,2,4,6], ddof=1)/2)
        self.assertEqual(batch["discarded_samples"], 1)
        self.assertEqual(batch["batch_duration_ps"], 1.)
        constant = get_series_statistics(np.ones(20), sample_interval_ps=.5, batch_sizes=(2,))
        self.assertIsNone(constant["effective_samples_heuristic"])
        self.assertEqual(constant["batch_mean_errors"][0]["standard_error"], 0)


if __name__ == "__main__":
    unittest.main()
