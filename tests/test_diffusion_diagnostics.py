"""Known stochastic references and failure cases for scientific diagnostics."""
import unittest
import numpy as np
from MartiniFuncs.diffusion_diagnostics import diffusion_diagnostics, scalar_stationarity


class DiffusionDiagnosticsTests(unittest.TestCase):
    def test_brownian_reference_and_units(self):
        rng = np.random.default_rng(44)
        diffusion = .12
        p = np.cumsum(rng.normal(scale=np.sqrt(2*diffusion), size=(6000, 60, 3)), axis=0)
        result = diffusion_diagnostics(p, np.arange(len(p)),
            fit_intervals_ps=[(2, 6), (4, 8), (6, 10)], block_frames=300, resamples=300)
        self.assertEqual(result["blocks"], 20)
        for fit in result["fits"]:
            self.assertEqual(fit["status"], "candidate", fit["reasons"])
            self.assertAlmostEqual(fit["D_m2_s"]/1e-8, diffusion, delta=.015)
            self.assertLess(fit["ci95_m2_s"][0], fit["D_m2_s"])
            self.assertGreater(fit["ci95_m2_s"][1], fit["D_m2_s"])

    def test_confined_motion_is_not_diffusive(self):
        p = np.random.default_rng(5).normal(size=(1600, 30, 3))
        result = diffusion_diagnostics(p, np.arange(len(p)),
            fit_intervals_ps=[(2, 6), (4, 8), (6, 10)], block_frames=100, resamples=200)
        self.assertTrue(all(f["status"] != "candidate" for f in result["fits"]))
        self.assertTrue(all("MSD scaling is not close to linear in lag" in f["reasons"] for f in result["fits"]))

    def test_short_and_invalid_data(self):
        p = np.zeros((9, 2, 3))
        result = diffusion_diagnostics(p, np.arange(9),
            fit_intervals_ps=[(10, 20)], block_frames=100, resamples=100)
        self.assertEqual(result["fits"][0]["status"], "insufficient sampling")
        self.assertEqual(result["blocks"], 0)
        with self.assertRaisesRegex(ValueError, "uniform"):
            diffusion_diagnostics(p, np.arange(9)**2, fit_intervals_ps=[(1, 3)], block_frames=10)

    def test_changing_mobility_is_rejected(self):
        rng = np.random.default_rng(24)
        increments = rng.normal(size=(4000, 50, 3))
        increments[2000:] *= 3
        p = increments.cumsum(axis=0)
        result = diffusion_diagnostics(p, np.arange(len(p)),
            fit_intervals_ps=[(2, 6), (4, 8), (6, 10)], block_frames=200, resamples=200)
        self.assertTrue(all("Earlier and later block estimates disagree" in f["reasons"] for f in result["fits"]))

    def test_scalar_equivalence_drift_and_missing_tolerance(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=4000)
        args = dict(times_ps=np.arange(len(x)), block_frames=200, resamples=300)
        stable = scalar_stationarity(x, tolerance=.2, **args)
        self.assertEqual(stable["status"], "supports stationarity", stable["reasons"])
        drift = scalar_stationarity(x+np.linspace(0, 4, len(x)), tolerance=.2, **args)
        self.assertNotEqual(drift["status"], "supports stationarity")
        self.assertNotEqual(scalar_stationarity(x, **args)["status"], "supports stationarity")
        self.assertNotEqual(scalar_stationarity(np.ones(len(x)), tolerance=.2, **args)["status"], "supports stationarity")


if __name__ == "__main__":
    unittest.main()
