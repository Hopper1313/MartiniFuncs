"""Dynamics diagnostics checked against cases with known analytic answers."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs.statistics import dynamics as dy
from MartiniFuncs.structure import periodic
from MartiniFuncs.trajectory.parallel import principal_axes

FIXTURES = Path(__file__).parent / "lammpsio_tests"


def random_walk(rng, frames, objects, sigma):
    return np.cumsum(rng.normal(scale=sigma, size=(frames, objects, 3)), axis=0)


class NonAffineTests(unittest.TestCase):
    def sheared_cells(self, frames, rate, length=20.):
        return np.array([[[length, 0, 0], [rate * length * t, length, 0], [0, 0, length]] for t in range(frames)])

    def test_pure_affine_motion_has_zero_nonaffine_displacement(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(0, 1, (25, 3))
        for cells, origins in ((self.sheared_cells(60, .01), np.zeros((60, 3))),
                               # NPT-like dilation about a fixed center: origin moves by -dL/2
                               (np.array([np.eye(3) * (20 + .3 * np.sin(t)) for t in range(60)]),
                                np.array([np.full(3, -(20 + .3 * np.sin(t)) / 2) for t in range(60)]))):
            points = np.array([o + s @ h for h, o in zip(cells, origins)])
            centers = origins + cells.sum(axis=1) / 2
            wrapped = np.array([periodic.minimum_image(p - c, h) for p, c, h in zip(points, centers, cells)])
            nonaffine = dy.unwrap_nonaffine(wrapped, cells, origins)
            assert_allclose(nonaffine - nonaffine[0], 0, atol=1e-9)
            lab = periodic.unwrap_com_trajectory(wrapped + centers[:, None], cells, centers)
            self.assertGreater(np.abs(lab - lab[0]).max(), .1)   # the lab frame does see the affine motion

    def test_shear_plus_diffusion_recovers_diffusive_msd(self):
        rng = np.random.default_rng(2)
        frames, rate, sigma = 2000, .02, .05
        cells = self.sheared_cells(frames, rate)
        origins = np.zeros((frames, 3))
        s = [rng.uniform(0, 1, (200, 3))]
        points = [s[0] @ cells[0]]
        for t in range(1, frames):
            # streaming u = s dH/dt (midpoint s) plus a Brownian step
            noise = rng.normal(scale=sigma, size=(200, 3))
            guess = points[-1] + noise
            s_mid = (np.linalg.solve(cells[t - 1].T, points[-1].T).T + np.linalg.solve(cells[t].T, guess.T).T) / 2
            points.append(points[-1] + noise + s_mid @ (cells[t] - cells[t - 1]))
        points = np.array(points)
        centers = origins + cells.sum(axis=1) / 2
        wrapped = np.array([periodic.minimum_image(p - c, h) for p, c, h in zip(points, centers, cells)])
        nonaffine = dy.unwrap_nonaffine(wrapped, cells, origins)
        lag = 100
        msd = np.mean(np.sum((nonaffine[lag:] - nonaffine[:-lag]) ** 2, axis=-1))
        self.assertAlmostEqual(msd / (3 * sigma ** 2 * lag), 1, delta=.08)
        lab = periodic.unwrap_com_trajectory(wrapped + centers[:, None], cells, centers)
        lab_msd = np.mean(np.sum((lab[lag:] - lab[:-lag]) ** 2, axis=-1))
        self.assertGreater(lab_msd, 2 * msd)   # Taylor dispersion from the flow in the lab frame

    def test_analyze_run_stores_nonaffine_consistent_with_array_version(self):
        from MartiniFuncs.trajectory.analysis import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(FIXTURES / "sheared_frames.lammpstrj", tmp)
            r = analyze_run(tmp, units="metal", timestep_ps=.0005)
        again = dy.unwrap_nonaffine(r["wrapped_com_angstrom"], r["cell_angstrom"], r["cell_origin_angstrom"])
        assert_allclose(r["com_nonaffine_angstrom"], again, atol=1e-10)
        difference = r["com_angstrom"] - r["com_nonaffine_angstrom"]
        self.assertGreater(np.abs(difference[..., 0]).max(), 1e-5)            # flow is along x
        assert_allclose(difference[..., 1:], 0, atol=1e-9)


class CagingTests(unittest.TestCase):
    def test_non_gaussian_parameter(self):
        rng = np.random.default_rng(3)
        walk = random_walk(rng, 3000, 100, .1)
        times = np.arange(3000) * .1
        gaussian = dy.displacement_moments(walk, times, lags=[1, 10, 100])
        assert_allclose(gaussian["alpha2"], 0, atol=.05)
        mixed = walk.copy()
        mixed[:, :50] = 0           # half the objects never move: alpha2 = 1/f - 1 = 1
        assert_allclose(dy.displacement_moments(mixed, times, lags=[1, 10, 100])["alpha2"], 1, atol=.06)

    def test_van_hove_matches_gaussian_for_brownian_motion(self):
        rng = np.random.default_rng(4)
        walk = random_walk(rng, 2000, 100, .1)
        rows = dy.van_hove_self(walk, np.arange(2000.), lags_ps=[25.], edges=np.linspace(0, 3, 61))
        p, g = rows[0]["p_r_per_angstrom"], rows[0]["gaussian_p_r_per_angstrom"]
        self.assertAlmostEqual(np.sum(p * .05), 1, delta=.01)
        self.assertLess(np.abs(p - g).max(), .05 * g.max())

    def test_step_correlation_signatures(self):
        rng = np.random.default_rng(5)
        times = np.arange(4000.)
        rattle = rng.normal(size=(4000, 50, 3))
        assert_allclose(dy.step_correlation(rattle, times, lags=[1, 5])["chi"], -.5, atol=.02)
        assert_allclose(dy.step_correlation(random_walk(rng, 4000, 50, 1.), times, lags=[1, 5])["chi"], 0, atol=.03)

    def test_sampling_diagnostics(self):
        rng = np.random.default_rng(6)
        velocity = rng.normal(size=(1, 30, 3)).repeat(200, axis=0)
        ballistic = np.cumsum(velocity, axis=0) * .1
        diag = dy.sampling_diagnostics(ballistic, velocity, np.arange(200) * .1)
        self.assertAlmostEqual(diag["ballistic_ratio"], 1, places=6)
        caged = rng.normal(scale=.1, size=(200, 30, 3))
        fast = rng.normal(size=(200, 30, 3))
        diag = dy.sampling_diagnostics(caged, fast, np.arange(200) * .5)
        self.assertLess(diag["ballistic_ratio"], .2)
        self.assertEqual(len(diag["notes"]), 2)


class TransportTests(unittest.TestCase):
    def test_green_kubo_exponential_vacf(self):
        gamma, sigma2 = 2.0, 1.5             # C(t) = 3 sigma^2 exp(-gamma t) -> D = sigma^2 / gamma
        lag = np.linspace(0, 10, 20001)
        result = dy.green_kubo_diffusion(lag, 3 * sigma2 * np.exp(-gamma * lag), upper_ps=10)
        self.assertAlmostEqual(result["D_angstrom2_per_ps"], sigma2 / gamma, places=6)
        self.assertAlmostEqual(result["D_cm2_per_s"], 1e-4 * sigma2 / gamma, places=10)

    def test_yeh_hummer_value(self):
        expected = 1.380649e-23 * 298.15 * 2.837297 / (6 * np.pi * 8.9e-4 * 30e-10) * 1e4
        result = dy.yeh_hummer_correction(2e-5, temperature_K=298.15, viscosity_pa_s=8.9e-4, box_length_angstrom=30)
        self.assertAlmostEqual(result["correction_cm2_s"], expected, delta=1e-12)
        self.assertAlmostEqual(result["D_infinite_cm2_s"], 2e-5 + expected, delta=1e-12)
        with self.assertRaisesRegex(ValueError, "cubic"):
            dy.cubic_box_length([[20, 20, 25]])

    def test_nemd_viscosity_units_and_sign(self):
        # eta = 1 mPa s at 0.01/ps: sigma_xy = eta * gamma = 1e7 Pa = 100 bar, P_xy = -100 bar
        result = dy.nemd_shear_viscosity(np.full(100, -100.), shear_rate_per_ps=.01)
        self.assertAlmostEqual(result["viscosity_pa_s"], 1e-3)


class RotationTests(unittest.TestCase):
    def test_rotational_diffusion_correlations(self):
        rng = np.random.default_rng(7)
        frames, objects, dr, dt = 3000, 200, .05, .1
        u = np.empty((frames, objects, 3))
        u[0] = rng.normal(size=(objects, 3))
        u[0] /= np.linalg.norm(u[0], axis=1, keepdims=True)
        for t in range(1, frames):
            omega = rng.normal(scale=np.sqrt(2 * dr * dt), size=(objects, 3))
            step = u[t - 1] + np.cross(omega, u[t - 1])
            u[t] = step / np.linalg.norm(step, axis=1, keepdims=True)
        times = np.arange(frames) * dt
        for order in (1, 2):
            c = dy.orientation_correlation(u, times, order=order, nlag=200)
            expected = np.exp(-order * (order + 1) * dr * c["lag_ps"])
            assert_allclose(c["correlation"][[0, 20, 50]], expected[[0, 20, 50]], atol=.03)
        flipped = u * rng.choice([-1, 1], size=(frames, objects, 1))
        assert_allclose(dy.orientation_correlation(flipped, times, order=2, nlag=50)["correlation"],
                        dy.orientation_correlation(u, times, order=2, nlag=50)["correlation"], atol=1e-10)
        tau = dy.correlation_time(*[dy.orientation_correlation(u, times, order=1, nlag=2000)[k]
                                    for k in ("lag_ps", "correlation")])
        self.assertAlmostEqual(tau["tau_ps"], 1 / (2 * dr), delta=1.5)

    def test_principal_axes_of_planar_and_rod_molecules(self):
        ring = np.array([[np.cos(a), np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]) * [2, 1, 1]
        rod = np.array([[x, 0, 0] for x in range(5)], dtype=float)
        whole = np.vstack([ring, rod])
        groups = np.array([0] * 6 + [1] * 5)
        masses = np.ones(11)
        com = np.array([ring.mean(0), rod.mean(0)])
        axes = principal_axes(whole, com, groups, masses, 2)
        assert_allclose(np.abs(axes[0, 0]), [1, 0, 0], atol=1e-12)   # ring elongated along x
        assert_allclose(np.abs(axes[0, 1]), [0, 0, 1], atol=1e-12)   # ring normal
        assert_allclose(np.abs(axes[1, 0]), [1, 0, 0], atol=1e-12)   # rod axis


class FreeEnergyTests(unittest.TestCase):
    def test_pmf_inverts_boltzmann_factor(self):
        r = np.linspace(3, 10, 50)
        w = .02 * np.cos(r)
        result = dy.pmf_from_rdf(r, np.exp(-w / (dy.KB_EV * 300)), temperature_K=300)
        assert_allclose(result["pmf_eV"], w, atol=1e-12)

    def test_isotropic_angles_give_flat_angular_pmf(self):
        rng = np.random.default_rng(8)
        v = rng.normal(size=(200000, 3))
        angle = np.degrees(np.arccos(np.abs(v[:, 2]) / np.linalg.norm(v, axis=1)))   # acute angle to z
        result = dy.pmf_1d(angle, temperature_K=300, edges=np.linspace(5, 85, 17), jacobian="sin")
        self.assertLess(np.ptp(result["pmf_eV"]), 3 * dy.KB_EV * 300 * .05)


class TopologySegmentTests(unittest.TestCase):
    def make_reacting_dump(self, tmp, event_frame=4, flicker_frame=2):
        """multiple_frames with one CPD moved onto a dienophile carbon from event_frame on,
        and for one isolated flicker frame (a transient contact that must be ignored)."""
        from MartiniFuncs.structure import graph
        from MartiniFuncs.io.lammps import iter_lammps_frames, _iter_blocks
        source = FIXTURES / "multiple_frames.lammpstrj"
        frames = list(iter_lammps_frames(source, units="metal"))
        atoms = frames[0].atoms
        bonded = graph.build_bond_graph(atoms, scale=1.3)
        components = graph.connected_components(bonded, atom_ids=graph.atom_ids(atoms))
        cpd = next(c for c in components if len(c) == 11)
        mal = next(c for c in components if len(c) == 23)
        moved = []
        for frame in frames:
            a = frame.atoms
            b = graph.build_bond_graph(a, scale=1.3)
            _, whole_cpd = graph.unwrap_component(a, b, cpd)
            _, whole_mal = graph.unwrap_component(a, b, mal)
            carbon = next(k for k, i in enumerate(cpd) if a.numbers[i] == 6)
            target = whole_mal[next(k for k, i in enumerate(mal) if a.numbers[i] == 6)]
            directions = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                                   if (i, j, k) != (0, 0, 0)], dtype=float)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            for offset in (directions[:, None, :] * np.array([1.6, 1.7, 1.8])[None, :, None]).reshape(-1, 3):
                trial = a.copy()
                trial.positions[cpd] = whole_cpd + (target + offset - whole_cpd[carbon])
                sizes = [len(c) for c in graph.connected_components(graph.build_bond_graph(trial, scale=1.3),
                                                                    atom_ids=graph.atom_ids(trial))]
                if sorted(sizes).count(34) == 1 and len(sizes) == len(components) - 1:
                    break
            else:
                raise AssertionError("could not place a clean 34-atom product")
            moved.append(trial.positions[cpd])
        lines_out = []
        for index, block in enumerate(_iter_blocks(source)):
            lines = block.splitlines()
            header = next(k for k, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))
            names = lines[header].split()[2:]
            ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
            if index >= event_frame or index == flicker_frame:
                rows = {int(line.split()[0]) - 1: k for k, line in enumerate(lines[header + 1:], header + 1)}
                for atom_id, xyz in zip(cpd, moved[index]):
                    tokens = lines[rows[atom_id]].split()
                    tokens[ix], tokens[iy], tokens[iz] = (repr(float(v)) for v in xyz)
                    if "ix" in names:
                        for name in ("ix", "iy", "iz"):
                            tokens[names.index(name)] = "0"
                    lines[rows[atom_id]] = " ".join(tokens)
            lines_out.append("\n".join(lines) + "\n")
        (Path(tmp) / "react.lammpstrj").write_text("".join(lines_out))
        return cpd, mal

    def test_segments_split_at_persistent_reaction_only(self):
        from MartiniFuncs.trajectory.analysis import analyze_run, analyze_segments
        with tempfile.TemporaryDirectory() as tmp:
            cpd, mal = self.make_reacting_dump(tmp)
            out = analyze_segments(tmp, units="metal", timestep_ps=.0005, topology_persistence=3)
            self.assertEqual(len(out["segments"]), 2)
            self.assertEqual([e["frame"] for e in out["events"]], [4])           # the frame-2 flicker is ignored
            self.assertEqual(out["events"][0]["formulas"], ["C16H15NO2"])
            first, second = out["segments"]
            self.assertEqual([r["frame"] for r in first["frames"]], [0, 1, 2, 3])
            self.assertEqual(first["source_range"], [0, 4])
            self.assertEqual(second["frames"][0]["frame"], 4)
            self.assertIn("C16H15NO2", set(second["species"]))
            self.assertEqual(len(second["species"]), 79)
            recorded = analyze_run(tmp, units="metal", timestep_ps=.0005, on_topology_change="record",
                                   topology_persistence=1)
            self.assertEqual([e["frame"] for e in recorded["topology_events"]], [2, 4])
            self.assertEqual(len(recorded["frames"]), 9)

    def test_source_range_matches_full_run(self):
        from MartiniFuncs.trajectory.analysis import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(FIXTURES / "multiple_frames.lammpstrj", tmp)
            full = analyze_run(tmp, units="metal", timestep_ps=.0005)
            part = analyze_run(tmp, units="metal", timestep_ps=.0005, source_range=(3, 7))
            parallel = analyze_run(tmp, units="metal", timestep_ps=.0005, source_range=(3, 7), pair_workers=2,
                                   pair_batch_size=2)
        self.assertEqual([r["frame"] for r in part["frames"]], [3, 4, 5, 6])
        self.assertEqual(part["frames"], full["frames"][3:7])
        assert_allclose(part["com_angstrom"] - part["com_angstrom"][0],
                        full["com_angstrom"][3:7] - full["com_angstrom"][3], atol=1e-9)
        self.assertEqual(parallel["frames"], part["frames"])
        assert_allclose(parallel["com_angstrom"], part["com_angstrom"])


if __name__ == "__main__":
    unittest.main()
