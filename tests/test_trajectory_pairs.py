"""Cached/process pair analysis agrees with independently reconstructed frames."""
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from MartiniFuncs.structure import graph
from MartiniFuncs.pairs import lifetimes
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_all_prereactive_pairs
from MartiniFuncs.io.lammps import iter_lammps_frames, read_lammps_frame
from MartiniFuncs.chemistry.cpd_n4t.selection import PairCriteria, assess_prereactive_pairs
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.pairs.topology import FixedPairTopology


FIXTURES = Path(__file__).parent / "lammpsio_tests"


def pair_ids(frames):
    """Compare candidate order and molecular identity, not graph traversal order."""
    return [[tuple(tuple(sorted(c)) for c in pair) for pair in pairs] for pairs in frames]


def whole_positions(atoms, bonded, components):
    """Independent graph-based reference: r_whole=r+n @ cell per molecule."""
    rows = dict(zip(atoms.arrays["global_index"], range(len(atoms))))
    whole = atoms.positions.copy()
    for component in components:
        _, xyz, _ = graph.place_component(atoms, bonded, component, strict=True)
        whole[[rows[i] for i in component]] = xyz
    return whole


class TrajectoryPairTests(unittest.TestCase):
    def compare_frame(self, atoms, radius=3.25):
        bonded = graph.build_bond_graph(atoms, scale=1.3)
        components = graph.connected_components(bonded, atom_ids=graph.atom_ids(atoms))
        topology = FixedPairTopology.from_frame(atoms, bonded, components)
        whole = whole_positions(atoms, bonded, components)
        reference_pairs, reference_shifts = get_all_prereactive_pairs(atoms, reactive_radius=radius)
        for criteria in (None, PairCriteria(), PairCriteria(max_axis_angle=45, max_plane_angle=5)):
            pairs, shifts, reports = topology.evaluate(whole, atoms.cell.array, atoms.pbc,
                                                      criteria=criteria, reactive_radius=radius)
            if criteria is None:
                self.assertEqual(pairs, reference_pairs)
                self.assertEqual(shifts, reference_shifts)
                continue
            reference = assess_prereactive_pairs(atoms, reference_pairs, reference_shifts, criteria=criteria)
            self.assertEqual(len(reports), len(reference))
            for actual, expected in zip(reports, reference):
                self.assertEqual(actual.reasons, expected.reasons)
                self.assertEqual(actual.accepted, expected.accepted)
                if expected.geometry is None:
                    self.assertIsNone(actual.geometry)
                    continue
                a, b = actual.geometry, expected.geometry
                for attr in ("forming_bonds", "forming_bonds_local", "forming_bonds_frame_local"):
                    self.assertEqual(getattr(a, attr), getattr(b, attr))
                assert_allclose(a.forming_distances, b.forming_distances, atol=1e-10)
                assert_allclose([a.axis_angle, a.plane_angle, a.approach_angle],
                                [b.axis_angle, b.plane_angle, b.approach_angle], atol=1e-9)
            accepted = [r.pair_index for r in reference if r.accepted]
            self.assertEqual(pairs, [reference_pairs[i] for i in accepted])
            self.assertEqual(shifts, [reference_shifts[i] for i in accepted])

    def test_geometry_identity_shifts_and_rejections(self):
        atoms = read_lammps_frame(FIXTURES / "frame_0.lammpstrj", units="metal").atoms
        self.compare_frame(atoms)
        # Sparse IDs and shuffled ASE rows must not become local-index lookups.
        atoms.arrays["global_index"] = atoms.arrays["global_index"] * 7 + 103
        atoms = atoms[np.random.default_rng(12).permutation(len(atoms))]
        atoms.positions += np.array([1, -2, 1]) @ atoms.cell.array
        self.compare_frame(atoms)

    def test_triclinic_cell(self):
        atoms = read_lammps_frame(FIXTURES / "frame_0.lammpstrj", units="metal").atoms
        cell = atoms.cell.array.copy()
        cell[1, 0] += .2
        atoms.set_cell(cell, scale_atoms=True)
        self.compare_frame(atoms)

    def test_cached_sites_and_geometry_across_all_fixture_frames(self):
        topology = None
        for frame in iter_lammps_frames(FIXTURES / "multiple_frames.lammpstrj", units="metal"):
            atoms = frame.atoms
            bonded = graph.build_bond_graph(atoms, scale=1.3)
            components = graph.connected_components(bonded, atom_ids=graph.atom_ids(atoms))
            if topology is None:
                topology = FixedPairTopology.from_frame(atoms, bonded, components)
            whole = whole_positions(atoms, bonded, components)
            pairs, shifts, reports = topology.evaluate(whole, atoms.cell.array, atoms.pbc, criteria=PairCriteria())
            rp, rs = get_all_prereactive_pairs(atoms)
            reference = assess_prereactive_pairs(atoms, rp, rs)
            self.assertEqual(len(reports), len(reference))
            for actual, expected in zip(reports, reference):
                self.assertEqual(actual.reasons, expected.reasons)
                a, b = actual.geometry, expected.geometry
                if b is None:
                    self.assertIsNone(a)
                    continue
                self.assertEqual(a.forming_bonds, b.forming_bonds)
                self.assertEqual(a.forming_bonds_frame_local, b.forming_bonds_frame_local)
                assert_allclose(a.forming_distances, b.forming_distances, atol=1e-10)
                assert_allclose([a.axis_angle, a.plane_angle, a.approach_angle],
                                [b.axis_angle, b.plane_angle, b.approach_angle], atol=1e-9)
                # Local pair rows follow each backend's component ordering.
                for geom, pair in ((a, pairs[actual.pair_index]), (b, rp[expected.pair_index])):
                    ids = pair[0] + pair[1]
                    self.assertEqual(tuple((ids[i], ids[j]) for i, j in geom.forming_bonds_local), geom.forming_bonds)

    def test_process_order_striding_and_lifetimes(self):
        with tempfile.TemporaryDirectory() as directory:
            shutil.copyfile(FIXTURES / "multiple_frames.lammpstrj", Path(directory) / "trajectory.lammpstrj")
            options = dict(units="metal", timestep_ps=.0005, compute_pairs=True,
                           pair_criteria=PairCriteria(max_axis_angle=60), frame_stride=2)
            reference = analyze_run(directory, pair_backend="reference", **options)
            for workers, backend in ((1, 'cached'), (2, 'cached'), (2, 'reference')):
                result = analyze_run(directory, pair_workers=workers, pair_backend=backend, pair_batch_size=2, **options)
                self.assertEqual(pair_ids(result["pairs_by_frame"]), pair_ids(reference["pairs_by_frame"]))
                self.assertEqual(result["pair_shifts_by_frame"], reference["pair_shifts_by_frame"])
                self.assertEqual(result["frames"], reference["frames"])
                assert_array_equal(result["com_angstrom"], reference["com_angstrom"])
                times = [r["time_ps"] for r in result["frames"]]
                self.assertEqual(lifetimes.get_pair_lifetime(result["pairs_by_frame"], times),
                                 lifetimes.get_pair_lifetime(reference["pairs_by_frame"], times))

    def test_worker_failure_propagates_in_order(self):
        """The production pool (trajectory.parallel.ordered_tasks) yields results in task
        order and re-raises a worker's exception at that task's position."""
        from MartiniFuncs.trajectory.parallel import ordered_tasks
        results = ordered_tasks(int, ["1", "2", "not a number", "4"], workers=2)
        self.assertEqual([next(results), next(results)], [1, 2])
        with self.assertRaisesRegex(ValueError, "not a number"):
            next(results)
        self.assertEqual(list(ordered_tasks(int, ["5", "6"], workers=1)), [5, 6])

    def test_configuration_validation(self):
        for options in ({"pair_workers": 0}, {"pair_workers": True},
                        {"pair_batch_size": 0}, {"pair_backend": "unknown"}):
            with self.assertRaises(ValueError):
                analyze_run("unused", **options)


if __name__ == "__main__":
    unittest.main()
