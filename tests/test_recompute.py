"""Recompute exported pairs/clusters: fragments, F_isolated/F_env, energies, relaxations, CLI.

ASE's EMT calculator stands in for UMA; every recomputed value is checked
against an independent EMT evaluation of the same fragment built by hand.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read

from MartiniFuncs.chemistry.cpd_n4t.pairs import get_all_prereactive_pairs
from MartiniFuncs.io.extxyz import write_extxyz
from MartiniFuncs.io.lammps import read_lammps_frame
from MartiniFuncs.pairs.structures import (get_cluster_positions_and_forces, get_pair_positions_and_forces,
                                           get_pair_shell_clusters, pair_data_to_extxyz)
from MartiniFuncs.recompute.driver import recompute_file
from MartiniFuncs.recompute.evaluation import FragmentEvaluator, FragmentRelaxer
from MartiniFuncs.recompute.fragments import Fragments
from MartiniFuncs.recompute.tasks import ForceDecomposition, FragmentRelaxation, FrozenShellOptimization
from MartiniFuncs.workflows.cli import main

FIXTURES = Path(__file__).parent / "lammpsio_tests"


def emt(structure, rows):
    """Independent reference: EMT energy and forces of the rows as a new isolated molecule."""
    fragment = Atoms(np.asarray(structure.get_chemical_symbols())[rows].tolist(), positions=structure.positions[rows])
    fragment.calc = EMT()
    return fragment.get_potential_energy(), fragment.get_forces()


class RecomputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        frame = read_lammps_frame(FIXTURES / "frame_0.lammpstrj", units="metal")
        atoms = frame.atoms
        pairs, shifts = get_all_prereactive_pairs(atoms)
        clusters = get_pair_shell_clusters(atoms, pairs, shifts, shell_radius=3.0)
        cls.clusters_file, cls.pairs_file = cls.root / "clusters.extxyz", cls.root / "pairs.extxyz"
        pair_data_to_extxyz(get_cluster_positions_and_forces(atoms, frame, clusters), cls.clusters_file,
                            role_labels=("CPD", "MAL"))
        pair_data_to_extxyz(get_pair_positions_and_forces(atoms, frame, pairs, shifts), cls.pairs_file,
                            role_labels=("CPD", "MAL"))
        cls.clusters = read(cls.clusters_file, index=":")
        cls.pairs = read(cls.pairs_file, index=":")
        cls.evaluator = FragmentEvaluator(EMT())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fragments_from_pair_roles_and_bond_graph(self):
        cluster = self.clusters[0]
        fragments = Fragments.from_structure(cluster)
        roles = cluster.arrays["pair_role"]
        assert_array_equal(fragments.first, np.flatnonzero(roles == "CPD"))
        assert_array_equal(fragments.second, np.flatnonzero(roles == "MAL"))
        self.assertGreater(len(fragments.shell), 0)
        shell_rows = np.concatenate(fragments.shell)
        assert_array_equal(np.sort(shell_rows), np.flatnonzero(roles == "shell"))   # each shell atom exactly once
        self.assertTrue(all(len(rows) in (11, 23) for rows in fragments.shell))    # whole CPD / N4T molecules
        self.assertEqual(Fragments.from_structure(self.pairs[0]).shell, ())
        stripped = cluster.copy()
        del stripped.arrays["pair_role"]
        with self.assertRaisesRegex(ValueError, "no pair_role"):
            Fragments.from_structure(stripped)

    def test_pair_mode_forces_energies_and_environment(self):
        cluster = self.clusters[0]
        fragments = Fragments.from_structure(cluster)
        out = ForceDecomposition(self.evaluator).apply(cluster).structure
        energy, forces = emt(cluster, fragments.pair)
        self.assertAlmostEqual(out.info["energy_pair"], energy, places=12)
        assert_allclose(out.arrays["isolated_forces"][fragments.pair], forces, atol=1e-12)
        for k, rows in enumerate(fragments.shell, start=1):
            shell_energy, shell_forces = emt(cluster, rows)
            assert_allclose(out.arrays["isolated_forces"][rows], shell_forces, atol=1e-12)
            self.assertAlmostEqual(out.info["energy_shell"][k - 1], shell_energy, places=12)
            self.assertTrue((out.arrays["isolated_fragment"][rows] == k).all())
        self.assertTrue((out.arrays["isolated_fragment"][fragments.pair] == 0).all())
        assert_allclose(out.arrays["environment_forces"],
                        cluster.arrays["lammps_forces"] - out.arrays["isolated_forces"], atol=0)
        for name in ("lammps_forces", "global_index", "pair_role", "local_index"):    # exported data kept
            assert_array_equal(out.arrays[name], cluster.arrays[name])
        self.assertNotIn("molecule_isolated_forces", out.arrays)
        self.assertEqual(out.info["recompute_fragments"], "pair")

    def test_molecule_mode_adds_members_and_interaction_energy(self):
        cluster = self.clusters[0]
        fragments = Fragments.from_structure(cluster)
        out = ForceDecomposition(self.evaluator, molecules=True).apply(cluster).structure
        energies = {}
        for label, rows in fragments.members():
            energies[label], forces = emt(cluster, rows)
            assert_allclose(out.arrays["molecule_isolated_forces"][rows], forces, atol=1e-12)
            self.assertAlmostEqual(out.info[f"energy_{label}"], energies[label], places=12)
        self.assertAlmostEqual(out.info["interaction_energy"],
                               out.info["energy_pair"] - energies["CPD"] - energies["MAL"], places=12)
        shell = np.concatenate(fragments.shell)
        assert_array_equal(out.arrays["molecule_isolated_forces"][shell], out.arrays["isolated_forces"][shell])
        assert_allclose(out.arrays["molecule_environment_forces"],
                        cluster.arrays["lammps_forces"] - out.arrays["molecule_isolated_forces"], atol=0)
        # The pair-level and molecule-level forces differ exactly by the partner's influence.
        self.assertGreater(np.abs(out.arrays["molecule_isolated_forces"][fragments.pair]
                                  - out.arrays["isolated_forces"][fragments.pair]).max(), 0)

    def test_file_round_trip_keeps_full_precision_and_never_overwrites(self):
        output = self.root / "decomposed.extxyz"
        summary = recompute_file(self.clusters_file, output, ForceDecomposition(self.evaluator, molecules=True))
        self.assertEqual(summary["structures"], len(self.clusters))
        back = read(output, index=":")
        direct = [ForceDecomposition(self.evaluator, molecules=True).apply(c).structure for c in self.clusters]
        for written, expected in zip(back, direct):
            for name in ("environment_forces", "isolated_forces", "molecule_environment_forces", "lammps_forces"):
                assert_array_equal(written.arrays[name], expected.arrays[name])
            self.assertEqual(written.info["interaction_energy"], expected.info["interaction_energy"])
        with self.assertRaises(FileExistsError):
            recompute_file(self.clusters_file, output, ForceDecomposition(self.evaluator))
        with self.assertRaisesRegex(ValueError, "differ from the input"):
            recompute_file(self.clusters_file, self.clusters_file, ForceDecomposition(self.evaluator))

    def test_failed_task_leaves_no_partial_output(self):
        class Failing:
            calls = 0

            def apply(self, structure):
                Failing.calls += 1
                if Failing.calls == 2:
                    raise ValueError("boom")
                return ForceDecomposition(RecomputeTests.evaluator).apply(structure)
        output = self.root / "failed.extxyz"
        with self.assertRaisesRegex(ValueError, "boom"):
            recompute_file(self.clusters_file, output, Failing())
        self.assertEqual(list(self.root.glob("failed*")), [])

    def test_fragment_relaxation(self):
        relaxer = FragmentRelaxer(self.evaluator, optimizer="BFGS", fmax=0.05, steps=15)
        output = self.root / "relaxed.extxyz"
        summary = recompute_file(self.pairs_file, output, FragmentRelaxation(relaxer, molecules=True))
        self.assertEqual(summary["extra_structures"], 3 * len(self.pairs))          # pair, CPD, MAL each
        annotated, relaxed = read(output, index=":"), read(summary["extras"], index=":")
        for structure in annotated:
            for label in ("pair", "CPD", "MAL"):
                self.assertAlmostEqual(structure.info[f"deformation_energy_{label}"],
                                       structure.info[f"energy_{label}"] - structure.info[f"relaxed_energy_{label}"])
                self.assertLessEqual(structure.info[f"relaxed_energy_{label}"], structure.info[f"energy_{label}"] + 1e-9)
        first, pair_fragment = self.pairs[0], relaxed[0]
        self.assertEqual(pair_fragment.info["fragment"], "pair")
        assert_array_equal(pair_fragment.arrays["global_index"], first.arrays["global_index"])
        assert_array_equal(pair_fragment.arrays["md_positions"], first.positions)
        energy, _ = emt(first, np.arange(len(first)))
        self.assertAlmostEqual(annotated[0].info["energy_pair"], energy, places=10)

    def test_frozen_shell_optimization(self):
        cluster = self.clusters[0]
        fragments = Fragments.from_structure(cluster)
        relaxer = FragmentRelaxer(self.evaluator, optimizer="FIRE", fmax=0.05, steps=30)
        out = FrozenShellOptimization(relaxer).apply(cluster).structure
        shell = np.concatenate(fragments.shell)
        assert_array_equal(out.positions[shell], cluster.positions[shell])        # shell exactly fixed
        self.assertGreater(np.abs(out.positions[fragments.pair] - cluster.positions[fragments.pair]).max(), 0)
        assert_array_equal(out.arrays["md_positions"], cluster.positions)
        assert_array_equal(np.flatnonzero(out.arrays["frozen"]), np.sort(shell))
        self.assertLess(out.info["energy_optimized"], out.info["energy_md_geometry"])
        energy, forces = emt(out, np.arange(len(out)))
        self.assertAlmostEqual(out.info["energy_optimized"], energy, places=10)
        assert_allclose(out.arrays["optimized_forces"], forces, atol=1e-10)
        with self.assertRaisesRegex(ValueError, "needs a cluster"):
            FrozenShellOptimization(relaxer).apply(self.pairs[0])

    def test_writer_logical_and_float_columns(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 1e-9]])
        atoms.new_array("frozen", np.array([True, False]))
        atoms.new_array("tiny", np.array([1.23456789e-12, -2.5]))
        path = self.root / "writer.extxyz"
        write_extxyz([atoms], path)
        back = read(path)
        assert_array_equal(back.arrays["frozen"], [True, False])
        assert_array_equal(back.arrays["tiny"], atoms.arrays["tiny"])
        assert_array_equal(back.positions, atoms.positions)

    def test_cli_recompute_uses_the_uma_calculator_factory(self):
        target = self.root / "cli_single.extxyz"
        with patch("MartiniFuncs.workflows.force_validation.local_uma_calculator", return_value=EMT()) as factory, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            status = main(["recompute", str(self.clusters_file), "--output", str(target), "--checkpoint", "uma.pt",
                           "--molecules", "--device", "cpu", "--quiet"])
        self.assertEqual(status, 0, errors.getvalue())
        self.assertEqual(factory.call_args.kwargs["task"], "omol")
        self.assertEqual(factory.call_args.kwargs["device"], "cpu")
        self.assertEqual(factory.call_args.kwargs["inference_settings"], {"merge_mole": False, "compile": False})
        written = read(target, index=0)
        self.assertIn("molecule_environment_forces", written.arrays)
        self.assertIn("interaction_energy", written.info)
        with patch("MartiniFuncs.workflows.force_validation.local_uma_calculator", return_value=EMT()), \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            status = main(["recompute", str(self.clusters_file), "--output", str(self.root / "x.extxyz"),
                           "--checkpoint", "uma.pt", "--task", "frozen-shell", "--molecules", "--quiet"])
        self.assertEqual(status, 2)
        self.assertIn("--molecules applies to", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
