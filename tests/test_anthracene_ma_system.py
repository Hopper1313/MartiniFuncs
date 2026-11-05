"""Anthracene + maleic anhydride: the generic pipeline and the anthracene_ma pair model on a second Diels-Alder system.

Fixtures (tests/anthracene_ma_tests) are the UMA-s-1p2p1 structures of the
anthracene + maleic anhydride hand-off: isolated reactants, reactant complex,
TS, IRC endpoints and adduct, all 33-atom files in one atom order.
"""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from ase.io import read

from MartiniFuncs.chemistry.anthracene_ma import AnthraceneMaCollection, AnthraceneMaCriteria, AnthraceneMaModel
from MartiniFuncs.chemistry.anthracene_ma.build import ROLE_LABELS, structure_metrics
from MartiniFuncs.chemistry.diels_alder import SHARED_METRICS
from MartiniFuncs.pairs.assessment import screen_frame
from MartiniFuncs.pairs.model import default_reactive_radius, get_pair_model, pair_model_names
from MartiniFuncs.structure import graph
from MartiniFuncs.structure.molecules import atoms_from_smiles
from MartiniFuncs.structure.packing import MoleculeSpec, pack_molecules
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.trajectory.reactions import PossibleReactionError
from MartiniFuncs.workflows.cli import main
from MartiniFuncs.workflows.postprocess import load_analysis_cache

FIXTURES = Path(__file__).parent / "anthracene_ma_tests"
BOX = 26.0   # Angstrom; the 33-atom complex fits with room to spare


def formulas(atoms):
    """Molecular formulas of a structure's bonded components (default bond rule)."""
    atoms = atoms.copy()
    atoms.set_array("global_index", np.arange(len(atoms)))
    components = graph.connected_components(graph.build_bond_graph(atoms, scale=1.3), atom_ids=graph.atom_ids(atoms))
    return sorted(atoms[[int(i) for i in c]].get_chemical_formula() for c in components)


def write_dump(path, frames, *, forces=False):
    """A LAMMPS text dump (metal units, types C=1 H=2 O=3) of finite structures centered in a cubic box.

    forces adds zero fx fy fz columns (pair structure exports require force columns).
    """
    types = {"C": 1, "H": 2, "O": 3}
    lines = []
    for step, atoms in enumerate(frames):
        xyz = atoms.positions - atoms.positions.mean(axis=0) + BOX / 2
        lines += ["ITEM: TIMESTEP", str(1000 * step), "ITEM: NUMBER OF ATOMS", str(len(atoms)),
                  "ITEM: BOX BOUNDS pp pp pp", *[f"0 {BOX}"] * 3,
                  "ITEM: ATOMS id type element x y z" + (" fx fy fz" if forces else "")]
        lines += [f"{i + 1} {types[s]} {s} {x:.8f} {y:.8f} {z:.8f}" + (" 0 0 0" if forces else "")
                  for i, (s, (x, y, z)) in enumerate(zip(atoms.get_chemical_symbols(), xyz))]
    Path(path).write_text("\n".join(lines) + "\n")


class AnthraceneMaleicAnhydrideTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_bond_rule_separates_reactants_ts_and_adduct(self):
        """The default bond rule (scale 1.3) keeps the TS as two molecules and the adduct as one."""
        reactants = ["C14H10", "C4H2O3"]
        for name in ("reactant_complex", "irc_reactant_endpoint", "transition_state"):
            self.assertEqual(formulas(read(FIXTURES / f"{name}.xyz")), reactants, name)
        for name in ("product", "irc_product_endpoint"):
            self.assertEqual(formulas(read(FIXTURES / f"{name}.xyz")), ["C18H12O3"], name)

    def make_reacting_run(self):
        """Reactant complex in frames 0-3 (with small noise), the adduct from frame 4."""
        run = self.root / "simulation"
        run.mkdir()
        complex_, adduct = read(FIXTURES / "reactant_complex.xyz"), read(FIXTURES / "product.xyz")
        rng = np.random.default_rng(3)
        frames = []
        for step in range(9):
            frame = (complex_ if step < 4 else adduct).copy()
            frame.positions += rng.normal(scale=.01, size=frame.positions.shape)
            frames.append(frame)
        write_dump(run / "trajectory.lammpstrj", frames)
        return run

    def test_reaction_event_is_recorded_at_the_adduct(self):
        run = self.make_reacting_run()
        result = analyze_run(run, units="metal", timestep_ps=.0005, on_topology_change="record",
                             topology_persistence=2)
        self.assertEqual([e["frame"] for e in result["topology_events"]], [4])
        self.assertEqual(result["topology_events"][0]["formulas"], ["C18H12O3"])
        self.assertEqual(len(result["frames"]), 9)
        truncated = analyze_run(run, units="metal", timestep_ps=.0005, on_topology_change="truncate",
                                topology_persistence=2)
        self.assertEqual([r["frame"] for r in truncated["frames"]], [0, 1, 2, 3])

    def test_legacy_product_size_screen(self):
        """--reaction-atom-counts 24 9: the first 33-atom component aborts the analysis."""
        run = self.make_reacting_run()
        with self.assertRaises(PossibleReactionError):
            analyze_run(run, units="metal", timestep_ps=.0005, reaction_atom_counts=(24, 9))

    def test_cli_molecular_statistics_without_a_pair_model(self):
        run = self.make_reacting_run()
        out = self.root / "analysis"
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            status = main([str(a) for a in ["run", run, "--output", out, "--units", "metal", "--timestep-ps", ".0005",
                                             "--no-pairs", "--reaction-persistence", "2", "--max-lag-ps", ".002",
                                             "--quiet"]])
        self.assertEqual(status, 0, errors.getvalue())
        result, _ = load_analysis_cache(out / "cache")
        self.assertEqual([e["frame"] for e in result["topology_events"]], [4])
        self.assertIsNone(result["pair_candidates"])

    def test_smiles_templates_pack_into_whole_molecules(self):
        """atoms_from_smiles + pack_molecules build an anthracene/MA box the bond graph reads back exactly."""
        anthracene, anhydride = atoms_from_smiles("c1ccc2cc3ccccc3cc2c1"), atoms_from_smiles("O=C1OC(=O)C=C1")
        self.assertEqual((anthracene.get_chemical_formula(), anhydride.get_chemical_formula()), ("C14H10", "C4H2O3"))
        packed = pack_molecules([MoleculeSpec("anthracene", anthracene, 4), MoleculeSpec("anhydride", anhydride, 4)],
                                [30.] * 3, seed=11)
        found = formulas(packed)
        self.assertEqual((found.count("C14H10"), found.count("C4H2O3"), len(found)), (4, 4, 8))


def periodic(atoms, shift):
    """atoms in a periodic cubic BOX, translated by shift and wrapped, with global_index (as a dump frame)."""
    atoms = atoms.copy()
    atoms.cell, atoms.pbc = [BOX] * 3, True
    atoms.positions += np.asarray(shift, dtype=float) - atoms.positions.mean(axis=0) + BOX / 2
    atoms.wrap()
    atoms.set_array("global_index", np.arange(len(atoms)))
    return atoms


class AnthraceneMaPairModelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_registration_and_settings(self):
        model = get_pair_model("anthracene_ma")
        self.assertIsInstance(model, AnthraceneMaModel)
        self.assertIn("anthracene_ma", pair_model_names())
        self.assertEqual(default_reactive_radius("anthracene_ma"), 4.0)
        self.assertEqual(model.role_labels, ROLE_LABELS)
        self.assertEqual(model.metric_names, SHARED_METRICS)
        self.assertEqual(model.species, frozenset({"C14H10", "C4H2O3"}))
        self.assertEqual(set(AnthraceneMaCriteria.thresholds()) - {"max_screen_distance"},
                         {"max_forming_distance", "max_distance_imbalance", "max_axis_angle",
                          "max_plane_angle", "max_approach_angle"})
        self.assertEqual(AnthraceneMaCollection(reactive_radius=3.5).reactive_radius, 3.5)
        self.assertEqual(model.classify([list(range(24)), list(range(9)), list(range(33))], {}), [0, 1, None])
        with self.assertRaisesRegex(ValueError, "expected sizes"):
            model.classify([list(range(11))], {})

    def test_screen_matches_the_builder_metrics(self):
        """Screen distance and metrics equal structure_metrics, also with the pair split across the boundary."""
        model = get_pair_model("anthracene_ma")
        for name in ("reactant_complex", "irc_reactant_endpoint", "transition_state"):
            reference = structure_metrics(read(FIXTURES / f"{name}.xyz"))
            for shift in ((0, 0, 0), (BOX / 2, BOX / 2 - 1, BOX / 2)):
                pairs, _, candidates = screen_frame(periodic(read(FIXTURES / f"{name}.xyz"), shift),
                                                    "anthracene_ma", return_candidates=True)
                self.assertEqual(len(pairs), 1, name)
                pair, _, distance, geometry = candidates[0]
                self.assertEqual((len(pair[0]), len(pair[1])), (24, 9))
                self.assertAlmostEqual(distance, reference["screen_distance"], places=6)
                metrics = model.metrics(geometry)
                for metric in SHARED_METRICS:
                    self.assertAlmostEqual(metrics[metric], reference[metric], places=6, msg=(name, metric))
        self.assertEqual(screen_frame(periodic(read(FIXTURES / "product.xyz"), (0, 0, 0)), "anthracene_ma"),
                         ([], []))       # the adduct is one molecule: no pair

    def test_cli_pair_analysis_and_trajectories(self):
        run = self.root / "simulation"
        run.mkdir()
        complex_, ts = read(FIXTURES / "reactant_complex.xyz"), read(FIXTURES / "transition_state.xyz")
        rng = np.random.default_rng(5)
        frames = [(complex_ if step < 3 else ts).copy() for step in range(5)]
        for frame in frames:
            frame.positions += rng.normal(scale=.01, size=frame.positions.shape)
        write_dump(run / "trajectory.lammpstrj", frames, forces=True)
        out = self.root / "analysis"

        def cli(*args):
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main([str(a) for a in args])
            return status, errors.getvalue()

        status, errors = cli("run", run, "--pair-model", "anthracene_ma", "--output", out, "--units", "metal",
                             "--timestep-ps", ".0005", "--max-lag-ps", ".002", "--quiet")
        self.assertEqual(status, 0, errors)
        result, _ = load_analysis_cache(out / "cache")
        candidates = result["pair_candidates"]
        self.assertEqual((str(candidates["model"]), float(candidates["reactive_radius"])), ("anthracene_ma", 4.0))
        self.assertEqual(list(candidates["metric_names"]), list(SHARED_METRICS))
        forming = candidates["metrics"][:, 1]
        self.assertEqual(len(forming), 5)
        self.assertTrue(np.all(forming[:3] > 3.1) and np.all(forming[3:] < 2.4), forming)
        status, errors = cli("trajectories", out / "cache", "--output", self.root / "traj", "--quiet",
                             "--max-forming-distance", "3.0")
        self.assertEqual(status, 0, errors)
        exported = read(self.root / "traj" / "all_pairs.extxyz", index=":")
        self.assertEqual(len(exported), 5)
        self.assertEqual(sorted(set(exported[0].arrays["pair_role"])), ["ANT", "MA"])
        self.assertEqual([int(a.arrays["accepted_binary"][0]) for a in exported], [0, 0, 0, 1, 1])
        status, errors = cli("trajectories", out / "cache", "--output", self.root / "bad", "--quiet",
                             "--max-ring-dihedral-deviation", "30")
        self.assertEqual(status, 2)
        self.assertIn("no threshold", errors)


if __name__ == "__main__":
    unittest.main()
