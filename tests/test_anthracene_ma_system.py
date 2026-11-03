"""Anthracene + maleic anhydride: the generic (pair-model-free) pipeline on a second Diels-Alder system.

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


def write_dump(path, frames):
    """A LAMMPS text dump (metal units, types C=1 H=2 O=3) of finite structures centered in a cubic box."""
    types = {"C": 1, "H": 2, "O": 3}
    lines = []
    for step, atoms in enumerate(frames):
        xyz = atoms.positions - atoms.positions.mean(axis=0) + BOX / 2
        lines += ["ITEM: TIMESTEP", str(1000 * step), "ITEM: NUMBER OF ATOMS", str(len(atoms)),
                  "ITEM: BOX BOUNDS pp pp pp", *[f"0 {BOX}"] * 3, "ITEM: ATOMS id type element x y z"]
        lines += [f"{i + 1} {types[s]} {s} {x:.8f} {y:.8f} {z:.8f}"
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


if __name__ == "__main__":
    unittest.main()
