"""Package-wide conventions: one type_symbols form and one bond-scale default."""
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MartiniFuncs.structure import graph
from MartiniFuncs.chemistry.cpd_n4t import selection
from MartiniFuncs.trajectory import reactions
from MartiniFuncs.trajectory import analysis
from MartiniFuncs.chemistry.cpd_n4t import pairs as cpd_pairs
from MartiniFuncs.legacy import geom_utils as legacy_geom
from MartiniFuncs.pairs import structures
from MartiniFuncs.structure.packing import write_initial_data
from MartiniFuncs.io.lammps_inputs import render_lammps_input
from MartiniFuncs.io.lammps import normalize_type_symbols, read_lammps_frame, type_symbol_list


class TypeSymbolTests(unittest.TestCase):
    def test_list_and_mapping_forms_are_equivalent(self):
        self.assertEqual(normalize_type_symbols(["C", "H"]), {1: "C", 2: "H"})
        self.assertEqual(normalize_type_symbols({"2": "H", "1": "C"}), {1: "C", 2: "H"})
        self.assertEqual(type_symbol_list({2: "H", 1: "C"}), ["C", "H"])
        self.assertIsNone(normalize_type_symbols(None))
        with self.assertRaises(TypeError):
            normalize_type_symbols("CH")
        with self.assertRaisesRegex(ValueError, "1..n"):
            type_symbol_list({1: "C", 3: "H"})

    def test_reader_accepts_a_type_order_list(self):
        # A list used to be indexed as type_symbols[type], silently shifting elements by one.
        text = ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: BOX BOUNDS pp pp pp\n0 9\n0 9\n0 9\n"
                "ITEM: ATOMS id type x y z\n1 1 1 1 1\n2 2 2 2 2\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.lammpstrj"
            path.write_text(text)
            from_list = read_lammps_frame(path, units="metal", type_symbols=["C", "H"]).atoms
            from_dict = read_lammps_frame(path, units="metal", type_symbols={1: "C", 2: "H"}).atoms
        self.assertEqual(from_list.get_chemical_symbols(), ["C", "H"])
        self.assertEqual(from_dict.get_chemical_symbols(), ["C", "H"])

    def test_writers_accept_a_mapping(self):
        options = dict(data_file="initial.data", ensemble="nvt", steps=100, force_source="external_uma",
                       temperature_K=400)
        listed = render_lammps_input(type_symbols=["C", "H"], **options)
        self.assertIn("dump_modify trajectory element C H sort id", listed)
        self.assertEqual(render_lammps_input(type_symbols={1: "C", 2: "H"}, **options), listed)
        from ase.build import molecule
        from MartiniFuncs.structure.packing import MoleculeSpec, pack_molecules
        atoms = pack_molecules([MoleculeSpec("water", molecule("H2O"), 5)], [12]*3, seed=4)
        with tempfile.TemporaryDirectory() as tmp:
            a = write_initial_data(atoms, Path(tmp) / "a.data", type_symbols=["O", "H"])
            b = write_initial_data(atoms, Path(tmp) / "b.data", type_symbols={1: "O", 2: "H"})
            self.assertEqual((Path(tmp) / "a.data").read_text().splitlines()[1:],
                             (Path(tmp) / "b.data").read_text().splitlines()[1:])
        self.assertEqual(a["type_symbols"], b["type_symbols"])


class BondScaleDefaultTests(unittest.TestCase):
    def test_every_public_default_is_one_point_three(self):
        functions = [graph.build_bond_graph, graph.repair_fragmented_molecules,
                     legacy_geom.get_diels_alder_reaction_indices_from_pair, cpd_pairs.get_all_prereactive_pairs,
                     structures.get_pair_shell_clusters, structures.get_pair_positions_and_forces,
                     legacy_geom.get_pairs_distance_distribution, cpd_pairs.get_pair_geometry,
                     selection.measure_pair_geometry, selection.assess_prereactive_pairs,
                     reactions.find_possible_products, reactions.scan_possible_products,
                     analysis.analyze_run]
        for function in functions:
            parameters = inspect.signature(function).parameters
            defaults = {name: p.default for name, p in parameters.items() if name in ("scale", "bond_scale")}
            self.assertTrue(defaults, function.__name__)
            self.assertEqual(set(defaults.values()), {1.3}, function.__name__)


if __name__ == "__main__":
    unittest.main()
