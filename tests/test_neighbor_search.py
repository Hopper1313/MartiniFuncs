"""KD-tree periodic neighbor search used by the bond graph, shell index and reaction screen."""
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from MartiniFuncs import graph
from MartiniFuncs.lammpsio import read_lammps_frame
from MartiniFuncs.reaction_detection import find_possible_products

FIXTURES = Path(__file__).parent / "lammpsio_tests"


class NeighborSearchTests(unittest.TestCase):
    def test_pairs_match_ase_for_random_cells(self):
        rng = np.random.default_rng(11)
        for trial in range(120):
            n = int(rng.integers(1, 40))
            cell = np.diag(rng.uniform(2, 12, 3)) + np.tril(rng.uniform(-3, 3, (3, 3)), -1) * rng.integers(0, 2)
            pbc = [True] * 3 if trial % 4 else [bool(x) for x in rng.integers(0, 2, 3)]
            positions = rng.uniform(-15, 15, (n, 3))
            cutoff = float(rng.uniform(.5, 9))   # up to several cell widths
            atoms = Atoms(f"H{n}", positions=positions, cell=cell, pbc=pbc)
            i, j, d, S = neighbor_list("ijdS", atoms, cutoff, self_interaction=False)
            expected = sorted(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))
            a, b, s, distances = graph.periodic_pairs(positions, cell, pbc, cutoff)
            self.assertEqual(list(zip(a.tolist(), b.tolist(), map(tuple, s.tolist()))), expected)
            np.testing.assert_allclose(distances, np.linalg.norm(positions[b] + s @ cell - positions[a], axis=1))

    def test_bond_graph_is_order_independent_and_symmetric(self):
        atoms = read_lammps_frame(FIXTURES / "sheared_frames.lammpstrj", 0, units="metal").atoms
        bonded = graph.build_bond_graph(atoms, scale=1.3)
        for i, neighbors in bonded.items():
            keys = [(j, tuple(s)) for j, s in neighbors]
            self.assertEqual(keys, sorted(keys))
            for j, s in neighbors:
                self.assertIn((i, tuple(-s)), [(k, tuple(t)) for k, t in bonded[j]])
        reordered = atoms[np.random.default_rng(1).permutation(len(atoms))]
        again = graph.build_bond_graph(reordered, scale=1.3)
        self.assertEqual({i: [(j, tuple(s)) for j, s in v] for i, v in bonded.items()},
                         {i: [(j, tuple(s)) for j, s in v] for i, v in again.items()})

    def test_reaction_screen_matches_graph_components(self):
        atoms = read_lammps_frame(FIXTURES / "multiple_frames.lammpstrj", 0, units="metal").atoms
        merged = atoms.copy()
        bonded = graph.build_bond_graph(merged, scale=1.3)
        components = graph.connected_components(bonded, atom_ids=graph.atom_ids(merged))
        cpd = next(c for c in components if len(c) == 11)
        mal = next(c for c in components if len(c) == 23)
        _, whole_cpd = graph.unwrap_component(merged, bonded, cpd)
        _, whole_mal = graph.unwrap_component(merged, bonded, mal)
        carbon = next(i for i in cpd if merged.numbers[i] == 6)
        anchor = whole_mal[next(k for k, i in enumerate(mal) if merged.numbers[i] == 6)]
        for offset in np.vstack([np.eye(3), -np.eye(3)]) * 1.6:
            # Place one CPD carbon 1.6 A from a dienophile carbon; keep the first
            # placement whose graph contains exactly one 34-atom component.
            merged.positions[cpd] = whole_cpd + (anchor + offset - whole_cpd[cpd.index(carbon)])
            reference = [sorted(c) for c in graph.connected_components(
                graph.build_bond_graph(merged, scale=1.3), atom_ids=graph.atom_ids(merged)) if len(c) == 34]
            if len(reference) == 1:
                break
        self.assertEqual(reference, [sorted(cpd + mal)])
        hits = find_possible_products(merged, reactant_atom_counts=(11, 23))
        self.assertEqual([h["atom_ids"] for h in hits], reference)
        self.assertEqual(hits[0]["formula"], "C16H15NO2")
        self.assertEqual(find_possible_products(atoms, reactant_atom_counts=(11, 23)), [])

    def test_index_map_cache_tracks_in_place_changes(self):
        atoms = Atoms("H3", positions=np.zeros((3, 3)))
        atoms.new_array("global_index", np.array([5, 6, 7]))
        self.assertEqual(graph.get_index_maps(atoms)[1], {5: 0, 6: 1, 7: 2})
        atoms.arrays["global_index"][:] = [7, 6, 5]
        self.assertEqual(graph.get_index_maps(atoms)[1], {7: 0, 6: 1, 5: 2})


if __name__ == "__main__":
    unittest.main()
