"""Export memo caches never change results and miss when molecular bonds change."""
import unittest
from pathlib import Path

import numpy as np

from MartiniFuncs import graph, pair_selection
from MartiniFuncs.geom_utils import get_all_prereactive_pairs, get_pair_shell_clusters
from MartiniFuncs.lammpsio import read_lammps_frame

FIXTURES = Path(__file__).parent / "lammpsio_tests"


class ExportCacheTests(unittest.TestCase):
    def setUp(self):
        self.atoms = read_lammps_frame(FIXTURES / "multiple_frames.lammpstrj", 0, units="metal").atoms
        self.pairs, self.shifts = get_all_prereactive_pairs(self.atoms, scale=1.3)

    def test_caches_do_not_change_geometry_or_shells(self):
        plain_geometry = pair_selection.assess_prereactive_pairs(self.atoms, self.pairs, self.shifts)
        plain_shells = get_pair_shell_clusters(self.atoms, self.pairs, self.shifts, shell_radius=3.0)
        with pair_selection.site_cache(), graph.frame_cache():
            cached_geometry = pair_selection.assess_prereactive_pairs(self.atoms, self.pairs, self.shifts)
            cached_shells = get_pair_shell_clusters(self.atoms, self.pairs, self.shifts, shell_radius=3.0)
        self.assertEqual(cached_geometry, plain_geometry)
        for a, b in zip(plain_shells, cached_shells):
            np.testing.assert_array_equal(a["cluster_ids"], b["cluster_ids"])
            np.testing.assert_array_equal(a["cluster_xyz"], b["cluster_xyz"])
        self.assertIsNone(graph._UNWRAP_MEMO)
        self.assertIsNone(pair_selection._SITE_MEMO)

    def test_site_cache_keys_on_intramolecular_bonds(self):
        bonded = graph.build_bond_graph(self.atoms, scale=1.3)
        elements = dict(zip(graph.atom_ids(self.atoms), self.atoms.get_chemical_symbols()))
        cpd = self.pairs[0][0]
        with pair_selection.site_cache():
            sites = pair_selection._molecule_sites(pair_selection._cpd_sites, bonded, elements, cpd)
            broken = {i: [(j, s) for j, s in v if not {i, j} == {cpd[0], cpd[1]}] for i, v in bonded.items()}
            broken_bond = any(cpd[1] == j for j, _ in bonded[cpd[0]])
            self.assertEqual(len(pair_selection._SITE_MEMO), 1)
            try:
                pair_selection._molecule_sites(pair_selection._cpd_sites, broken, elements, cpd)
            except ValueError:
                pass
            self.assertEqual(len(pair_selection._SITE_MEMO), 2 if broken_bond else 1)
            self.assertEqual(pair_selection._molecule_sites(pair_selection._cpd_sites, bonded, elements, cpd), sites)


if __name__ == "__main__":
    unittest.main()
