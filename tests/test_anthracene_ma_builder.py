"""Anthracene + maleic anhydride sites, pair builder, placement read-back and the interactive page."""
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
from ase.io import read
from numpy.testing import assert_allclose

from MartiniFuncs.chemistry.anthracene_ma.build import (PairFrames, build_pair, measure_placement, rdkit_templates,
                                                        reference_structure, structure_metrics, uma_templates)
from MartiniFuncs.chemistry.anthracene_ma.sites import anhydride_sites, anthracene_sites
from MartiniFuncs.structure import graph

FIXTURES = Path(__file__).parent / "anthracene_ma_tests"
MA_ROWS = [0, 1, 2, 3, 4, 5, 20, 21, 32]      # maleic anhydride rows of the hand-off's 33-atom files
ANT_ROWS = [i for i in range(33) if i not in MA_ROWS]


def contact_signature(atoms, first, second):
    """Sorted intermolecular distances per element pair: equal for structures related by relabelling symmetric atoms."""
    symbols, xyz, out = atoms.get_chemical_symbols(), atoms.positions, []
    for e1 in "CH":
        for e2 in "CHO":
            a = [i for i in first if symbols[i] == e1]
            b = [j for j in second if symbols[j] == e2]
            out.append(np.sort(np.linalg.norm(xyz[a][:, None] - xyz[b][None], axis=-1).ravel()))
    return np.concatenate(out)


class SiteTests(unittest.TestCase):
    def test_sites_match_the_hand_off_mapping(self):
        """Found from bonding alone, the sites equal the hand-off manifest's reactive indices."""
        manifest = json.loads((FIXTURES / "handoff_manifest.json").read_text())["combined_system"]["reactive_indices"]
        for name in ("reactant_complex", "transition_state", "irc_reactant_endpoint"):
            atoms = read(FIXTURES / f"{name}.xyz")
            atoms.new_array("global_index", np.arange(len(atoms)))
            bonds = graph.build_bond_graph(atoms, scale=1.3)
            elements = dict(enumerate(atoms.get_chemical_symbols()))
            termini, ring = anthracene_sites(bonds, elements, ANT_ROWS)
            alkene, carbons, oxygen = anhydride_sites(bonds, elements, MA_ROWS)
            self.assertEqual(list(termini), manifest["anthracene_reactive_carbons"], name)
            self.assertEqual(list(ring), manifest["anthracene_central_ring"], name)
            self.assertEqual(list(alkene), manifest["maleic_anhydride_alkene"], name)
            self.assertEqual((carbons, oxygen), ((1, 3, 5, 20), 2), name)

    def test_wrong_molecule_is_rejected(self):
        atoms = read(FIXTURES / "maleic_anhydride.xyz")
        bonds = graph.build_bond_graph(atoms, scale=1.3, index_array=None)
        with self.assertRaisesRegex(ValueError, "C14H10"):
            anthracene_sites(bonds, dict(enumerate(atoms.get_chemical_symbols())), list(range(len(atoms))))


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = PairFrames()

    def test_stacked_pose_and_distance_kinds(self):
        atoms, geometry = build_pair(distance=3.2, frames=self.frames)
        self.assertEqual(len(atoms), 33)
        self.assertEqual(list(atoms.arrays["pair_role"][[0, 23, 24, 32]]), ["ANT", "ANT", "MA", "MA"])
        self.assertAlmostEqual(atoms.info["screen_distance"], 3.2)
        for metric in ("axis_angle", "plane_angle", "approach_angle"):
            self.assertLess(atoms.info[f"measured_{metric}"], 0.01, metric)       # stacked and parallel
        self.assertLess(atoms.info["measured_distance_imbalance"], 1e-3)
        for kind, target, read_back in (("centers", 3.4, lambda a, g: a.info["screen_distance"]),
                                        ("max_forming", 3.0, lambda a, g: max(g.forming_distances))):
            atoms, geometry = build_pair(distance=target, distance_kind=kind, offset=(0.4, -0.3), twist=20, fold=10,
                                         frames=self.frames)
            self.assertAlmostEqual(read_back(atoms, geometry), target, places=9)
        self.assertAlmostEqual(build_pair(distance=3.3, twist=90, frames=self.frames)[0].info["measured_axis_angle"],
                               90., places=3)
        self.assertAlmostEqual(build_pair(distance=3.3, fold=40, frames=self.frames)[0].info["measured_plane_angle"],
                               40., places=3)
        with self.assertRaisesRegex(ValueError, "min_contact"):
            build_pair(distance=1.5, min_contact=2.0, frames=self.frames)
        with self.assertRaises(ValueError):
            build_pair(distance=1.0, distance_kind="centers", offset=(2.0, 0.0), frames=self.frames)

    def test_structure_metrics_agree_with_the_builder(self):
        atoms, _ = build_pair(distance=3.0, distance_kind="max_forming", offset=(0.5, 0.2), twist=25, roll=10,
                              fold=-20, frames=self.frames)
        for name, value in structure_metrics(atoms).items():
            self.assertAlmostEqual(value, atoms.info.get(f"measured_{name}", atoms.info.get(name)), places=9, msg=name)

    def test_placement_round_trip(self):
        """measure_placement recovers simple placements exactly, and any placement up to molecular symmetry."""
        rng = np.random.default_rng(0)
        ant, ma = list(range(24)), list(range(24, 33))
        for _ in range(60):
            twist, roll, fold = rng.uniform(-180, 180), rng.uniform(-85, 85), rng.uniform(-175, 175)
            offset = tuple(rng.uniform(-2, 2, 2))
            atoms, _ = build_pair(distance=rng.uniform(1.0, 5.0), offset=offset, twist=twist, roll=roll, fold=fold,
                                  frames=self.frames)
            placement = measure_placement(atoms)
            again, _ = build_pair(distance=placement["height"], offset=(placement["offset_x"], placement["offset_y"]),
                                  twist=placement["twist"], roll=placement["roll"], fold=placement["fold"],
                                  frames=self.frames)
            assert_allclose(contact_signature(again, ant, ma), contact_signature(atoms, ant, ma), atol=1e-4)
            if max(abs(twist), abs(roll), abs(fold)) < 89:
                assert_allclose([placement[k] for k in ("offset_x", "offset_y", "twist", "roll", "fold")],
                                [*offset, twist, roll, fold], atol=1e-8)
                self.assertAlmostEqual(placement["height"], atoms.info["height"], places=9)

    def test_reference_structures_read_back_and_rebuild(self):
        """The UMA complex and TS as builder placements: stacked; the TS closer, with the ring folded ~15 degrees."""
        ant, ma = list(range(24)), list(range(24, 33))
        placements = {}
        for name in ("reactant_complex", "transition_state"):
            reference = reference_structure(name)
            placements[name] = p = measure_placement(reference)
            self.assertLess(max(abs(p["twist"]), abs(p["roll"])), 1.0, name)
            rebuilt, _ = build_pair(reference[ANT_ROWS], reference[MA_ROWS], distance=p["height"],
                                    offset=(p["offset_x"], p["offset_y"]), twist=p["twist"], roll=p["roll"],
                                    fold=p["fold"])
            assert_allclose(contact_signature(rebuilt, ant, ma), contact_signature(reference, ANT_ROWS, MA_ROWS),
                            atol=0.05)
        self.assertAlmostEqual(placements["reactant_complex"]["fold"], 1.5, delta=0.5)
        self.assertAlmostEqual(placements["transition_state"]["fold"], 15.2, delta=0.5)
        self.assertAlmostEqual(placements["transition_state"]["height"], 2.33, delta=0.01)
        # Hand-off landmarks: central-ring centroid to C=C midpoint.
        self.assertAlmostEqual(structure_metrics(reference_structure("reactant_complex"))["screen_distance"], 3.169, 3)
        self.assertAlmostEqual(structure_metrics(reference_structure("transition_state"))["screen_distance"], 2.335, 3)

    def test_templates(self):
        for anthracene, anhydride in (uma_templates(), rdkit_templates()):
            self.assertEqual((anthracene.get_chemical_formula(), anhydride.get_chemical_formula()), ("C14H10", "C4H2O3"))
            atoms, _ = build_pair(anthracene, anhydride, distance=3.2)
            self.assertLess(atoms.info["measured_plane_angle"], 0.1)


@unittest.skipUnless(importlib.util.find_spec("Psience"), "Psience is needed to render the page")
class ViewerTests(unittest.TestCase):
    def test_page_holds_both_scenes_data_and_presets(self):
        from MartiniFuncs.chemistry.anthracene_ma.viewer import write_builder_html
        with tempfile.TemporaryDirectory() as tmp:
            page = write_builder_html(Path(tmp) / "builder.html").read_text(encoding="utf-8")
        self.assertIn("x3dom-full.js", page)
        placement = page[page.index('<transform id="anhydride-placement"'):]
        self.assertEqual(placement[:placement.index("</transform>")].count("<Sphere"), 9)      # the MA atoms
        data = json.loads(re.search(r'id="builder-data">(.*?)</script>', page, re.S).group(1))
        self.assertEqual((len(data["anthracene"]["xyz"]), len(data["anhydride"]["xyz"])), (24, 9))
        names = [p["name"] for p in data["presets"]]
        self.assertIn("UMA TS (rigid)", names)
        self.assertEqual([r["name"] for r in data["references"]], ["UMA reactant complex", "UMA TS"])


if __name__ == "__main__":
    unittest.main()
