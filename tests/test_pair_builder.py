"""CPD + N4T pair builder: rigid CPD placed against a fixed N4T at requested geometry."""
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from MartiniFuncs.chemistry.cpd_n4t.build import build_pair, rdkit_templates, templates_from_frame
from MartiniFuncs.chemistry.cpd_n4t.selection import measure_pair_geometry
from MartiniFuncs.io.lammps import read_lammps_frame

FIXTURE = Path(__file__).parent / "lammpsio_tests/frame_0.lammpstrj"


def pairwise(xyz):
    return np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)


class PairBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cpd, cls.n4t = rdkit_templates()
        except ImportError:
            raise unittest.SkipTest("RDKit is not installed")

    def remeasure(self, atoms):
        """Independent path: rebuild the bond graph of the combined pair and measure it."""
        ids, roles = atoms.arrays["global_index"], atoms.arrays["pair_role"]
        return measure_pair_geometry(atoms, [ids[roles == "CPD"], ids[roles == "MAL"]], (0, 0, 0))

    def test_requested_geometry_is_built_and_cpd_is_rigid(self):
        for distance, dihedral, tilt, face in [(3.0, 0, 0, 1), (3.0, 180, 15, 1), (2.8, -35, -20, -1),
                                               (3.5, 90, 40, 1), (4.0, -150, 5, -1)]:
            atoms, geometry = build_pair(self.cpd, self.n4t, distance=distance, ring_dihedral=dihedral,
                                         plane_angle=tilt, face=face)
            again = self.remeasure(atoms)
            expected_phi = (dihedral + 180) % 360 - 180
            self.assertAlmostEqual(np.cos(np.radians(again.ring_dihedral - expected_phi)), 1., places=10)
            self.assertAlmostEqual(atoms.info["center_separation"], distance, places=10)
            centers = (atoms.positions[list(again.forming_bonds[0][:1]) + list(again.forming_bonds[1][:1])].mean(0),
                       atoms.positions[[again.forming_bonds[0][1], again.forming_bonds[1][1]]].mean(0))
            assert_allclose(np.linalg.norm(centers[1] - centers[0]), distance, atol=1e-9)
            assert_allclose(again.plane_angle, abs(tilt), atol=0.5)      # least-squares normals; templates ~planar
            assert_allclose(again.approach_angle, abs(tilt), atol=0.5)
            self.assertEqual(again.approach_type, "exo" if abs(expected_phi) < 90 else "endo")
            assert_array_equal(atoms.positions[len(self.cpd):], self.n4t.positions)             # N4T fixed
            assert_allclose(pairwise(atoms.positions[:len(self.cpd)]), pairwise(self.cpd.positions), atol=1e-9)
            assert_allclose(geometry.ring_dihedral, again.ring_dihedral, atol=1e-9)

    def test_tilt_sign_opens_ch2_side_away(self):
        """Positive plane_angle moves the CH2 carbon away from the N4T C=C midpoint."""
        def ch2_distance(tilt):
            atoms, geometry = build_pair(self.cpd, self.n4t, distance=3.0, ring_dihedral=180, plane_angle=tilt)
            ch2 = [k for k in range(len(self.cpd)) if atoms[k].symbol == "C"
                   and sum(1 for j in range(len(self.cpd)) if atoms[j].symbol == "H"
                           and atoms.get_distance(k, j) < 1.25) == 2][0]
            cc = [geometry.forming_bonds[0][1], geometry.forming_bonds[1][1]]
            return np.linalg.norm(atoms.positions[ch2] - atoms.positions[cc].mean(0))
        self.assertGreater(ch2_distance(20), ch2_distance(0))
        self.assertLess(ch2_distance(-20), ch2_distance(0))

    def test_distance_kinds(self):
        atoms, geometry = build_pair(self.cpd, self.n4t, distance=3.2, ring_dihedral=120, plane_angle=30,
                                     distance_kind="max_forming")
        assert_allclose(max(self.remeasure(atoms).forming_distances), 3.2, atol=1e-9)
        atoms, _ = build_pair(self.cpd, self.n4t, distance=3.25, ring_dihedral=180, plane_angle=10,
                              distance_kind="screen")
        cc = [b[1] for b in self.remeasure(atoms).forming_bonds]
        cc_com = atoms[cc].get_center_of_mass()
        assert_allclose(np.linalg.norm(atoms[:len(self.cpd)].get_center_of_mass() - cc_com), 3.25, atol=1e-9)

    def test_validation_and_contact_check(self):
        for kwargs in (dict(distance=0), dict(distance=3, plane_angle=90), dict(distance=3, face=0),
                       dict(distance=3, distance_kind="bogus")):
            with self.assertRaises(ValueError):
                build_pair(self.cpd, self.n4t, **kwargs)
        with self.assertRaisesRegex(ValueError, "min_contact"):
            build_pair(self.cpd, self.n4t, distance=1.0, min_contact=1.5)
        with self.assertRaises(ValueError):
            build_pair(self.n4t, self.cpd, distance=3.0)      # roles swapped

    def test_templates_from_trajectory_frame(self):
        cpd, n4t = templates_from_frame(read_lammps_frame(FIXTURE, units="metal").atoms)
        self.assertEqual((cpd.get_chemical_formula(), n4t.get_chemical_formula()), ("C5H6", "C11H9NO2"))
        atoms, _ = build_pair(cpd, n4t, distance=3.0, ring_dihedral=180)
        self.assertAlmostEqual(abs(self.remeasure(atoms).ring_dihedral), 180., places=8)


if __name__ == "__main__":
    unittest.main()
