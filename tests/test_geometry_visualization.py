import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from ase import Atoms

from MartiniFuncs.structure.alignment import align_geometry
from MartiniFuncs.chemistry.cpd_n4t.legacy import align_atoms
from MartiniFuncs.chemistry.cpd_n4t.visualization import (force_arrow_displacements, to_display_molecule,
                                                          plot_cluster)


class GeometryVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.atoms = Atoms("CHNO", positions=[[0, 0, 0], [1, 0, 0],
                                            [0, 2, 0], [0, 0, 3]])
        self.atoms.new_array("global_index", np.array([50, 2, 900, 17]))
        self.atoms.new_array("lammps_forces", np.arange(12).reshape(4, 3) / 10)
        self.atoms.new_array("velocities", np.arange(12).reshape(4, 3) / 100)
        self.atoms.set_velocities(self.atoms.arrays["velocities"])
        self.atoms.info["force_units"] = "eV/Angstrom"

    def test_rotation_translation_and_reordered_reference(self):
        a = self.atoms
        r = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        reference = a.copy()
        reference.positions = a.positions @ r + [3, -2, 7]
        aligned, rmsd, transform = align_geometry(a, reference[::-1])
        assert_allclose(aligned.positions, reference.positions, atol=1e-12)
        assert_allclose(transform.rotation, r, atol=1e-12)
        for name in ("lammps_forces", "velocities", "momenta"):
            assert_allclose(aligned.arrays[name], a.arrays[name] @ r, atol=1e-12)
        assert_allclose(aligned.get_velocities(), a.get_velocities() @ r, atol=1e-12)
        assert_array_equal(aligned.arrays["global_index"], a.arrays["global_index"])
        assert_allclose(a.positions[0], [0, 0, 0])
        self.assertLess(rmsd, 1e-12)
        self.assertIsNone(aligned.calc)
        # Scalar work is invariant when displacement and force rotate together.
        delta = np.array([.2, .3, -.1])
        assert_allclose(aligned.arrays["lammps_forces"] @ transform.vectors(delta),
                        a.arrays["lammps_forces"] @ delta)

    def test_legacy_return_and_subset_fit(self):
        reference = self.atoms.copy()
        reference.positions += 3
        aligned, rmsd = align_atoms(self.atoms, reference, indices=[0, 1, 2])
        assert_allclose(aligned.positions, reference.positions, atol=1e-12)
        self.assertLess(rmsd, 1e-12)

    def test_invalid_correspondence_and_periodic_input(self):
        reference = self.atoms.copy()
        reference.arrays["global_index"][0] = -1
        with self.assertRaisesRegex(ValueError, "missing atom ID"):
            align_geometry(self.atoms, reference)
        self.atoms.pbc = True
        with self.assertRaisesRegex(ValueError, "finite"):
            align_geometry(self.atoms, reference)

    def test_force_display_scale_units_and_zero_vectors(self):
        assert_allclose(force_arrow_displacements(self.atoms, scale=.5),
                        self.atoms.arrays["lammps_forces"] * .5)
        self.atoms.arrays["lammps_forces"][:] = 0
        assert_allclose(force_arrow_displacements(self.atoms), 0)
        self.atoms.info["force_units"] = "Hartree/Bohr"
        with self.assertRaisesRegex(ValueError, "eV/Angstrom"):
            force_arrow_displacements(self.atoms)

    def test_display_conversion_does_not_attach_calculator(self):
        from McUtils.Data import UnitsData
        molecule = to_display_molecule(self.atoms)
        assert_allclose(molecule.coords, self.atoms.positions *
                        UnitsData.convert("Angstroms", "BohrRadius"))
        self.assertIsNone(self.atoms.calc)
        self.assertIsNone(molecule.energy_evaluator)
        # Conversion must work without evaluating an energy or loading weights.
        self.assertEqual(len(molecule.atoms), len(self.atoms))

    def test_one_angstrom_and_reverse_conversion(self):
        from MartiniFuncs.io.psience import ase_atoms_to_molecule, molecule_to_ase_atoms
        atoms = Atoms("H2", positions=[[0., 0., 0.], [1., 0., 0.]])
        for converter in (to_display_molecule, ase_atoms_to_molecule):
            molecule = converter(atoms)
            # Independent numerical reference catches a missing/double conversion.
            self.assertAlmostEqual(float(np.linalg.norm(molecule.coords[1] - molecule.coords[0])),
                                   1.889726125, places=6)
            recovered = molecule_to_ase_atoms(molecule)
            assert_allclose(recovered.positions, atoms.positions, atol=1e-12)

    def test_psience_receives_bohr_arrow_displacements_not_raw_forces(self):
        from Psience.Molecools import Molecule
        atoms = Atoms("H2", positions=[[0., 0., 0.], [1., 0., 0.]])
        forces = np.array([[1., 0., 0.], [0., 0., 0.]])
        atoms.new_array("lammps_forces", forces.copy())
        with patch.object(Molecule, "plot", autospec=True) as plot:
            plot_cluster(atoms, forces=True, force_scale=0.2)
        molecule = plot.call_args.args[0]
        arrows = plot.call_args.kwargs["mode_vectors"].reshape(-1, 3)
        self.assertAlmostEqual(float(arrows[0, 0]), 0.2 * 1.889726125, places=6)
        self.assertAlmostEqual(float(arrows[0, 0] / molecule.coords[1, 0]), 0.2)
        assert_array_equal(arrows[1], [0., 0., 0.])
        assert_array_equal(atoms.arrays["lammps_forces"], forces)
        assert_array_equal(atoms.positions[1], [1., 0., 0.])

    def test_plot_preserves_psience_bond_defaults(self):
        from Psience.Molecools import Molecule
        with patch.object(Molecule, "plot", autospec=True) as plot:
            plot_cluster(self.atoms)
            self.assertNotIn("bonds", plot.call_args.kwargs)
            self.assertNotIn("reconcile_bonds", plot.call_args.kwargs)



if __name__ == "__main__":
    unittest.main()
