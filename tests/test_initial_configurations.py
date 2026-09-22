from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms
from ase.build import molecule
from ase.io.lammpsdata import read_lammps_data

from MartiniFuncs.initial_configurations import MoleculeSpec, PackingError, pack_molecules, write_initial_data
from MartiniFuncs.lammps_inputs import render_lammps_input


class InitialConfigurationTests(unittest.TestCase):
    def test_80_molecule_composition_geometry_and_periodic_separation(self):
        specs = [MoleculeSpec("water", molecule("H2O"), 40), MoleculeSpec("methane", molecule("CH4"), 40)]
        atoms = pack_molecules(specs, [30]*3, seed=123, min_distance=2)
        other = pack_molecules(specs, [30]*3, seed=123, min_distance=2)
        assert_allclose(atoms.positions, other.positions)
        self.assertEqual(len(atoms), 320)
        self.assertEqual(len(np.unique(atoms.arrays["molecule_id"])), 80)
        whole = atoms.positions + atoms.arrays["image_flags"]*30
        for mid in np.unique(atoms.arrays["molecule_id"]):
            mask = atoms.arrays["molecule_id"] == mid
            name = atoms.arrays["molecule_name"][mask][0]
            template = next(s.atoms for s in specs if s.name == name)
            delta = whole[mask][:,None]-whole[mask][None,:]
            assert_allclose(np.linalg.norm(delta,axis=-1), template.get_all_distances(), atol=1e-12)
        # Independent ASE minimum-image distance check, excluding same-molecule pairs.
        distances = atoms.get_all_distances(mic=True)
        intermolecular = atoms.arrays["molecule_id"][:,None] != atoms.arrays["molecule_id"][None,:]
        self.assertGreaterEqual(distances[intermolecular].min(), 2-1e-12)
        self.assertIsNone(atoms.calc)
        self.assertNotIn("momenta", atoms.arrays)

    def test_three_species_and_bounded_failure(self):
        specs = [MoleculeSpec(s, Atoms(s), n) for s,n in [("He",2),("Ne",3),("Ar",4)]]
        atoms = pack_molecules(specs, [15]*3, seed=11)
        self.assertEqual(atoms.info["composition"], {"He":2,"Ne":3,"Ar":4})
        with self.assertRaises(PackingError):
            pack_molecules([MoleculeSpec("He",Atoms("He"),2)], [2.1]*3, seed=1,
                           min_distance=2, max_attempts_per_molecule=3)
        with self.assertRaises(ValueError):
            pack_molecules(specs, [[10,0,0],[1,10,0],[0,0,10]], seed=1)

    def test_atomic_data_roundtrip_preserves_types_mass_images_and_identity(self):
        atoms = pack_molecules([MoleculeSpec("water",molecule("H2O"),40)], [20]*3, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"initial.data"
            metadata = write_initial_data(atoms,path,type_symbols=["O","H"])
            restored = read_lammps_data(path, atom_style="atomic", units="metal", Z_of_type={1:8,2:1})
            assert_allclose(restored.positions, atoms.positions + atoms.arrays["image_flags"]*20, atol=1e-10)
            assert_allclose(restored.get_masses(),atoms.get_masses())
            self.assertEqual(restored.get_chemical_symbols(),atoms.get_chemical_symbols())
            self.assertEqual(json.loads(Path(str(path)+".json").read_text()),metadata)
            self.assertEqual([a["molecule_id"] for a in metadata["atoms"]], atoms.arrays["molecule_id"].tolist())
            with self.assertRaises(FileExistsError):
                write_initial_data(atoms,path)

    def test_run_templates_and_parameter_checks(self):
        for ensemble in ("npt","nvt","nve"):
            text = render_lammps_input(data_file="initial.data",type_symbols=["C","H"],ensemble=ensemble,
                        steps=100,force_source="external_uma",temperature_K=400,
                        pressure_bar=25000 if ensemble=="npt" else None)
            self.assertTrue(text.splitlines()[-1].startswith("run 100 "))
            self.assertIn(f"fix integrate all {ensemble}",text)
            self.assertIn("dump_modify trajectory element C H sort id",text)
            self.assertNotIn("pair_style",text)
            if ensemble == "nve":
                self.assertNotIn(" npt ",text)
                self.assertNotIn(" nvt ",text)
        with self.assertRaises(ValueError):
            render_lammps_input(data_file="initial.data",type_symbols=["C"],ensemble="npt",
                                steps=100,force_source="external_uma",temperature_K=400)


if __name__ == "__main__":
    unittest.main()
