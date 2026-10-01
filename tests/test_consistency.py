"""Regression checks for issues identified in the package consistency review."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from ase import Atoms

from MartiniFuncs.io import psience
from MartiniFuncs.io import orca
from MartiniFuncs.statistics import md
from MartiniFuncs.trajectory.analysis import analyze_run, molecular_window_statistics


class ConsistencyTests(unittest.TestCase):
    def test_fragment_intersection_includes_atom_zero(self):
        molecule = SimpleNamespace(fragment_indices=[[0], [1, 2]])
        self.assertTrue(psience.check_if_internal_part_of_multiple_fragments([0, 1], molecule))
        self.assertFalse(psience.check_if_internal_part_of_multiple_fragments([1, 2], molecule))

    def test_mass_lookup_does_not_return_dummy_element(self):
        self.assertEqual(psience.get_symbol_from_mass(1.0), "H")
        for mass in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                psience.get_symbol_from_mass(mass)

    def test_displacement_accepts_one_shot_id_iterables(self):
        atoms = Atoms("HH", positions=[[0, 0, 0], [.7, 0, 0]])
        atoms.new_array("global_index", np.array([10, 20]))
        moved = atoms.copy()
        moved.positions += [2, 3, 4]
        assert_allclose(md.get_molecule_displacement(atoms, moved, iter([10, 20])), [2, 3, 4])

    def test_orca_constraint_failure_is_not_overwritten(self):
        # Test validation bookkeeping independently of Psience geometry/evaluators.
        structures = [SimpleNamespace(coords=np.array([q])) for q in (1., 1.2, 1.)]
        with patch.object(orca, "_get_specs_from_inp", return_value=[(0, 1)]), \
             patch.object(orca, "read", return_value=structures), \
             patch.object(orca, "ase_atoms_to_molecule", side_effect=lambda a: a), \
             patch.object(orca.nput, "internal_coordinate_tensors", side_effect=lambda coords, *a, **kw: [coords]), \
             redirect_stdout(StringIO()):
            self.assertTrue(orca._validate_constraint_in_optimization("unused", "unused", endpoints_only=True))
            self.assertFalse(orca._validate_constraint_in_optimization("unused", "unused", endpoints_only=False))
            structures[-1].coords[0] = 1.1
            self.assertFalse(orca._validate_constraint_in_optimization("unused", "unused", endpoints_only=True))

    def test_downsampling_does_not_hide_timestep_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.lammpstrj"
            source = Path(__file__).parent / "lammpsio_tests/multiple_frames.lammpstrj"
            text = source.read_text().replace("ITEM: TIMESTEP\n1000\n", "ITEM: TIMESTEP\n0\n", 1)
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "Timesteps must increase"):
                analyze_run(directory, units="metal", timestep_ps=.0005, frame_stride=2, max_frames=5)

    def test_uniform_time_rule_matches_direct_statistics(self):
        times = np.array([0., 1., 2.000005])
        positions = np.zeros((3, 1, 3))
        with self.assertRaisesRegex(ValueError, "uniformly"):
            md.get_molecular_msd(positions, times, unwrapped=True, selection="H2")
        result = {"frames": [{"frame": i, "time_ps": t} for i, t in enumerate(times)], "species": np.array(["H2"]),
                  "com_angstrom": positions, "com_velocity_angstrom_ps": None, "net_force_eV_angstrom": None}
        with self.assertRaisesRegex(ValueError, "uniformly"):
            molecular_window_statistics(result, )


if __name__ == "__main__":
    unittest.main()
