"""Image flags stay consistent with stored positions; explicit box origin/cell; XYZ writer; lazy coreio."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs.lammpsio import read_lammps_frame, read_input_datafile

FIXTURES = Path(__file__).parent / "lammpsio_tests"


def dump(rows, header, box="ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10"):
    return (f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(rows)}\n{box}\n"
            f"ITEM: ATOMS {header}\n" + "\n".join(rows) + "\n")


class ImageFlagTests(unittest.TestCase):
    def load(self, text, **options):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.lammpstrj"
            path.write_text(text)
            return read_lammps_frame(path, units="metal", **options)

    @staticmethod
    def unwrapped(frame):
        atoms = frame.atoms
        return atoms.positions + atoms.arrays["image_flags"] @ atoms.cell.array

    def test_wrap_keeps_unwrapped_positions_invariant(self):
        # Positions outside the box, with nonzero dumped images, in a triclinic cell.
        box = "ITEM: BOX BOUNDS xy xz yz pp pp pp\n0 12 2\n0 10 0\n0 10 0"
        rows = ["1 C 10.4 1 1 0 0 0", "2 C 11.9 9.5 1 1 -1 0", "3 C -3 -12 25 0 2 -1"]
        text = dump(rows, "id element x y z ix iy iz", box)
        plain, wrapped = self.load(text), self.load(text, wrap=True)
        assert_allclose(self.unwrapped(wrapped), self.unwrapped(plain), atol=1e-12)
        self.assertFalse(np.allclose(wrapped.atoms.positions, plain.atoms.positions))
        np.testing.assert_array_equal(wrapped.columns["ix"], plain.columns["ix"])   # raw columns untouched

    def test_unwrapped_columns_do_not_double_count_images(self):
        wrapped = self.load(dump(["1 C 2 3 4 1 0 -1"], "id element x y z ix iy iz"))
        unwrapped = self.load(dump(["1 C 12 3 -6 1 0 -1"], "id element xu yu zu ix iy iz"))
        np.testing.assert_array_equal(unwrapped.atoms.arrays["image_flags"], [[0, 0, 0]])
        assert_allclose(self.unwrapped(unwrapped), self.unwrapped(wrapped))
        assert_allclose(self.unwrapped(self.load(dump(["1 C 12 3 -6 1 0 -1"], "id element xu yu zu ix iy iz"),
                                                 wrap=True)), self.unwrapped(wrapped))

    def test_box_bounds_report_true_origin_and_cell(self):
        frame = read_lammps_frame(FIXTURES / "sheared_frames.lammpstrj", 9, units="metal")
        assert_allclose(frame.box_bounds["origin"], frame.atoms.get_celldisp().ravel())
        assert_allclose(frame.box_bounds["cell"], frame.atoms.cell.array)
        self.assertGreater(frame.box_bounds["cell"][1][0], 0)       # tilt present
        self.assertNotEqual(frame.box_bounds["xhi"] - frame.box_bounds["xlo"], frame.box_bounds["cell"][0][0])
        data_bounds = read_input_datafile(FIXTURES / "frame_0.data", units="metal")[2]
        assert_allclose(np.asarray(data_bounds["cell"]).diagonal(),
                        [data_bounds["xhi"] - data_bounds["xlo"], data_bounds["yhi"] - data_bounds["ylo"],
                         data_bounds["zhi"] - data_bounds["zlo"]])


class WriterTests(unittest.TestCase):
    def test_xyz_writer_optional_velocities_and_full_precision(self):
        from MartiniFuncs.geom_utils import pair_data_to_xyz
        record = dict(positions=[[0., 0, 0], [1, 0, 0]], symbols=["H", "H"],
                      forces=[[1.23456789e-9, 0, 0], [0, 0, -2.5]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.xyz"
            pair_data_to_xyz([record], path)
            lines = path.read_text().splitlines()
        self.assertIn("columns=x,y,z,fx,fy,fz", lines[1])
        self.assertEqual(float(lines[2].split()[4]), 1.23456789e-9)   # %12.8f wrote 0.00000000


class CoreioImportTests(unittest.TestCase):
    def test_mass_helpers_import_without_psience(self):
        code = ("import sys; sys.modules['Psience'] = None; sys.modules['McUtils'] = None\n"
                "from MartiniFuncs import coreio\nprint(coreio.get_symbol_from_mass(12.0))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=Path(__file__).parents[1])
        self.assertEqual(out.stdout.strip(), "C", out.stderr)

    def test_no_evaluator_option(self):
        try:
            from MartiniFuncs.coreio import ase_atoms_to_molecule
            from ase import Atoms
            ase_atoms_to_molecule(Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]), energy_evaluator=None)
        except ImportError:
            self.skipTest("Psience is not installed")


if __name__ == "__main__":
    unittest.main()
