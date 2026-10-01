"""Single-pass dump reader: fast column parse equals the strict parser; anomalies fall back."""
import unittest
from pathlib import Path

import numpy as np

from MartiniFuncs import lammpsio

FIXTURES = Path(__file__).parent / "lammpsio_tests"


def atom_lines(path, frame=0):
    block = list(lammpsio._iter_blocks(path))[frame]
    lines = block.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS "))
    count = int(lines[lines.index("ITEM: NUMBER OF ATOMS") + 1])
    return lines[header].split()[2:], lines[header + 1:header + 1 + count]


class FastColumnTests(unittest.TestCase):
    def test_fast_columns_are_bit_identical_to_strict_parse(self):
        for path in (FIXTURES / "multiple_frames.lammpstrj", FIXTURES / "sheared_frames.lammpstrj",
                     FIXTURES / "frame_0.lammpstrj"):
            names, lines = atom_lines(path)
            fast, strict = lammpsio._atom_columns_fast(lines, names), lammpsio._atom_columns_strict(lines, names)
            self.assertIsNotNone(fast, path)
            self.assertEqual(list(fast), list(strict))
            for name in names:
                self.assertEqual(fast[name].dtype.kind, strict[name].dtype.kind, name)
                np.testing.assert_array_equal(fast[name], strict[name])

    def test_anomalies_defer_to_strict_parser(self):
        names = ["id", "element", "x", "y"]
        good = ["1 C 0.5 1.5", "2 H 2.5 3.5"]
        self.assertIsNotNone(lammpsio._atom_columns_fast(good, names))
        for lines in (["1 C 0.5 1.5 9", "2 H 2.5 3.5"],   # surplus column loadtxt would ignore
                      ["1 C 0.5", "2 H 2.5 3.5"],          # missing column
                      ["1  C 0.5 1.5", "2 H 2.5 3.5"],     # irregular spacing
                      ["1\tC 0.5 1.5", "2 H 2.5 3.5"],     # tab
                      ["1.0 C 0.5 1.5", "2 H 2.5 3.5"],    # non-integer ID token
                      ["1 C nan 1.5", "2 H 2.5 3.5"]):     # non-finite
            self.assertIsNone(lammpsio._atom_columns_fast(lines, names), lines)

    def test_blocks_split_identically_with_tiny_reads(self):
        import tempfile
        frame = "ITEM: TIMESTEP\n{}\nITEM: NUMBER OF ATOMS\n1\nITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 1\nITEM: ATOMS id x y z\n1 0 0 0\n"
        text = "\n" + "".join(frame.format(step) for step in range(5))
        expected = [frame.format(step) for step in range(5)]
        original = lammpsio._READ_BLOCK
        self.addCleanup(setattr, lammpsio, "_READ_BLOCK", original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.lammpstrj"
            path.write_text(text)
            for size in (1, 2, 13, 14, 15, 16, 17, 4096):   # every split position of the marker
                lammpsio._READ_BLOCK = size
                self.assertEqual(list(lammpsio._iter_blocks(path)), expected, size)
        lammpsio._READ_BLOCK = 4096
        blocks = list(lammpsio._iter_blocks(FIXTURES / "multiple_frames.lammpstrj"))
        lammpsio._READ_BLOCK = original
        self.assertEqual(blocks, list(lammpsio._iter_blocks(FIXTURES / "multiple_frames.lammpstrj")))
        self.assertEqual(len(blocks), 9)


if __name__ == "__main__":
    unittest.main()
