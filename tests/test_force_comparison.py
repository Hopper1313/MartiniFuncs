"""Fast unit tests for force-comparison code; no UMA reproduction is run here."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from ase.calculators.calculator import Calculator, all_changes

from MartiniFuncs.workflows.force_validation import (compare_frame_forces, local_uma_calculator,
                                                     validate_forces, MD_INFERENCE_SETTINGS)
from MartiniFuncs.io.lammps import read_lammps_frame


class HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, sign=-1):
        super().__init__()
        self.sign = sign

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": float(np.sum(atoms.positions**2)),
                        "forces": self.sign * 2 * atoms.positions}


class ForceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "forces.dump"
        self.path.write_text(
            "ITEM: TIMESTEP\n10\nITEM: NUMBER OF ATOMS\n2\n"
            "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
            "ITEM: ATOMS id element x y z fx fy fz\n"
            "9 H 1 2 3 -2 -4 -6\n2 H 2 3 4 -4 -6 -8\n"
        )
        self.frame = read_lammps_frame(self.path, units="metal")
        metadata = patch("MartiniFuncs.workflows.force_validation.version", return_value="test")
        metadata.start()
        self.addCleanup(metadata.stop)

    def test_exact_reproduction_does_not_mutate_frame(self):
        positions = self.frame.atoms.positions.copy()
        report = compare_frame_forces(self.frame, HarmonicCalculator(), charge=0, spin=1, atol=1e-12)
        self.assertTrue(report["passed"])
        self.assertEqual(report["component_rmse"], 0)
        self.assertIsNone(self.frame.atoms.calc)
        self.assertNotIn("charge", self.frame.atoms.info)
        assert_array_equal(positions, self.frame.atoms.positions)

    def test_wrong_sign_and_wrong_units_fail(self):
        report = compare_frame_forces(self.frame, HarmonicCalculator(sign=1), charge=0, spin=1, atol=.01)
        self.assertFalse(report["passed"])
        self.assertEqual(report["max_component_error"], 16)
        self.assertEqual(report["worst_lammps_id"], 2)
        wrong_units = read_lammps_frame(self.path, units="real")
        report = compare_frame_forces(wrong_units, HarmonicCalculator(), charge=0, spin=1, atol=.01)
        self.assertFalse(report["passed"])

    def test_invalid_tolerance_and_absent_local_model(self):
        with self.assertRaises(ValueError):
            compare_frame_forces(self.frame, HarmonicCalculator(), charge=0, spin=1, atol=-1)
        with self.assertRaises(FileNotFoundError):
            local_uma_calculator(Path(self.temp.name) / "missing.pt", task="omol",
                                 device="cpu", inference_settings="default")

    def test_nonfinite_source_forces_are_rejected(self):
        self.frame.atoms.arrays["lammps_forces"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Source forces must be finite"):
            compare_frame_forces(self.frame, HarmonicCalculator(), charge=0, spin=1, atol=.01)

    def test_selected_frames_reuse_model_and_report_timings(self):
        original = self.path.read_text()
        self.path.write_text(original + original.replace("TIMESTEP\n10", "TIMESTEP\n20")
                             + original.replace("TIMESTEP\n10", "TIMESTEP\n30"))
        checkpoint = Path(self.temp.name) / "model.pt"
        checkpoint.write_bytes(b"test checkpoint placeholder")
        calc = HarmonicCalculator()
        calc.martini_inference_settings = MD_INFERENCE_SETTINGS.copy()
        with patch("MartiniFuncs.workflows.force_validation.local_uma_calculator", return_value=calc) as loader:
            input_file = Path(self.temp.name)/"in.continue"
            input_file.write_text("include unresolved_settings.in\n")
            result = validate_forces(self.path, checkpoint, units="metal", indices=[2, 0],
                                      input_file=input_file, charge=0, spin=1, atol=1e-12)
        loader.assert_called_once()
        self.assertTrue(result["passed"])
        self.assertEqual([r["timestep"] for r in result["frames"]], [10, 30])
        self.assertTrue(all(r["evaluation_seconds"] >= 0 for r in result["frames"]))
        self.assertEqual(result["inference_settings"], MD_INFERENCE_SETTINGS)
        self.assertEqual(len(result["checkpoint_sha256"]), 64)

    def test_default_one_frame_and_missing_frame(self):
        checkpoint = Path(self.temp.name) / "model.pt"
        checkpoint.write_bytes(b"placeholder")
        calc = HarmonicCalculator()
        calc.martini_inference_settings = MD_INFERENCE_SETTINGS.copy()
        with patch("MartiniFuncs.workflows.force_validation.local_uma_calculator", return_value=calc) as loader:
            result = validate_forces(self.path, checkpoint, units="metal", charge=0, spin=1, atol=.01)
            self.assertEqual(len(result["frames"]), 1)
            with self.assertRaises(IndexError):
                validate_forces(self.path, checkpoint, units="metal", indices=[2], charge=0, spin=1, atol=.01)
            self.assertEqual(loader.call_count, 1)
        for indices in ([], [-1], [0, 0]):
            with self.assertRaises(ValueError):
                validate_forces(self.path, checkpoint, units="metal", indices=indices, charge=0, spin=1, atol=.01)


if __name__ == "__main__":
    unittest.main()
