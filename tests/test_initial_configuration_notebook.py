"""Run the packing notebook, including exports, without launching a simulation."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class InitialConfigurationNotebookTests(unittest.TestCase):
    def test_clean_execution_and_export(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "examples/03_build_initial_configurations.ipynb").read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        with tempfile.TemporaryDirectory() as directory, \
                patch("pathlib.Path.cwd", return_value=root / "examples"), \
                patch("IPython.display.display") as display, redirect_stdout(StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
                if "WRITE_FILES = False" in "".join(cell["source"]):
                    namespace["WRITE_FILES"] = True
                    namespace["OUTPUT_DIR"] = Path(directory)
            self.assertEqual(len(namespace["packed"]), 1360)
            self.assertGreaterEqual(namespace["minimum_distance"], 2.)
            for name in ("initial.data", "initial.data.json", "initial.extxyz", "npt.in", "nvt.in", "nve.in"):
                self.assertTrue((Path(directory) / name).is_file())
            figures = [call.args[0] for call in display.call_args_list
                       if hasattr(call.args[0], "to_widget")]
            self.assertEqual(len(figures), 4)
            for fig in figures:
                self.assertIn("x3d", fig._repr_html_().lower())
            # Default protection must reject a repeated export.
            export = next(c for c in notebook["cells"] if c["cell_type"] == "code"
                          and "if WRITE_FILES:" in "".join(c["source"]))
            with self.assertRaises(FileExistsError):
                exec("".join(export["source"]), namespace)


if __name__ == "__main__":
    unittest.main()
