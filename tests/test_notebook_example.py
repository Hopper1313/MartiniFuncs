"""Execute the example's code cells in order with an empty namespace, no model."""
from contextlib import redirect_stdout
from io import StringIO
import json
from importlib.util import find_spec
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


@unittest.skipUnless(find_spec("IPython"), "Install the notebooks extra for the notebook smoke test")
class NotebookExampleTests(unittest.TestCase):
    def test_clean_execution_and_export_identity(self):
        self._run_example()

    @unittest.skipUnless(
        (Path(__file__).parent / "lammpsio_tests/more_test_data/2_337_shear_only.lammpstrj").exists(),
        "Optional shear trajectory is not installed",
    )
    def test_shear_frame_zero_execution(self):
        self._run_example(dataset="shear")

    def _run_example(self, dataset="equilibration"):
        from McUtils.Plots import Disk, Line
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "examples/01_extract_and_visualize.ipynb").read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        with patch("pathlib.Path.cwd", return_value=root), \
                patch("McUtils.Plots.Disk", wraps=Disk) as disks, \
                patch("McUtils.Plots.Line", wraps=Line) as lines, \
                patch("IPython.display.display") as display, redirect_stdout(StringIO()):
            for i, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    source = source.replace('DATASET = "equilibration"', f'DATASET = "{dataset}"')
                    exec(compile(source, f"notebook-cell-{i}", "exec"), namespace)
        cluster, frame = namespace["cluster"], namespace["frame"]
        source = {value: i for i, value in enumerate(frame.atoms.arrays["global_index"])}
        rows = [source[value] for value in cluster.arrays["global_index"]]
        np.testing.assert_allclose(cluster.arrays["lammps_forces"],
                                   frame.atoms.arrays["lammps_forces"][rows], atol=1e-8)
        self.assertEqual(set(cluster.arrays["pair_role"]), {"CPD", "MAL", "shell"})
        self.assertFalse(cluster.pbc.any())
        self.assertIsNone(cluster.calc)
        self.assertGreaterEqual(display.call_count, 4)
        # Exercise HTML serialization as well as Python figure construction.
        figures = [call.args[0] for call in display.call_args_list
                   if hasattr(call.args[0], "to_widget")]
        self.assertEqual(len(figures), 3)
        for figure in figures:
            html = figure._repr_html_()
            self.assertIn("x3d", html.lower())
            self.assertNotIn("nan", html.lower())
        geometry = namespace["selected"].geometry
        pair = namespace["pair"]
        mapped = tuple(tuple(int(pair.arrays["global_index"][i]) for i in bond)
                       for bond in geometry.forming_bonds_local)
        self.assertEqual(mapped, geometry.forming_bonds)
        # Overlay coordinates must be in the figure's Angstrom units, not Bohr.
        expected_centers = pair.positions[np.asarray(geometry.forming_bonds_local)].mean(axis=0)
        self.assertEqual(disks.call_count, 2)
        for call, center in zip(disks.call_args_list, expected_centers):
            np.testing.assert_allclose(call.args[0], center)
            self.assertEqual(call.args[1], 2.0)
            self.assertEqual(call.kwargs["transparency"], .5)
        np.testing.assert_allclose(lines.call_args.args[0], expected_centers)
        normals = [call.kwargs["normal"] for call in disks.call_args_list]
        plane_angle = np.degrees(np.arccos(np.clip(abs(np.dot(*normals)), 0, 1)))
        self.assertAlmostEqual(plane_angle, geometry.plane_angle, places=5)
        np.testing.assert_allclose(
            [pair.get_distance(i, j) for i, j in geometry.forming_bonds_local],
            geometry.forming_distances, atol=2e-8)
        local = {value: i for i, value in enumerate(cluster.arrays["global_index"])}
        distances = [cluster.get_distance(local[i], local[j])
                     for i, j in geometry.forming_bonds]
        np.testing.assert_allclose(distances, geometry.forming_distances, atol=2e-8)


if __name__ == "__main__":
    unittest.main()
