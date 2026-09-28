"""Execute the general notebooks on temporary inputs, without research caches."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.build import molecule
from ase import Atoms
from ase.io import write, read


class WorkflowNotebookTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        # Deliberately no LAMMPS input file or saved reports.
        shutil.copyfile(self.root/"tests/lammpsio_tests/multiple_frames.lammpstrj",
                        self.directory/"trajectory.lammpstrj")
        self.addCleanup(plt.close,"all")

    def run_notebook(self, name, **overrides):
        path = self.root/"examples/workflows"/name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {}
        with patch("IPython.display.display") as display, patch("matplotlib.pyplot.show"), redirect_stdout(StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                exec(compile("".join(cell["source"]),str(path)+":"+cell["id"],"exec"),namespace)
                if "configuration" in cell["metadata"].get("tags",[]):
                    namespace.update(RUN_DIR=self.directory,SIMULATION_DIR=self.directory,**overrides)
        return namespace, display

    def test_inspection_statistics_and_stability(self):
        inspected,_ = self.run_notebook("01_read_and_inspect_trajectory.ipynb")
        self.assertEqual(len(inspected["inventory"]),80)
        result,_ = self.run_notebook("04_molecular_statistics.ipynb",COMPUTE_PAIRS=True,
                                   DIFFUSION_FIT_PS=(.5,1.5))
        self.assertEqual(len(result["frames"]),9)
        stable,_ = self.run_notebook("05_stability_and_structure.ipynb",N_BLOCKS=2,
                                   BATCH_DURATIONS_PS=[1.,1.5])
        self.assertEqual(stable["positions"].shape,(9,80,3))
        self.assertEqual(len(stable["nearest_rows"]),6)
        # Instantaneous wrapped geometry yields small box-centered coordinates.
        self.assertTrue(np.all(abs(stable["positions"]) <= stable["boxes"][:,None,:]/2+1e-8))

    def test_general_species_packing_exports(self):
        templates = []
        for name,atoms in (("water",molecule("H2O")),("methane",molecule("CH4")),("argon",Atoms("Ar"))):
            path = self.directory/f"{name}.xyz"
            write(path,atoms)
            templates.append((name,path,4))
        namespace,_ = self.run_notebook("02_build_initial_configurations.ipynb",MOLECULE_FILES=templates,
                                       OUTPUT_DIR=self.directory/"packed",WRITE_FILES=True)
        self.assertEqual(len(namespace["packed"]),36)
        self.assertTrue((self.directory/"packed/npt.in").is_file())

    def test_diffusion_assessment_short_dump_is_inconclusive(self):
        (self.directory/"run.out").write_text("Step Time Temp Press PotEng\n" + "".join(
            f"{step} {step*.0005} {400+step/8000} 25000 -1000\n" for step in range(0, 8001, 2000)))
        # Simulate an open kernel retaining the pre-dump_file function signature.
        with patch("MartiniFuncs.thermo.attach_thermo", lambda frames, thermo, block=None: None):
            result, _ = self.run_notebook("08_diffusion_and_equilibration.ipynb",
                BOOTSTRAP_RESAMPLES=100, WRITE_FILES=True, OUTPUT_DIR=self.directory/"assessment")
        table = result["diffusion_table"]
        self.assertEqual(len(table), 12)
        self.assertTrue((table["status"] == "insufficient sampling").all())
        self.assertTrue((result["diffusion_summary"]["Finding"] == "Requested lag ranges cannot be evaluated").all())
        self.assertTrue((result["sampling_table"]["Complete blocks"] < result["sampling_table"]["Required blocks"]).all())
        self.assertTrue((self.directory/"assessment/diffusion_findings.csv").is_file())
        self.assertTrue((self.directory/"assessment/settings.json").is_file())
        self.assertEqual(result["thermo_coverage"]["matched_frames"], 5)
        self.assertEqual(result["frames"]["temperature_K"].notna().sum(), 5)
        self.assertIn("pressure_bar", result["scalar_summary"]["Observable"].tolist())

    def test_diffusion_assessment_plots_block_intervals(self):
        rng = np.random.default_rng(51)
        p = rng.normal(scale=.4, size=(1000, 20, 3)).cumsum(axis=0)
        synthetic = dict(frames=[dict(frame=i, time_ps=float(i), density_g_cm3=float(1+rng.normal(0, .001)),
                                      volume_angstrom3=1000.) for i in range(len(p))],
                         species=np.array(["synthetic"]*20), com_angstrom=p,
                         wrapped_com_angstrom=(p+5) % 10-5,
                         box_lengths_angstrom=np.ones((len(p), 3))*10,
                         frames_read=len(p), settings={}, files={}, notices=[])
        with patch("MartiniFuncs.trajectory_analysis.analyze_run", return_value=synthetic):
            result, _ = self.run_notebook("08_diffusion_and_equilibration.ipynb",
                FIT_INTERVALS_PS=[(1, 4), (2, 5), (3, 6)], BLOCK_DURATIONS_PS=[60, 120],
                BOOTSTRAP_RESAMPLES=100, SCALAR_TOLERANCES={"density_g_cm3": .01}, SHOW_DETAILS=True)
        self.assertTrue(np.isfinite(result["diffusion_table"]["D_m2_s"]).all())

    def test_pair_export_reaction_preparation_and_manual_mode(self):
        output = self.directory/"extracted"
        selected,display = self.run_notebook("03_select_pairs_and_environments.ipynb",
                                            WRITE_FILES=True,OUTPUT_DIR=output)
        self.assertEqual(len(selected["pair"]),34)
        figures = [c.args[0] for c in display.call_args_list if hasattr(c.args[0],"to_widget")]
        self.assertIn("x3d",figures[-1]._repr_html_().lower())
        prepared,_ = self.run_notebook("07_prepare_reaction_inputs.ipynb",CLUSTER_FILE=output/"cluster.extxyz",
                                      OUTPUT_DIR=self.directory/"prepared",WRITE_FILES=True)
        np.testing.assert_allclose(prepared["source_pair_forces"],selected["pair"].arrays["lammps_forces"])
        self.assertEqual(len(prepared["environment"].constraints[0].get_indices()),len(prepared["shell_rows"]))
        manual,_ = self.run_notebook("03_select_pairs_and_environments.ipynb",
                                    MANUAL_PAIR_IDS=selected["pairs"][0],MANUAL_SHIFT=selected["shifts"][0])
        self.assertIsNone(manual["assessment"])
        np.testing.assert_allclose(manual["pair"].positions,selected["pair"].positions)

    def test_force_notebook_is_opt_in_and_passes_explicit_settings(self):
        with patch("MartiniFuncs.force_validation.validate_forces") as validate:
            self.run_notebook("06_reproduce_uma_forces.ipynb")
            validate.assert_not_called()
            validate.return_value = {"frames":[],"passed":True}
            self.run_notebook("06_reproduce_uma_forces.ipynb",RUN_VALIDATION=True,CHARGE=0,SPIN_MULTIPLICITY=1)
            self.assertEqual(validate.call_args.kwargs["units"],"metal")
            self.assertEqual(validate.call_args.kwargs["task"],"omol")


if __name__ == "__main__":
    unittest.main()
