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
import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs import md_statistics as md
from MartiniFuncs.trajectory_analysis import analyze_run, molecular_window_statistics, resolve_run_files, select_frame_window


class TrajectoryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        fixtures = Path(__file__).parent / "lammpsio_tests"
        shutil.copyfile(fixtures / "infile.in", self.directory / "infile.in")
        shutil.copyfile(fixtures / "multiple_frames.lammpstrj", self.directory / "trajectory.lammpstrj")

    def test_molecular_scope_and_reference_statistics(self):
        result = analyze_run(self.directory, degrees_of_freedom="3N-3")
        self.assertEqual(result["frames_read"], 9)
        self.assertEqual(result["com_angstrom"].shape, (9, 80, 3))
        window = molecular_window_statistics(result, max_lag_ps=2)
        times = np.array([r["time_ps"] for r in result["frames"]])
        for name, entry in window["species"].items():
            mask = result["species"] == name
            self.assertEqual(entry["n_molecules"], 40)
            reference = md.get_molecular_msd(result["com_angstrom"][:,mask], times, unwrapped=True, selection=name)
            assert_allclose(entry["msd_angstrom2"], reference["msd_angstrom2"][:5])
            corr = md.get_molecular_velocity_autocorrelation(result["com_velocity_angstrom_ps"][:,mask], times, selection=name)
            assert_allclose(entry["vacf"], corr["correlation"][:5])
        sampled = analyze_run(self.directory, frame_stride=2, max_frames=7)
        assert_allclose(sampled["com_angstrom"],result["com_angstrom"][:7:2])
        self.assertNotIn("temperature_K",sampled["frames"][0])

    def test_ambiguous_files_require_override(self):
        shutil.copyfile(self.directory/"trajectory.lammpstrj",self.directory/"another.dump")
        with self.assertRaisesRegex(ValueError,"explicit"):
            resolve_run_files(self.directory)
        self.assertEqual(resolve_run_files(self.directory,dump_file="another.dump")["dump"].name,"another.dump")

    def test_source_frame_bounds_and_full_default_after_striding(self):
        result = analyze_run(self.directory, frame_stride=2)
        full = molecular_window_statistics(result)
        self.assertEqual(full["frame_window"], [0, 9])
        self.assertEqual(full["time_window_ps"], [0., 4.])
        self.assertTrue(full["frame_mask"].all())
        window = molecular_window_statistics(result, startframe=3, endframe=9)
        self.assertEqual(window["frame_window"], [3, 9])
        self.assertEqual(window["time_window_ps"], [2., 4.])
        self.assertEqual(np.flatnonzero(window["frame_mask"]).tolist(), [2, 3, 4])
        self.assertEqual(select_frame_window(result, startframe=0, endframe=4).tolist(), [True, True, False, False, False])
        for bounds in ({"startframe": -1}, {"startframe": True}, {"endframe": 2.5},
                       {"startframe": 4, "endframe": 4}, {"endframe": 10},
                       {"startframe": 1, "endframe": 2}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                select_frame_window(result, **bounds)

    def test_directory_discovery_precedence_and_optional_ambiguity(self):
        nested = self.directory / "archive"
        nested.mkdir()
        (nested / "old.dump").touch()
        (nested / "old.in").touch()
        (self.directory / "initial.data").touch()
        files = resolve_run_files(self.directory)
        self.assertEqual(files["dump"], self.directory / "trajectory.lammpstrj")
        self.assertEqual(files["input"], self.directory / "infile.in")
        self.assertEqual(files["data"], self.directory / "initial.data")
        (self.directory / "final.data").touch()
        with self.assertRaisesRegex(ValueError, "data_file.*final.data.*initial.data"):
            resolve_run_files(self.directory)
        files = resolve_run_files(self.directory, data_file="initial.data", dump_file="archive/old.dump")
        self.assertEqual(files["dump"], nested / "old.dump")
        (self.directory / "in.restart").touch()
        with self.assertRaisesRegex(ValueError, "input_file"):
            resolve_run_files(self.directory, data_file="initial.data", require_input=False)

    def test_nested_discovery_and_missing_optional_files(self):
        nested = self.directory / "output"
        nested.mkdir()
        (self.directory / "trajectory.lammpstrj").rename(nested / "trajectory.lammpstrj")
        (self.directory / "infile.in").unlink()
        files = resolve_run_files(self.directory, require_input=False)
        self.assertEqual(files["dump"], nested / "trajectory.lammpstrj")
        self.assertIsNone(files["input"])
        self.assertIsNone(files["data"])
        with self.assertRaisesRegex(ValueError, "input_file"):
            resolve_run_files(self.directory)
        (self.directory / "in.production").touch()
        self.assertEqual(resolve_run_files(self.directory)["input"].name, "in.production")

    def test_restart_units_from_data_header_and_explicit_fallbacks(self):
        (self.directory/"infile.in").write_text("read_restart previous.restart\ntimestep 0.0005\n")
        with self.assertRaisesRegex(ValueError,"Units are absent"):
            analyze_run(self.directory,max_frames=3)
        explicit = analyze_run(self.directory,max_frames=3,units_style="metal")
        self.assertEqual(explicit["settings"]["units_sources"],{"explicit units":"metal"})
        source = Path(__file__).parent/"lammpsio_tests/frame_0.data"
        body = source.read_text().splitlines(keepends=True)[1:]
        data = self.directory/"final.data"
        data.write_text("LAMMPS data file via write_data, version 22 Jul 2025, timestep = 8000, units = metal\n"+"".join(body))
        inferred = analyze_run(self.directory,max_frames=3)
        self.assertEqual(inferred["settings"]["units_sources"],{"LAMMPS data header":"metal"})
        assert_allclose(inferred["com_angstrom"],explicit["com_angstrom"])
        with self.assertRaisesRegex(ValueError,"only units metal"):
            analyze_run(self.directory,max_frames=3,units_style="real")
        changed = analyze_run(self.directory,max_frames=3,timestep_ps=.001)
        self.assertEqual(changed["frames"][-1]["time_ps"],2.)
        (self.directory/"infile.in").write_text("read_restart previous.restart\n")
        with self.assertRaisesRegex(ValueError,"Timestep missing"):
            analyze_run(self.directory,max_frames=3)
        inferred = analyze_run(self.directory,max_frames=3,timestep_ps=.0005)
        self.assertEqual(inferred["frames"][-1]["time_ps"],1.)

    def test_explicit_settings_work_without_input_and_bypass_unresolved_input(self):
        path = self.directory/"infile.in"
        path.unlink()
        result = analyze_run(self.directory,units="metal",timestep_ps=.001,max_frames=3)
        self.assertIsNone(result["files"]["input"])
        self.assertEqual(result["frames"][-1]["time_ps"],2.)
        path.write_text("include missing_settings.in\ntimestep invalid\n")
        direct = analyze_run(self.directory,units="metal",timestep_ps=.001,max_frames=3)
        assert_allclose(direct["com_angstrom"],result["com_angstrom"])
        with self.assertRaisesRegex(ValueError,"must agree"):
            analyze_run(self.directory,units="metal",units_style="real",timestep_ps=.001)

    def test_missing_vectors_leave_molecular_msd_available(self):
        path = self.directory/"trajectory.lammpstrj"
        lines = path.read_text().splitlines()
        rewritten, columns, remaining = [], None, 0
        count = 0
        for index, line in enumerate(lines):
            if line == "ITEM: NUMBER OF ATOMS":
                count = int(lines[index+1])
            if line.startswith("ITEM: ATOMS "):
                names = line.split()[2:]
                columns = [i for i, name in enumerate(names) if name not in {"vx","vy","vz","fx","fy","fz"}]
                rewritten.append("ITEM: ATOMS " + " ".join(names[i] for i in columns))
                remaining = count
            elif remaining:
                fields = line.split()
                rewritten.append(" ".join(fields[i] for i in columns))
                remaining -= 1
            else:
                rewritten.append(line)
        path.write_text("\n".join(rewritten)+"\n")
        result = analyze_run(self.directory)
        self.assertIsNone(result["com_velocity_angstrom_ps"])
        self.assertIsNone(result["net_force_eV_angstrom"])
        window = molecular_window_statistics(result)
        self.assertTrue(all("msd_angstrom2" in s for s in window["species"].values()))

    def test_notebook_runs_directly_from_directory_including_pair_branch(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/"examples/04_plot_md_statistics_from_directory.ipynb").read_text(encoding="utf-8"))
        namespace = {}
        with patch("IPython.display.display"), patch("matplotlib.pyplot.show"), redirect_stdout(StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                exec(compile(source,cell["id"],"exec"),namespace)
                if "SIMULATION_DIR =" in source:
                    namespace.update(SIMULATION_DIR=self.directory,COMPUTE_PAIRS=True,
                                     DIFFUSION_FIT_PS=(.5,1.5))
        self.assertEqual(len(namespace["result"]["frames"]),9)
        self.assertEqual(namespace["window"]["time_window_ps"],[0.,4.])
        self.assertIn("lifetime_table",namespace)
        import matplotlib.pyplot as plt
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
