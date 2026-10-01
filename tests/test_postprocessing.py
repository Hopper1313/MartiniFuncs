"""End-to-end CLI/cache tests on temporary user-style simulation directories."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ase.io import read
import numpy as np
from numpy.testing import assert_allclose

from MartiniFuncs.workflows.cli import main, build_parser
from MartiniFuncs.io.lammps import read_lammps_frame
from MartiniFuncs.workflows.postprocess import cache_analysis, load_analysis_cache, write_report, export_pairs


class PostprocessingTests(unittest.TestCase):
    def test_lifetimes_shells_without_lag_calculations(self):
        args = ['run', self.run, '--output', self.output, '--units', 'metal',
                '--timestep-ps', '.0005', '--shell-radius', '2', '--no-lag-statistics', '--quiet']
        # All np.mean lag reductions must be skipped, not merely hidden in output.
        with patch('MartiniFuncs.trajectory.analysis.np.mean', side_effect=AssertionError('lag reduction called')):
            status, errors = self.call(args)
        self.assertEqual(status, 0, errors)
        report = self.output/'reports/default'
        self.assertTrue((report/'pair_lifetimes.csv').exists())
        self.assertTrue((report/'frames.csv').exists())
        self.assertEqual(list(report.glob('species_*_lags.csv')), [])
        self.assertTrue((self.output/'pairs/clusters.extxyz').exists())
        messages = []
        write_report(self.output/'cache', self.root/'rdf_only', lag_statistics=[],
                     pair_lifetimes=False, frame_summary=False, rdf_bins=5, progress=messages.append)
        self.assertTrue((self.root/'rdf_only/rdf_species.json').exists())
        self.assertFalse((self.root/'rdf_only/pair_lifetimes.csv').exists())
        self.assertFalse((self.root/'rdf_only/frames.csv').exists())
        self.assertTrue(any('Computing COM RDF' in m for m in messages))
        write_report(self.output/'cache', self.root/'vacf', lag_statistics=['vacf'])
        columns = (self.root/'vacf/species_000_lags.csv').read_text().splitlines()[0]
        self.assertIn('vacf', columns)
        self.assertNotIn('msd_angstrom2', columns)
        self.assertNotIn('force_acf', columns)
        with self.assertRaisesRegex(ValueError, 'requires msd'):
            write_report(self.output/'cache', self.root/'bad', lag_statistics=[], fit_interval_ps=(0, 1))

    def test_parallel_full_frames_and_shells_match_serial(self):
        """Worker count/chunk boundaries must not change any scientific output."""
        from MartiniFuncs.trajectory.analysis import analyze_run
        options = dict(units='metal', timestep_ps=.0005, compute_pairs=True,
                       reaction_atom_counts=(11, 23), degrees_of_freedom='3N-3')
        serial = analyze_run(self.run, **options)
        parallel = analyze_run(self.run, pair_workers=2, pair_batch_size=3, **options)
        self.assertEqual(serial['frames'], parallel['frames'])
        self.assertEqual(serial['pairs_by_frame'], parallel['pairs_by_frame'])
        self.assertEqual(serial['pair_shifts_by_frame'], parallel['pair_shifts_by_frame'])
        for key in ('com_angstrom', 'wrapped_com_angstrom', 'com_velocity_angstrom_ps',
                    'net_force_eV_angstrom', 'box_lengths_angstrom'):
            np.testing.assert_array_equal(serial[key], parallel[key])
        cache = self.output / 'cache'
        cache_analysis(self.run, cache, **options, progress=None)
        for workers in (1, 2):
            export_pairs(cache, self.root / f'export{workers}', shell_radius=2,
                         workers=workers, batch_size=3, progress=None)
        for name in ('pairs.extxyz', 'clusters.extxyz', 'index.jsonl', 'frames.csv'):
            self.assertEqual((self.root / 'export1' / name).read_bytes(),
                             (self.root / 'export2' / name).read_bytes(), name)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "simulation"
        self.run.mkdir()
        self.dump = self.run / "trajectory.lammpstrj"
        shutil.copyfile(Path(__file__).parent / "lammpsio_tests/multiple_frames.lammpstrj", self.dump)
        self.output = self.root / "analysis"

    def call(self, args):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            status = main(list(map(str, args)))
        return status, errors.getvalue()

    def test_cli_run_exports_forces_shells_and_reusable_reports(self):
        args = ["run", self.run, "--output", self.output, "--units", "metal", "--timestep-ps", ".0005",
                "--temperature-dof", "3N-3", "--shell-radius", "2", "--rdf-bins", "10",
                "--stability", "--max-lag-ps", "1.5", "--fit-ps", ".5", "1.5", "--quiet"]
        status, errors = self.call(args)
        self.assertEqual(status, 0, errors)
        cache = self.output / "cache"
        result, manifest = load_analysis_cache(cache)
        self.assertIsInstance(result["com_angstrom"], np.memmap)
        self.assertEqual(result["frames_read"], 9)
        full_report = json.loads((self.output / "reports/default/manifest.json").read_text())
        self.assertEqual(full_report["frame_window"], [0, 9])
        self.assertEqual(full_report["time_window_ps"], [0., 4.])
        self.assertEqual(manifest["units"]["force"], "eV/Angstrom")
        index = [json.loads(line) for line in (self.output / "pairs/index.jsonl").read_text().splitlines()]
        pairs = read(self.output / "pairs/pairs.extxyz", index=":")
        clusters = read(self.output / "pairs/clusters.extxyz", index=":")
        self.assertEqual(len(index), sum(len(p) for p in result["pairs_by_frame"]))
        self.assertEqual(len(pairs), len(index))
        self.assertEqual(len(clusters), len(index))
        item, pair, cluster = index[0], pairs[0], clusters[0]
        frame = read_lammps_frame(self.dump, index=item["frame"], units="metal")
        lookup = dict(zip(frame.atoms.arrays["global_index"], range(len(frame.atoms))))
        rows = [lookup[i] for i in pair.arrays["global_index"]]
        assert_allclose(pair.arrays["lammps_forces"], frame.atoms.arrays["lammps_forces"][rows])
        assert_allclose(cluster.positions[:len(pair)], pair.positions)
        ids = pair.arrays["global_index"]
        geometry = item["geometry"]
        for global_bond, local_bond in zip(geometry["forming_bonds"], geometry["forming_bonds_local"]):
            self.assertEqual(ids[local_bond].tolist(), global_bond)
        self.assertTrue((self.output / "reports/default/diffusion.csv").exists())
        self.assertTrue((self.output / "reports/default/scalar_stability.json").exists())
        # Repeating an identical run must not analyze or export the dump again.
        with patch("MartiniFuncs.workflows.postprocess.analyze_run", side_effect=AssertionError("cache not reused")):
            self.assertEqual(self.call(args)[0], 0)
        # Report generation still works after the original trajectory is removed.
        self.dump.unlink()
        status, errors = self.call(["report", cache, "--output", self.output / "reports/frames_6_9", "--startframe", "6", "--endframe", "9"])
        self.assertEqual(status, 0, errors)
        report = json.loads((self.output / "reports/frames_6_9/manifest.json").read_text())
        self.assertEqual(report["time_window_ps"], [3., 4.])
        self.assertEqual(report["frame_window"], [6, 9])
        status, _ = self.call(["extract", cache, "--output", self.output / "new_pairs"])
        self.assertEqual(status, 2)

    def test_cache_mismatch_incomplete_and_report_collision(self):
        cache = self.output / "cache"
        cache_analysis(self.run, cache, units="metal", timestep_ps=.0005, compute_pairs=False, progress=None)
        with self.assertRaisesRegex(ValueError, "does not match"):
            cache_analysis(self.run, cache, units="metal", timestep_ps=.001, compute_pairs=False, progress=None)
        write_report(cache, self.output / "report")
        with self.assertRaisesRegex(ValueError, "different settings"):
            write_report(cache, self.output / "report", startframe=2)
        self.dump.write_text(self.dump.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            cache_analysis(self.run, cache, units="metal", timestep_ps=.0005, compute_pairs=False, progress=None)
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_analysis_cache(incomplete)

    def test_generic_mode_and_slurm_allocation(self):
        args = ["run", self.run, "--output", self.output, "--units", "metal", "--timestep-ps", ".0005", "--no-pairs", "--quiet"]
        self.assertEqual(self.call(args)[0], 0)
        self.assertFalse((self.output / "pairs").exists())
        with patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "1"}):
            status, message = self.call(args + ["--workers", "2"])
            self.assertEqual(status, 2)
            self.assertIn("SLURM_CPUS_PER_TASK", message)

    def test_process_workers_and_lifetime_window(self):
        args = ["run", self.run, "--output", self.output, "--units", "metal", "--timestep-ps", ".0005",
                "--workers", "2", "--batch-size", "2", "--no-export-pairs", "--quiet"]
        status, message = self.call(args)
        self.assertEqual(status, 0, message)
        self.assertTrue((self.output / "reports/default/pair_lifetimes.csv").exists())
        result, _ = load_analysis_cache(self.output / "cache")
        self.assertEqual(result["pair_analysis"], {"backend": "cached", "workers": 2, "batch_size": 2})
        export_pairs(self.output / "cache", self.output / "late_pairs", startframe=6, endframe=9, progress=None)
        lines = (self.output / "late_pairs/index.jsonl").read_text().splitlines()
        self.assertTrue(all(3 <= json.loads(line)["time_ps"] <= 4 for line in lines))

    def test_run_selection_applies_to_reports_and_exports_but_not_cache(self):
        args = ["run", self.run, "--output", self.output, "--units", "metal", "--timestep-ps", ".0005",
                "--startframe", "0", "--endframe", "5", "--quiet"]
        status, message = self.call(args)
        self.assertEqual(status, 0, message)
        result, _ = load_analysis_cache(self.output / "cache")
        self.assertEqual(result["frames_read"], 9)
        manifest = json.loads((self.output / "reports/default/manifest.json").read_text())
        self.assertEqual(manifest["frame_window"], [0, 5])
        exported = [json.loads(line) for line in (self.output / "pairs_frames_0_5/index.jsonl").read_text().splitlines()]
        self.assertGreater(len(exported), 0)
        self.assertTrue(all(0 <= record["frame"] < 5 for record in exported))
        self.assertEqual(len(exported), sum(len(p) for p in result["pairs_by_frame"][:5]))

    def test_cli_frame_help_and_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["report", "cache", "--output", "report"])
        self.assertEqual(args.startframe, 0)
        self.assertIsNone(args.endframe)
        self.assertIn("zero-based", parser.description)
        status, message = self.call(["extract", "unused", "--output", "unused", "--startframe", "5", "--endframe", "4"])
        self.assertEqual(status, 2)
        self.assertIn("startframe < endframe", message)

    def test_collect_all_frames_parallel_identity_and_shell_forces(self):
        args = ["collect", self.run, "--output", self.output, "--units", "metal",
                "--timestep-ps", ".0005", "--shell-radius", "2", "--workers", "2",
                "--batch-size", "2", "--quiet"]
        status, errors = self.call(args)
        self.assertEqual(status, 0, errors)
        self.assertFalse((self.output/"reports").exists())
        result, manifest = load_analysis_cache(self.output/"cache")
        # collect records reaction (topology) events by default instead of aborting on them
        self.assertIsNone(result["reaction_screen"])
        self.assertEqual(manifest["configuration"]["on_topology_change"], "record")
        self.assertEqual(result["topology_events"], [])
        self.assertEqual(result["pair_analysis"]["workers"], 2)
        pairs = read(self.output/"pairs/pairs.extxyz", index=":")
        clusters = read(self.output/"pairs/clusters.extxyz", index=":")
        self.assertEqual(len(pairs), sum(len(p) for p in result["pairs_by_frame"]))
        self.assertEqual(len(pairs), len(clusters))
        source_frames = {i: read_lammps_frame(self.dump, index=i, units="metal").atoms for i in range(9)}
        for ordinal, (pair, cluster) in enumerate(zip(pairs, clusters)):
            for image in (pair, cluster):
                self.assertEqual(image.info["structure_index"], ordinal)
                source = source_frames[image.info["frame_index"]]
                rows = image.arrays["frame_local_index"]
                np.testing.assert_array_equal(image.arrays["local_index"], np.arange(len(image)))
                np.testing.assert_array_equal(image.arrays["global_index"], source.arrays["global_index"][rows])
                np.testing.assert_array_equal(image.arrays["lammps_id"], source.arrays["lammps_id"][rows])
                np.testing.assert_array_equal(image.arrays["lammps_forces"], source.arrays["lammps_forces"][rows])
            self.assertEqual(sum(pair.arrays["pair_role"] == "CPD"), 11)
            self.assertEqual(sum(pair.arrays["pair_role"] == "MAL"), 23)
            assert_allclose(cluster.positions[:len(pair)], pair.positions)
        import pandas as pd
        coverage = pd.read_csv(self.output/"pairs/frames.csv")
        self.assertEqual(coverage.frame.tolist(), list(range(9)))
        self.assertEqual(coverage.pair_count.sum(), len(pairs))
        with patch("MartiniFuncs.workflows.postprocess.analyze_run", side_effect=AssertionError("must reuse")):
            self.assertEqual(self.call(args)[0], 0)

    def test_collect_zero_pairs_still_records_every_frame(self):
        status, errors = self.call(["collect", self.run, "--output", self.output, "--units", "metal",
            "--timestep-ps", ".0005", "--shell-radius", "2", "--max-forming-distance", "0", "--quiet"])
        self.assertEqual(status, 0, errors)
        import pandas as pd
        coverage = pd.read_csv(self.output/"pairs/frames.csv")
        self.assertEqual(coverage.frame.tolist(), list(range(9)))
        self.assertTrue((coverage.pair_count == 0).all())
        self.assertEqual((self.output/"pairs/pairs.extxyz").stat().st_size, 0)
        self.assertEqual((self.output/"pairs/clusters.extxyz").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
