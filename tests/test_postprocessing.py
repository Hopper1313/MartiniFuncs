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
                '--timestep-ps', '.0005', '--shell-radius', '2', '--no-lag-statistics', '--quiet',
                '--pair-model', 'cpd_n4t']
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
        options = dict(units='metal', timestep_ps=.0005, compute_pairs=True, pair_model='cpd_n4t',
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

    def test_pair_trajectories_flag_every_frame(self):
        """One periodic file per collected pair, every selected frame, flags from the cached criteria."""
        from MartiniFuncs.pairs.candidates import candidate_mask
        from MartiniFuncs.pairs.model import get_pair_model
        cache = self.output / 'cache'
        cache_analysis(self.run, cache, units='metal', timestep_ps=.0005, compute_pairs=True, pair_model='cpd_n4t',
                       progress=None)
        result, _ = load_analysis_cache(cache)
        candidates = result['pair_candidates']
        criteria = get_pair_model('cpd_n4t').criteria_class(max_screen_distance=3.1)
        status, errors = self.call(['trajectories', cache, '--output', self.root / 'traj', '--quiet',
                                    '--max-screen-distance', '3.1', '--workers', '2', '--batch-size', '4'])
        self.assertEqual(status, 0, errors)
        export_pairs(cache, self.root / 'plain', progress=None)
        plain = read(self.root / 'plain/pairs.extxyz', index=':')
        # Expected flags straight from the cached candidates.
        accepted = candidate_mask(candidates, criteria)
        expected = {}
        for frame, (lo, hi) in enumerate(zip(candidates['offsets'][:-1], candidates['offsets'][1:])):
            for row in range(lo, hi):
                expected[(frame, *candidates['molecules'][row].tolist())] = bool(accepted[row])
        summary = (self.root / 'traj/pairs.csv').read_text().splitlines()
        files = sorted((self.root / 'traj').glob('pair_*.extxyz'))
        self.assertEqual(len(files), len({key[1:] for key in expected}))
        self.assertEqual(len(summary), len(files) + 1)
        center = np.asarray(result['cell_angstrom'][0]).sum(axis=0) / 2 + result['cell_origin_angstrom'][0]
        for path in files:
            first, second = (int(part.lstrip('cpdmal')) for part in path.stem.split('_')[2:])
            frames = read(path, index=':')
            self.assertEqual([a.info['frame'] for a in frames], list(range(9)))
            for atoms in frames:
                key = (atoms.info['frame'], first, second)
                self.assertTrue(atoms.pbc.all())
                assert_allclose(atoms.cell.array, result['cell_angstrom'][atoms.info['frame']])
                self.assertEqual(atoms.info['collected'], key in expected)
                self.assertEqual(atoms.info['accepted'], expected.get(key, False))
                self.assertEqual(atoms.arrays['accepted_binary'].tolist(), [int(expected.get(key, False))] * len(atoms))
                self.assertEqual(np.isnan(atoms.info['screen_distance']), key not in expected)
                for name in ('screen_distance', 'plane_angle', 'ring_dihedral', 'geometry_ok'):
                    assert_allclose(atoms.arrays[name], np.full(len(atoms), atoms.info[name]), equal_nan=True)
                # The first member follows its cached unwrapped COM (masses differ slightly from the cache's).
                role = atoms.arrays['pair_role'] == 'CPD'
                masses = atoms.get_masses()[role]
                com = masses @ atoms.positions[role] / masses.sum()
                assert_allclose(com, result['com_angstrom'][atoms.info['frame']][first] + center, atol=.1)
                if key in expected:
                    # Same structure as extract, up to one lattice translation of the whole pair.
                    match, = [p for p in plain if p.info['frame_index'] == key[0]
                              and set(p.arrays['global_index']) == set(atoms.arrays['global_index'])]
                    order = {g: i for i, g in enumerate(match.arrays['global_index'])}
                    rows = [order[g] for g in atoms.arrays['global_index']]
                    shift = atoms.positions - match.positions[rows]
                    assert_allclose(shift, np.broadcast_to(shift[0], shift.shape), atol=1e-9)
                    fractional = shift[0] @ np.linalg.inv(atoms.cell.array)
                    assert_allclose(fractional, np.rint(fractional), atol=1e-9)
                    assert_allclose(atoms.arrays['lammps_forces'], match.arrays['lammps_forces'][rows])
        # all_pairs.extxyz: per frame, every pair's structure concatenated in pair order, labelled by pair_id.
        together = read(self.root / 'traj/all_pairs.extxyz', index=':')
        per_pair = [read(path, index=':') for path in files]
        self.assertEqual(len(together), 9)
        for t, atoms in enumerate(together):
            parts = [frames[t] for frames in per_pair]
            self.assertEqual(atoms.info['frame'], t)
            self.assertEqual(atoms.info['n_accepted'], sum(p.info['accepted'] for p in parts))
            assert_allclose(atoms.positions, np.vstack([p.positions for p in parts]))
            self.assertEqual(atoms.arrays['pair_id'].tolist(), [i for i, p in enumerate(parts) for _ in p])
            self.assertEqual(atoms.arrays['accepted_binary'].tolist(),
                             [b for p in parts for b in p.arrays['accepted_binary'].tolist()])
            assert_allclose(atoms.arrays['screen_distance'],
                            np.concatenate([p.arrays['screen_distance'] for p in parts]), equal_nan=True)
        status, _ = self.call(['trajectories', cache, '--output', self.root / 'traj', '--quiet', '--every', '2'])
        self.assertEqual(status, 2)        # existing output with different settings is never overwritten
        status, errors = self.call(['trajectories', cache, '--output', self.root / 'thin', '--quiet', '--every', '3',
                                    '--startframe', '1'])
        self.assertEqual(status, 0, errors)
        thinned = read(next((self.root / 'thin').glob('pair_*.extxyz')), index=':')
        self.assertEqual([a.info['frame'] for a in thinned], [1, 4, 7])

    def test_exports_read_a_moved_dump(self):
        """A cache made elsewhere exports from a same-size copy of its dump; other files are refused."""
        from MartiniFuncs.workflows.postprocess import export_pair_trajectories
        cache = self.output / 'cache'
        cache_analysis(self.run, cache, units='metal', timestep_ps=.0005, compute_pairs=True, pair_model='cpd_n4t',
                       progress=None)
        export_pairs(cache, self.root / 'original', progress=None)
        moved = self.root / 'laptop' / 'copy.lammpstrj'
        moved.parent.mkdir()
        shutil.move(self.dump, moved)
        with self.assertRaises(FileNotFoundError):
            export_pairs(cache, self.root / 'refused', progress=None)
        export_pairs(cache, self.root / 'moved', dump_file=moved, progress=None)
        # Identical structures; only the recorded source path differs.
        original = read(self.root / 'original/pairs.extxyz', index=':')
        moved_pairs = read(self.root / 'moved/pairs.extxyz', index=':')
        self.assertEqual(len(original), len(moved_pairs))
        for a, b in zip(original, moved_pairs):
            a.info.pop('source'), b.info.pop('source')
            self.assertEqual(a.info.keys(), b.info.keys())
            for key in a.arrays:
                np.testing.assert_array_equal(a.arrays[key], b.arrays[key])
        status, errors = self.call(['trajectories', cache, '--output', self.root / 'traj', '--quiet',
                                    '--dump-file', moved])
        self.assertEqual(status, 0, errors)
        truncated = self.root / 'laptop' / 'short.lammpstrj'
        truncated.write_bytes(moved.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError, 'not the same trajectory'):
            export_pair_trajectories(cache, self.root / 'short', dump_file=truncated, progress=None)

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
                "--stability", "--max-lag-ps", "1.5", "--fit-ps", ".5", "1.5", "--quiet", "--pair-model", "cpd_n4t"]
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
                "--workers", "2", "--batch-size", "2", "--no-export-pairs", "--quiet", "--pair-model", "cpd_n4t"]
        status, message = self.call(args)
        self.assertEqual(status, 0, message)
        self.assertTrue((self.output / "reports/default/pair_lifetimes.csv").exists())
        result, _ = load_analysis_cache(self.output / "cache")
        self.assertEqual(result["pair_analysis"], {"model": "cpd_n4t", "collection": {"reactive_radius": 3.25},
                                                   "backend": "cached", "workers": 2, "batch_size": 2})
        export_pairs(self.output / "cache", self.output / "late_pairs", startframe=6, endframe=9, progress=None)
        lines = (self.output / "late_pairs/index.jsonl").read_text().splitlines()
        self.assertTrue(all(3 <= json.loads(line)["time_ps"] <= 4 for line in lines))

    def test_run_selection_applies_to_reports_and_exports_but_not_cache(self):
        args = ["run", self.run, "--output", self.output, "--units", "metal", "--timestep-ps", ".0005",
                "--startframe", "0", "--endframe", "5", "--quiet", "--pair-model", "cpd_n4t"]
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
                "--batch-size", "2", "--quiet", "--pair-model", "cpd_n4t"]
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
            "--timestep-ps", ".0005", "--shell-radius", "2", "--max-forming-distance", "0", "--quiet",
            "--pair-model", "cpd_n4t"])
        self.assertEqual(status, 0, errors)
        import pandas as pd
        coverage = pd.read_csv(self.output/"pairs/frames.csv")
        self.assertEqual(coverage.frame.tolist(), list(range(9)))
        self.assertTrue((coverage.pair_count == 0).all())
        self.assertEqual((self.output/"pairs/pairs.extxyz").stat().st_size, 0)
        self.assertEqual((self.output/"pairs/clusters.extxyz").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
