"""Selecting pairs from cached candidates equals a fresh analysis with those settings."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from MartiniFuncs.pairs import lifetimes
from MartiniFuncs.workflows.cli import main
from MartiniFuncs.pairs.candidates import candidate_table, select_pairs
from MartiniFuncs.chemistry.cpd_n4t.selection import PairCriteria
from MartiniFuncs.workflows.postprocess import cache_analysis, load_analysis_cache
from MartiniFuncs.trajectory.analysis import analyze_run

FIXTURES = Path(__file__).parent / "lammpsio_tests"
OPTIONS = dict(units="metal", timestep_ps=.0005, compute_pairs=True)


class PairReselectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temp.name) / "sim"
        cls.run_dir.mkdir()
        shutil.copy(FIXTURES / "multiple_frames.lammpstrj", cls.run_dir)
        cls.wide = analyze_run(cls.run_dir, reactive_radius=5.0, pair_criteria=PairCriteria(), **OPTIONS)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assertSamePairs(self, selected, reference):
        pairs, shifts = selected
        self.assertEqual(pairs, reference["pairs_by_frame"])
        self.assertEqual(shifts, [[list(map(int, s)) for s in frame] for frame in reference["pair_shifts_by_frame"]])

    def test_no_selection_returns_the_analysis_pairs(self):
        pairs, shifts = select_pairs(self.wide)
        self.assertIs(pairs, self.wide["pairs_by_frame"])
        self.assertGreater(sum(map(len, pairs)), sum(map(len, analyze_run(self.run_dir, **OPTIONS)["pairs_by_frame"])))

    def test_smaller_radius_equals_fresh_analysis(self):
        for radius in (2.5, 3.25, 4.0):
            fresh = analyze_run(self.run_dir, reactive_radius=radius, pair_criteria=PairCriteria(), **OPTIONS)
            self.assertSamePairs(select_pairs(self.wide, reactive_radius=radius, criteria=PairCriteria()), fresh)
            loose = analyze_run(self.run_dir, reactive_radius=radius, **OPTIONS)        # no geometry requirement
            self.assertSamePairs(select_pairs(self.wide, reactive_radius=radius), loose)

    def test_criteria_equal_fresh_analysis(self):
        for criteria in (PairCriteria(max_axis_angle=30), PairCriteria(max_forming_distance=4.5, max_plane_angle=40)):
            fresh = analyze_run(self.run_dir, reactive_radius=5.0, pair_criteria=criteria, **OPTIONS)
            self.assertSamePairs(select_pairs(self.wide, criteria=criteria), fresh)
            both = analyze_run(self.run_dir, reactive_radius=3.25, pair_criteria=criteria, **OPTIONS)
            self.assertSamePairs(select_pairs(self.wide, reactive_radius=3.25, criteria=criteria), both)

    def test_reference_backend_candidates_match_cached_backend(self):
        reference = analyze_run(self.run_dir, reactive_radius=5.0, pair_criteria=PairCriteria(),
                                pair_backend="reference", **OPTIONS)
        a, b = candidate_table(self.wide), candidate_table(reference)
        self.assertEqual(len(a), len(b))
        for column in ("frame", "cpd_molecule", "dienophile_molecule", "shift_x", "shift_y", "shift_z", "geometry_ok"):
            self.assertEqual(a[column].tolist(), b[column].tolist(), column)
        for column in ("screen_distance", "max_forming_distance", "axis_angle", "plane_angle", "approach_angle"):
            self.assertLess((a[column] - b[column]).abs().max(), 1e-9, column)

    def test_radius_above_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most|cached"):
            select_pairs(self.wide, reactive_radius=5.5)

    def test_cli_report_and_extract_select_after_the_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "analysis"
            args = ["run", self.run_dir, "--output", out, "--units", "metal", "--timestep-ps", ".0005",
                    "--reactive-radius", "5", "--no-export-pairs", "--max-lag-ps", ".002", "--quiet"]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main([str(a) for a in args])
            self.assertEqual(status, 0, errors.getvalue())
            cached, _ = load_analysis_cache(out / "cache")
            self.assertEqual(cached["pair_candidates"]["reactive_radius"], 5.0)
            del cached   # release memory-mapped arrays (Windows cannot delete open files)
            fresh = analyze_run(self.run_dir, reactive_radius=3.0, pair_criteria=PairCriteria(max_axis_angle=45), **OPTIONS)
            selection = ["--reactive-radius", "3", "--max-axis-angle", "45"]
            for command, target in (("report", out / "r3"), ("extract", out / "e3")):
                extra = ["--max-lag-ps", ".002"] if command == "report" else []
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                    status = main([command, str(out / "cache"), "--output", str(target), "--quiet"] + selection + extra)
                self.assertEqual(status, 0, errors.getvalue())
            times = [r["time_ps"] for r in fresh["frames"]]
            expected = lifetimes.get_pair_lifetime(fresh["pairs_by_frame"], times)
            with (out / "r3" / "pair_lifetimes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(expected))
            self.assertEqual([int(r["detected_frames"]) for r in rows], [e.detected_frames for e in expected])
            with (out / "r3" / "frames.csv").open() as handle:
                counts = [int(r["selected_pairs"]) for r in csv.DictReader(handle)]
            self.assertEqual(counts, [len(p) for p in fresh["pairs_by_frame"]])
            self.assertTrue((out / "r3" / "pair_metric_histograms.csv").exists())
            index = [json.loads(line) for line in (out / "e3" / "index.jsonl").read_text().splitlines()]
            exported = {}
            for record in index:
                exported.setdefault(record["frame"], []).append(record["pair_atom_ids"])
            self.assertEqual([exported.get(f, []) for f in range(9)], fresh["pairs_by_frame"])
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(["report", str(out / "cache"), "--output", str(out / "too_wide"), "--quiet",
                               "--reactive-radius", "6"])
            self.assertEqual(status, 2)
            self.assertIn("cached", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
