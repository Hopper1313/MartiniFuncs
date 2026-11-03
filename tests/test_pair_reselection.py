"""Two-stage pairs: selecting from a wide collection equals a narrower collection and the same selection."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCollection, CpdN4tCriteria
from MartiniFuncs.io.lammps import iter_lammps_frames
from MartiniFuncs.pairs import lifetimes
from MartiniFuncs.pairs.assessment import screen_frame
from MartiniFuncs.pairs.candidates import candidate_table, select_pairs
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.workflows.cli import main
from MartiniFuncs.workflows.postprocess import load_analysis_cache

FIXTURES = Path(__file__).parent / "lammpsio_tests"
OPTIONS = dict(units="metal", timestep_ps=.0005, compute_pairs=True, pair_model="cpd_n4t")


def collect(run_dir, radius, **options):
    return analyze_run(run_dir, pair_collection=CpdN4tCollection(reactive_radius=radius), **OPTIONS, **options)


def identities(frames):
    """Pair identity per frame, independent of atom traversal order within a molecule."""
    return [[tuple(tuple(sorted(c)) for c in pair) for pair in pairs] for pairs in frames]


class PairReselectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temp.name) / "sim"
        cls.run_dir.mkdir()
        shutil.copy(FIXTURES / "multiple_frames.lammpstrj", cls.run_dir)
        cls.wide = collect(cls.run_dir, 5.0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assertSamePairs(self, selected, reference):
        pairs, shifts = selected
        reference_pairs, reference_shifts = reference
        self.assertEqual(pairs, reference_pairs)
        self.assertEqual(shifts, [[list(map(int, s)) for s in frame] for frame in reference_shifts])

    def test_no_criteria_returns_every_collected_pair(self):
        pairs, shifts = select_pairs(self.wide)
        self.assertIs(pairs, self.wide["pairs_by_frame"])
        candidates = self.wide["pair_candidates"]
        self.assertEqual(sum(map(len, pairs)), len(candidates["molecules"]))      # nothing filtered at collection
        self.assertEqual(self.wide["pair_analysis"]["collection"], {"reactive_radius": 5.0})
        self.assertGreater(sum(map(len, pairs)), sum(map(len, analyze_run(self.run_dir, **OPTIONS)["pairs_by_frame"])))

    def test_smaller_screen_distance_equals_smaller_collection(self):
        for radius in (2.5, 3.25, 4.0):
            narrow = collect(self.run_dir, radius)
            self.assertSamePairs(select_pairs(self.wide, CpdN4tCriteria(max_screen_distance=radius)),
                                 select_pairs(narrow))
            geometry = CpdN4tCriteria(require_geometry=True)
            self.assertSamePairs(select_pairs(self.wide, CpdN4tCriteria(max_screen_distance=radius, require_geometry=True)),
                                 select_pairs(narrow, geometry))

    def test_criteria_equal_on_wide_and_narrow_collections(self):
        narrow = collect(self.run_dir, 3.25)
        for criteria in (CpdN4tCriteria(max_axis_angle=30), CpdN4tCriteria(max_forming_distance=4.5, max_plane_angle=40)):
            wide_selection = select_pairs(self.wide, replace(criteria, max_screen_distance=3.25))
            self.assertSamePairs(wide_selection, select_pairs(narrow, criteria))
            self.assertLessEqual(sum(map(len, wide_selection[0])), sum(map(len, select_pairs(self.wide, criteria)[0])))

    def test_cache_selection_matches_per_frame_selection(self):
        """select_pairs on the cache and screen_frame on each frame apply the same rules."""
        criteria = CpdN4tCriteria(max_screen_distance=4.0, max_axis_angle=45)
        selected = select_pairs(self.wide, criteria)[0]
        per_frame = [screen_frame(frame.atoms, "cpd_n4t", collection=CpdN4tCollection(reactive_radius=5.0),
                                  criteria=criteria)[0]
                     for frame in iter_lammps_frames(self.run_dir / "multiple_frames.lammpstrj", units="metal")]
        self.assertEqual(identities(selected), identities(per_frame))
        self.assertGreater(sum(map(len, selected)), 0)

    def test_reference_backend_candidates_match_cached_backend(self):
        reference = collect(self.run_dir, 5.0, pair_backend="reference")
        a, b = candidate_table(self.wide), candidate_table(reference)
        self.assertEqual(len(a), len(b))
        for column in ("frame", "cpd_molecule", "dienophile_molecule", "shift_x", "shift_y", "shift_z", "geometry_ok"):
            self.assertEqual(a[column].tolist(), b[column].tolist(), column)
        for column in ("screen_distance", "max_forming_distance", "axis_angle", "plane_angle", "approach_angle"):
            self.assertLess((a[column] - b[column]).abs().max(), 1e-9, column)

    def test_screen_distance_above_collection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at most 5 Angstrom"):
            select_pairs(self.wide, CpdN4tCriteria(max_screen_distance=5.5))

    def test_cli_report_and_extract_select_after_the_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "analysis"
            args = ["run", self.run_dir, "--output", out, "--units", "metal", "--timestep-ps", ".0005",
                    "--pair-model", "cpd_n4t", "--reactive-radius", "5", "--no-export-pairs", "--max-lag-ps", ".002", "--quiet"]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main([str(a) for a in args])
            self.assertEqual(status, 0, errors.getvalue())
            cached, manifest = load_analysis_cache(out / "cache")
            self.assertEqual(cached["pair_candidates"]["reactive_radius"], 5.0)
            self.assertEqual(manifest["configuration"]["pair_collection"], {"reactive_radius": 5.0})
            self.assertNotIn("pair_criteria", manifest["configuration"])
            del cached   # release memory-mapped arrays (Windows cannot delete open files)
            expected_pairs = select_pairs(collect(self.run_dir, 3.0), CpdN4tCriteria(max_axis_angle=45))[0]
            selection = ["--max-screen-distance", "3", "--max-axis-angle", "45"]
            for command, target in (("report", out / "r3"), ("extract", out / "e3")):
                extra = ["--max-lag-ps", ".002"] if command == "report" else []
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                    status = main([command, str(out / "cache"), "--output", str(target), "--quiet"] + selection + extra)
                self.assertEqual(status, 0, errors.getvalue())
            times = [r["time_ps"] for r in self.wide["frames"]]
            expected = lifetimes.get_pair_lifetime(expected_pairs, times)
            with (out / "r3" / "pair_lifetimes.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(expected))
            self.assertEqual([int(r["detected_frames"]) for r in rows], [e.detected_frames for e in expected])
            with (out / "r3" / "frames.csv").open() as handle:
                counts = [int(r["selected_pairs"]) for r in csv.DictReader(handle)]
            self.assertEqual(counts, [len(p) for p in expected_pairs])
            self.assertTrue((out / "r3" / "pair_metric_histograms.csv").exists())
            index = [json.loads(line) for line in (out / "e3" / "index.jsonl").read_text().splitlines()]
            exported = {}
            for record in index:
                exported.setdefault(record["frame"], []).append(record["pair_atom_ids"])
            self.assertEqual([exported.get(f, []) for f in range(9)], expected_pairs)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(["report", str(out / "cache"), "--output", str(out / "too_wide"), "--quiet",
                               "--max-screen-distance", "6"])
            self.assertEqual(status, 2)
            self.assertIn("at most 5", errors.getvalue())

    def test_cli_run_selection_does_not_change_the_cache(self):
        """Selection flags on run shape its report only; a second run with other thresholds reuses the cache."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "analysis"
            base = ["run", str(self.run_dir), "--output", str(out), "--units", "metal", "--timestep-ps", ".0005",
                    "--pair-model", "cpd_n4t", "--reactive-radius", "5", "--no-export-pairs", "--max-lag-ps", ".002"]
            reports, logs = {}, {}
            for name, flags in (("axis45", ["--max-axis-angle", "45"]), ("all", [])):
                with redirect_stdout(StringIO()) as output, redirect_stderr(StringIO()) as errors:
                    status = main(base + ["--report-name", name] + flags)
                self.assertEqual(status, 0, errors.getvalue())
                logs[name] = output.getvalue()
                with (out / "reports" / name / "pair_lifetimes.csv").open() as handle:
                    reports[name] = sum(int(r["detected_frames"]) for r in csv.DictReader(handle))
            self.assertNotIn("Reusing completed cache", logs["axis45"])
            self.assertIn("Reusing completed cache", logs["all"])          # different thresholds, same cache
            self.assertEqual(reports["all"], sum(map(len, self.wide["pairs_by_frame"])))
            self.assertEqual(reports["axis45"], sum(map(len, select_pairs(self.wide, CpdN4tCriteria(max_axis_angle=45))[0])))
            self.assertLess(reports["axis45"], reports["all"])
            manifest = json.loads((out / "cache" / "manifest.json").read_text())
            self.assertEqual(manifest["configuration"]["pair_collection"], {"reactive_radius": 5.0})


if __name__ == "__main__":
    unittest.main()
