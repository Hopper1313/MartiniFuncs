"""CLI on a reacting trajectory: reaction events are recorded and never terminate the job."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import csv
import json
import tempfile
import unittest
from pathlib import Path

from MartiniFuncs.workflows.cli import main
from MartiniFuncs.workflows.postprocess import load_analysis_cache
import tests.test_dynamics as dynamics_tests


def call(args):
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
        status = main([str(a) for a in args])
    return status, errors.getvalue()


class ReactionCliTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run_dir = self.root / "simulation"
        self.run_dir.mkdir()
        # CPD bonded to a dienophile from frame 4 on, plus a one-frame contact at frame 2.
        dynamics_tests.TopologySegmentTests().make_reacting_dump(self.run_dir)
        self.common = ["--units", "metal", "--timestep-ps", ".0005", "--quiet", "--reaction-persistence", "3"]

    def test_run_records_events_and_exports_everything(self):
        out = self.root / "run"
        status, errors = call(["run", self.run_dir, "--output", out, "--shell-radius", "2.5",
                               "--workers", "2", "--batch-size", "2", "--max-lag-ps", ".002"] + self.common)
        self.assertEqual(status, 0, errors)
        result, manifest = load_analysis_cache(out / "cache")
        self.assertEqual([e["frame"] for e in result["topology_events"]], [4])     # flicker at 2 ignored
        self.assertEqual(result["topology_events"][0]["formulas"], ["C16H15NO2"])
        self.assertEqual(len(result["frames"]), 9)                                 # analysis kept going
        with (out / "reports/default/topology_events.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(r["frame"]) for r in rows], [4])
        export = json.loads((out / "pairs/manifest.json").read_text())
        self.assertEqual(export["frames_processed"], 9)
        index = [json.loads(line) for line in (out / "pairs/index.jsonl").read_text().splitlines()]
        self.assertEqual(sum(r["membership_changed"] for r in index), export["structures_with_changed_membership"])
        self.assertGreaterEqual(export["structures_with_changed_membership"], 1)   # flagged, not aborted
        self.assertTrue(all(not r["membership_changed"] for r in index if r["frame"] < 2))

    def test_stop_ends_cleanly_before_the_event(self):
        out = self.root / "stop"
        status, errors = call(["run", self.run_dir, "--output", out, "--no-export-pairs", "--max-lag-ps", ".002",
                               "--reaction-events", "stop"] + self.common)
        self.assertEqual(status, 0, errors)
        result, _ = load_analysis_cache(out / "cache")
        self.assertEqual([r["frame"] for r in result["frames"]], [0, 1, 2, 3])
        self.assertEqual(result["topology_event"]["frame"], 4)

    def test_collect_records_by_default_and_legacy_screen_still_aborts(self):
        status, errors = call(["collect", self.run_dir, "--output", self.root / "collect", "--shell-radius", "2.5",
                               "--workers", "2", "--batch-size", "3"] + self.common)
        self.assertEqual(status, 0, errors)
        result, _ = load_analysis_cache(self.root / "collect" / "cache")
        self.assertEqual([e["frame"] for e in result["topology_events"]], [4])
        status, errors = call(["run", self.run_dir, "--output", self.root / "legacy", "--no-export-pairs",
                               "--reaction-atom-counts", "11", "23"] + self.common)
        self.assertEqual(status, 2)
        self.assertIn("Possible reaction", errors)


if __name__ == "__main__":
    unittest.main()
