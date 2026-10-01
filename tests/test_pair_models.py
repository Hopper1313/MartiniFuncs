"""The pair engine is chemistry-independent: a second model runs end to end; criteria validate."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCriteria
from MartiniFuncs.pairs.candidates import candidate_table, select_pairs
from MartiniFuncs.pairs.model import (PairCriteria, PairModel, get_pair_model, pair_model_names,
                                      register_pair_model, threshold)
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.workflows.cli import main
from MartiniFuncs.workflows.postprocess import load_analysis_cache

FIXTURES = Path(__file__).parent / "lammpsio_tests"
OPTIONS = dict(units="metal", timestep_ps=.0005, compute_pairs=True)


@dataclass(frozen=True)
class ToyGeometry:
    centroid_distance: float


@dataclass(frozen=True)
class ToyCriteria(PairCriteria):
    max_centroid_distance: float | None = threshold("centroid_distance", help="placed centroid distance")


class ToyModel(PairModel):
    """Whole-molecule COM screen, one metric: distance between the placed molecules' centroids."""
    name = "toy_whole_molecules"
    roles = ("small", "large")
    role_labels = ("SMALL", "LARGE")
    criteria_class = ToyCriteria
    metric_names = ("centroid_distance",)
    metric_ranges = {"centroid_distance": (0, 10)}

    def classify(self, components, elements):
        return [0 if len(c) == 11 else 1 if len(c) == 23 else None for c in components]

    def sites(self, graph, elements, component, role):
        return None

    def measure(self, first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows):
        return ToyGeometry(float(np.linalg.norm(first_xyz.mean(axis=0) - second_xyz.mean(axis=0))))

    def metrics(self, geometry):
        return {"centroid_distance": geometry.centroid_distance}


register_pair_model(ToyModel())


class PairModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.temp.name) / "sim"
        cls.run_dir.mkdir()
        shutil.copy(FIXTURES / "multiple_frames.lammpstrj", cls.run_dir)
        cls.toy = analyze_run(cls.run_dir, reactive_radius=6.0, pair_model="toy_whole_molecules",
                              pair_criteria=ToyCriteria(), **OPTIONS)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_registry_and_pickling(self):
        self.assertIn("cpd_n4t", pair_model_names())
        self.assertIn("toy_whole_molecules", pair_model_names())
        self.assertIs(get_pair_model(), get_pair_model("cpd_n4t"))
        self.assertIs(pickle.loads(pickle.dumps(get_pair_model("toy_whole_molecules"))),
                      get_pair_model("toy_whole_molecules"))
        with self.assertRaisesRegex(ValueError, "Unknown pair model"):
            get_pair_model("no_such_model")

    def test_criteria_validation_and_reasons(self):
        with self.assertRaisesRegex(ValueError, "max_axis_angle must be at most 90 degrees"):
            CpdN4tCriteria(max_axis_angle=91)
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            ToyCriteria(max_centroid_distance=-1)
        criteria = ToyCriteria(max_centroid_distance=2.0)
        self.assertEqual(criteria.active(), {"centroid_distance": 2.0})
        self.assertEqual(criteria.rejection_reasons({"centroid_distance": 2.0}), ())
        self.assertEqual(criteria.rejection_reasons({"centroid_distance": 2.5}), ("max_centroid_distance: 2.5 > 2",))

    def test_toy_model_runs_end_to_end(self):
        self.assertEqual(self.toy["pair_analysis"]["model"], "toy_whole_molecules")
        candidates = self.toy["pair_candidates"]
        self.assertEqual(candidates["metric_names"], ["centroid_distance"])
        self.assertGreater(len(candidates["molecules"]), 0)
        table = candidate_table(self.toy)
        self.assertEqual(list(table.columns[2:4]), ["small_molecule", "large_molecule"])
        self.assertTrue((table["geometry_ok"] == 1).all())
        # Selecting a centroid threshold equals filtering the candidate table by hand.
        pairs, _ = select_pairs(self.toy, criteria=ToyCriteria(max_centroid_distance=5.0))
        expected = table[table["centroid_distance"] <= 5.0].groupby("frame").size()
        self.assertEqual([len(p) for p in pairs], [int(expected.get(f, 0)) for f in range(len(pairs))])
        # The analysis itself with the same threshold gives the same pairs.
        fresh = analyze_run(self.run_dir, reactive_radius=6.0, pair_model="toy_whole_molecules",
                            pair_criteria=ToyCriteria(max_centroid_distance=5.0), **OPTIONS)
        self.assertEqual(pairs, fresh["pairs_by_frame"])

    def test_toy_backends_and_workers_agree(self):
        reference = analyze_run(self.run_dir, reactive_radius=6.0, pair_model="toy_whole_molecules",
                                pair_criteria=ToyCriteria(), pair_backend="reference", **OPTIONS)
        parallel = analyze_run(self.run_dir, reactive_radius=6.0, pair_model="toy_whole_molecules",
                               pair_criteria=ToyCriteria(), pair_workers=2, pair_batch_size=3, **OPTIONS)
        for other in (reference, parallel):
            self.assertEqual(other["pairs_by_frame"], self.toy["pairs_by_frame"])
            np.testing.assert_allclose(other["pair_candidates"]["metrics"], self.toy["pair_candidates"]["metrics"],
                                       rtol=1e-9, atol=1e-9)

    def test_wrong_criteria_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pair_criteria must be ToyCriteria"):
            analyze_run(self.run_dir, pair_model="toy_whole_molecules", pair_criteria=CpdN4tCriteria(), **OPTIONS)

    def test_cli_uses_the_cache_model_for_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "analysis"
            args = ["run", str(self.run_dir), "--output", str(out), "--units", "metal", "--timestep-ps", ".0005",
                    "--pair-model", "toy_whole_molecules", "--reactive-radius", "6", "--no-export-pairs",
                    "--max-lag-ps", ".002", "--quiet"]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(args)
            self.assertEqual(status, 0, errors.getvalue())
            cached, _ = load_analysis_cache(out / "cache")
            self.assertEqual(cached["pair_candidates"]["model"], "toy_whole_molecules")
            del cached   # release memory-mapped arrays (Windows cannot delete open files)
            report = ["report", str(out / "cache"), "--output", str(out / "r"), "--max-lag-ps", ".002", "--quiet"]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(report + ["--max-centroid-distance", "5"])
            self.assertEqual(status, 0, errors.getvalue())
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(report + ["--max-axis-angle", "45"])
            self.assertEqual(status, 2)
            self.assertIn("has no threshold --max-axis-angle", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
