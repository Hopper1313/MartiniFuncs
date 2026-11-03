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

from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCollection, CpdN4tCriteria
from MartiniFuncs.io.lammps import iter_lammps_frames
from MartiniFuncs.pairs.assessment import screen_frame
from MartiniFuncs.pairs.candidates import candidate_table, select_pairs
from MartiniFuncs.pairs.model import (PairCollection, PairCriteria, PairModel, get_collection, get_pair_model,
                                      pair_model_names, register_pair_model, threshold)
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.workflows.cli import main
from MartiniFuncs.workflows.postprocess import load_analysis_cache

FIXTURES = Path(__file__).parent / "lammpsio_tests"
OPTIONS = dict(units="metal", timestep_ps=.0005, compute_pairs=True)
TOY = dict(pair_model="toy_whole_molecules", pair_collection=PairCollection(reactive_radius=6.0))


@dataclass(frozen=True)
class ToyGeometry:
    centroid_distance: float


@dataclass(frozen=True)
class ToyCriteria(PairCriteria):
    max_centroid_distance: float | None = threshold("centroid_distance", help="placed centroid distance")


class ToyModel(PairModel):
    """Whole-molecule COM screen, one metric: distance between the placed molecules' centroids.

    It keeps the base PairCollection, which has no default radius: analyses must pass one.
    """
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
        cls.toy = analyze_run(cls.run_dir, **TOY, **OPTIONS)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_registry_and_pickling(self):
        self.assertIn("cpd_n4t", pair_model_names())
        self.assertIn("toy_whole_molecules", pair_model_names())
        with self.assertRaisesRegex(ValueError, "No pair model given"):     # there is no default model
            get_pair_model(None)
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
        # Every model's criteria inherit the screen threshold and the geometry requirement.
        self.assertEqual(ToyCriteria.thresholds(), {"max_screen_distance": "screen_distance",
                                                    "max_centroid_distance": "centroid_distance"})
        self.assertFalse(ToyCriteria().requires_geometry())
        self.assertFalse(ToyCriteria(max_screen_distance=3.).requires_geometry())
        self.assertTrue(ToyCriteria(require_geometry=True).requires_geometry())
        self.assertTrue(criteria.requires_geometry())                 # a geometric threshold implies it
        self.assertEqual(criteria.rejection_reasons({}, geometry_ok=False, error="no sites"), ("no sites",))
        self.assertEqual(ToyCriteria(max_screen_distance=3.).rejection_reasons(
            {"screen_distance": 2.}, geometry_ok=False, error="no sites"), ())
        with self.assertRaisesRegex(ValueError, "needs the 'screen_distance' value"):
            ToyCriteria(max_screen_distance=3.).rejection_reasons({"centroid_distance": 1.})
        with self.assertRaisesRegex(ValueError, "require_geometry must be True or False"):
            ToyCriteria(require_geometry="yes")

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
        # Per-frame selection (screen_frame) with the same collection and criteria gives the same pairs.
        per_frame = [screen_frame(frame.atoms, "toy_whole_molecules", collection=TOY["pair_collection"],
                                  criteria=ToyCriteria(max_centroid_distance=5.0))[0]
                     for frame in iter_lammps_frames(self.run_dir / "multiple_frames.lammpstrj", units="metal")]
        self.assertEqual([[tuple(map(sorted, p)) for p in f] for f in pairs],
                         [[tuple(map(sorted, p)) for p in f] for f in per_frame])

    def test_toy_backends_and_workers_agree(self):
        reference = analyze_run(self.run_dir, pair_backend="reference", **TOY, **OPTIONS)
        parallel = analyze_run(self.run_dir, pair_workers=2, pair_batch_size=3, **TOY, **OPTIONS)
        for other in (reference, parallel):
            self.assertEqual(other["pairs_by_frame"], self.toy["pairs_by_frame"])
            np.testing.assert_allclose(other["pair_candidates"]["metrics"], self.toy["pair_candidates"]["metrics"],
                                       rtol=1e-9, atol=1e-9)

    def test_wrong_collection_or_criteria_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pair_collection must be CpdN4tCollection"):
            analyze_run(self.run_dir, pair_model="cpd_n4t", pair_collection=PairCollection(reactive_radius=4.0),
                        **OPTIONS)
        with self.assertRaisesRegex(ValueError, "criteria must be ToyCriteria"):
            select_pairs(self.toy, CpdN4tCriteria(max_axis_angle=45))
        generic = select_pairs(self.toy, PairCriteria(max_screen_distance=4.0))[0]     # any model's screen
        expected = candidate_table(self.toy)
        expected = expected[expected["screen_distance"] <= 4.0].groupby("frame").size()
        self.assertEqual([len(p) for p in generic], [int(expected.get(f, 0)) for f in range(len(generic))])

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

    def test_pair_analysis_requires_a_named_model(self):
        """There is no default model: pair analyses name one; --no-pairs needs none."""
        with self.assertRaisesRegex(ValueError, "No pair model given"):
            analyze_run(self.run_dir, **OPTIONS)
        with tempfile.TemporaryDirectory() as tmp:
            base = ["run", str(self.run_dir), "--units", "metal", "--timestep-ps", ".0005", "--no-export-pairs",
                    "--max-lag-ps", ".002", "--quiet"]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(base + ["--output", str(Path(tmp) / "unnamed")])
            self.assertEqual(status, 2)
            self.assertIn("--pair-model is required", errors.getvalue())
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
                status = main(base + ["--output", str(Path(tmp) / "no_pairs"), "--no-pairs"])
            self.assertEqual(status, 0, errors.getvalue())

    def test_collection_default_belongs_to_the_model(self):
        from MartiniFuncs.workflows.postprocess import _analysis_configuration
        self.assertIsInstance(get_collection("cpd_n4t"), CpdN4tCollection)
        self.assertEqual(get_collection("cpd_n4t").reactive_radius, 3.25)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            CpdN4tCollection(reactive_radius=-1.)
        # The toy model keeps the base PairCollection, which has no default radius.
        with self.assertRaisesRegex(ValueError, "has no default collection"):
            get_collection("toy_whole_molecules")
        with self.assertRaisesRegex(ValueError, "has no default collection"):
            analyze_run(self.run_dir, pair_model="toy_whole_molecules", **OPTIONS)
        # Omitted and explicit default collection record the same cache configuration; the model is required.
        with self.assertRaisesRegex(ValueError, "No pair model given"):
            _analysis_configuration(dict(units="metal", timestep_ps=.0005, compute_pairs=True))
        omitted = _analysis_configuration(dict(units="metal", timestep_ps=.0005, compute_pairs=True,
                                               pair_model="cpd_n4t"))
        explicit = _analysis_configuration(dict(units="metal", timestep_ps=.0005, compute_pairs=True,
                                                pair_collection=CpdN4tCollection(reactive_radius=3.25),
                                                pair_model=get_pair_model("cpd_n4t")))
        self.assertEqual(omitted, explicit)
        self.assertEqual((omitted["pair_collection"], omitted["pair_model"]), ({"reactive_radius": 3.25}, "cpd_n4t"))
        self.assertIsNone(_analysis_configuration(dict(units="metal", timestep_ps=.0005))["pair_collection"])
        # The default collection is the model's in the analysis itself.
        default = analyze_run(self.run_dir, pair_model="cpd_n4t", **OPTIONS)
        stated = analyze_run(self.run_dir, pair_model="cpd_n4t", pair_collection=CpdN4tCollection(reactive_radius=3.25),
                             **OPTIONS)
        self.assertEqual(default["pair_candidates"]["reactive_radius"], 3.25)
        self.assertEqual(default["pair_analysis"]["collection"], {"reactive_radius": 3.25})
        self.assertEqual(default["pairs_by_frame"], stated["pairs_by_frame"])

if __name__ == "__main__":
    unittest.main()
