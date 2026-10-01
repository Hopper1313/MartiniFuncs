"""Schema-2 cache layout, schema-1 compatibility, saved frame offsets and the offset scanner."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from array import array

import numpy as np

from MartiniFuncs import frame_parallel
from MartiniFuncs.postprocessing import (SCHEMA_VERSION, cache_analysis, cached_frame_offsets, export_pairs,
                                         load_analysis_cache, write_report)

FIXTURES = Path(__file__).parent / "lammpsio_tests"


def line_scan_offsets(path):
    """The original line-by-line implementation, kept as the reference."""
    offsets = array("Q")
    with open(path, "rb") as handle:
        position = 0
        for line in iter(handle.readline, b""):
            if line.strip() == b"ITEM: TIMESTEP":
                offsets.append(position)
            position += len(line)
        offsets.append(position)
    return np.asarray(offsets, dtype=np.uint64)


class CacheFormatTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "simulation"
        self.run.mkdir()
        shutil.copyfile(FIXTURES / "multiple_frames.lammpstrj", self.run / "trajectory.lammpstrj")
        self.cache = self.root / "cache"
        self.result, self.manifest = cache_analysis(self.run, self.cache, units="metal", timestep_ps=.0005,
                                                    progress=None)

    def test_round_trip_is_exact_and_compact(self):
        self.assertEqual(self.manifest["schema_version"], SCHEMA_VERSION)
        loaded, _ = load_analysis_cache(self.cache)
        self.assertEqual(loaded["frames"], self.result["frames"])
        self.assertEqual(list(loaded["pairs_by_frame"]), self.result["pairs_by_frame"])
        self.assertEqual(list(loaded["pair_shifts_by_frame"]),
                         [[list(map(int, s)) for s in frame] for frame in self.result["pair_shifts_by_frame"]])
        pairs = loaded["pairs_by_frame"]
        self.assertEqual(pairs[-1], self.result["pairs_by_frame"][-1])
        self.assertEqual(pairs[2:5], self.result["pairs_by_frame"][2:5])
        with self.assertRaises(IndexError):
            pairs[len(pairs)]
        metadata = json.loads((self.cache / "analysis.json").read_text())
        self.assertNotIn("pairs_by_frame", metadata)
        self.assertNotIn("frames", metadata)
        self.assertEqual(metadata["pair_storage"], "molecule_index")

    def test_schema_one_caches_still_load_and_report(self):
        loaded, manifest = load_analysis_cache(self.cache)
        old = self.root / "schema1"
        old.mkdir()
        for key in manifest["arrays"]:
            shutil.copyfile(self.cache / f"{key}.npy", old / f"{key}.npy")
        metadata = {k: v for k, v in loaded.items() if not isinstance(v, np.ndarray) and v is not None or k in
                    ("pairs_by_frame", "pair_shifts_by_frame")}
        metadata["pairs_by_frame"] = list(loaded["pairs_by_frame"])
        metadata["pair_shifts_by_frame"] = list(loaded["pair_shifts_by_frame"])
        metadata = {k: v for k, v in metadata.items() if k not in ("species",)}
        (old / "analysis.json").write_text(json.dumps(metadata))
        (old / "manifest.json").write_text(json.dumps(dict(manifest, schema_version=1)))
        legacy, _ = load_analysis_cache(old)
        self.assertEqual(legacy["frames"], loaded["frames"])
        self.assertEqual(legacy["pairs_by_frame"], list(loaded["pairs_by_frame"]))
        a = write_report(self.cache, self.root / "new_report", max_lag_ps=1.)
        b = write_report(old, self.root / "old_report", max_lag_ps=1.)
        for name in ("pair_lifetimes.csv", "species_000_lags.csv", "frames.csv"):
            self.assertEqual((a / name).read_text(), (b / name).read_text())

    def test_export_saves_offsets_and_reads_only_selected_frames(self):
        export_pairs(self.cache, self.root / "every_other", startframe=1, endframe=8, progress=None)
        offsets = np.load(self.cache / "frame_offsets.npy")
        np.testing.assert_array_equal(offsets, line_scan_offsets(self.run / "trajectory.lammpstrj"))
        np.testing.assert_array_equal(cached_frame_offsets(self.cache, self.result), offsets)
        rows = (self.root / "every_other/frames.csv").read_text().splitlines()[1:]
        self.assertEqual([int(r.split(",")[0]) for r in rows], list(range(1, 8)))

    def test_offset_scanner_matches_line_scan_at_every_block_size(self):
        sources = [FIXTURES / "multiple_frames.lammpstrj", FIXTURES / "frame_0.lammpstrj"]
        texts = [p.read_bytes() for p in sources]
        texts += [texts[0].replace(b"\n", b"\r\n"), b"\n\n" + texts[1],
                  b"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n0\nITEM: TIMESTEP\n5\nITEM: NUMBER OF ATOMS\n0\n"]
        original = frame_parallel._SCAN_BLOCK
        self.addCleanup(setattr, frame_parallel, "_SCAN_BLOCK", original)
        for k, text in enumerate(texts):
            path = self.root / f"case{k}.lammpstrj"
            path.write_bytes(text)
            for block in (original, 15, 16, 37, 4096):
                frame_parallel._SCAN_BLOCK = block
                np.testing.assert_array_equal(frame_parallel.frame_offsets(path), line_scan_offsets(path))
        (self.root / "bad.lammpstrj").write_bytes(b"garbage\nITEM: TIMESTEP\n")
        with self.assertRaisesRegex(ValueError, "expected ITEM: TIMESTEP"):
            frame_parallel.frame_offsets(self.root / "bad.lammpstrj")


if __name__ == "__main__":
    unittest.main()
