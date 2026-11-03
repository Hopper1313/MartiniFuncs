"""Benchmark equivalent pair analyses: python -m examples.benchmark_pair_analysis RUN_DIR.

Uses explicit physics settings, bounded source-frame count, and no saved reports.
The reference backend is intentionally slower; start with --max-frames 100.
"""
import argparse
from time import perf_counter

import numpy as np

from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCollection
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.pairs.lifetimes import get_pair_lifetime


def main():
    """Print wall times and verify pairs, shifts, COMs, and lifetime equality."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--units", required=True)
    parser.add_argument("--timestep-ps", type=float, required=True)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-reference", action="store_true",
                        help="Compare cached worker counts only; useful for full-trajectory scaling")
    parser.add_argument("--input-file")
    parser.add_argument("--dump-file")
    parser.add_argument("--data-file")
    args = parser.parse_args()
    reference = None
    configurations = ([] if args.skip_reference else [("reference", 1)]) + [("cached", n) for n in args.workers]
    if args.skip_reference:
        print("Skipping dynamic reference; agreement compares cached worker counts only.", flush=True)
    for backend, workers in configurations:
        start = perf_counter()
        result = analyze_run(args.directory, units=args.units, timestep_ps=args.timestep_ps,
                             input_file=args.input_file, dump_file=args.dump_file, data_file=args.data_file,
                             max_frames=args.max_frames, compute_pairs=True, pair_model="cpd_n4t",
                             pair_collection=CpdN4tCollection(),
                             pair_backend=backend, pair_workers=workers, pair_batch_size=args.batch_size)
        elapsed = perf_counter() - start
        times = [r["time_ps"] for r in result["frames"]]
        if reference is None:
            reference = result
        else:
            identities = lambda frames: [[tuple(tuple(sorted(c)) for c in pair) for pair in pairs] for pairs in frames]
            if identities(result["pairs_by_frame"]) != identities(reference["pairs_by_frame"]):
                raise AssertionError("Pair identities/order differ from reference")
            for key in ("pair_shifts_by_frame", "frames"):
                if result[key] != reference[key]:
                    raise AssertionError(f"{backend}/{workers}: {key} differs from reference")
            np.testing.assert_allclose(result["com_angstrom"], reference["com_angstrom"])
            if get_pair_lifetime(result["pairs_by_frame"], times) != get_pair_lifetime(reference["pairs_by_frame"], times):
                raise AssertionError("Pair lifetimes differ from reference")
        print(f"{backend:9s} workers={workers}: {elapsed:.3f} s, "
              f"{len(times)} frames, {elapsed/len(times):.4f} s/frame; agreement OK", flush=True)


if __name__ == "__main__":
    main()
