# Faster trajectory pair analysis

`analyze_run` now defaults to cached pair analysis when `compute_pairs=True`.
It identifies molecules/reactive sites once and reuses the whole-molecule
coordinates already computed for MD statistics. The distance screen, image
selection, angle equations, and threshold logic are shared with the reference
implementation in `chemistry.cpd_n4t.pairs` and `chemistry.cpd_n4t.selection`.

```python
from MartiniFuncs.trajectory.analysis import analyze_run
from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCollection

result = analyze_run(
    RUN_DIR, units="metal", timestep_ps=0.0005,
    compute_pairs=True, pair_model="cpd_n4t", pair_collection=CpdN4tCollection(reactive_radius=3.25),
    pair_backend="cached", pair_workers=1, pair_batch_size=32,
)
```

- Start with **one worker**. Caching removes most pair-analysis work, so process
  startup and array transfer can cost more than they save. Try 2 or 4 workers
  and measure on your actual system.
- `pair_workers > 1` uses spawned processes. Each task contains at most
  `pair_batch_size` retained frames; at most `2 * pair_workers` tasks are in
  flight. Workers receive coordinates/cells/PBC and initial topology, not ASE
  objects, calculators, or the entire trajectory.
- Reading, molecular reconstruction and temporal TOR unwrapping remain sequential.
  Results are collected in frame order; pair lifetimes can span task boundaries.
- `frame_stride` reduces retained-frame pair work but still reads/reconstructs
  every source frame. Selecting a late statistics window still happens after
  reading. No dump splitting, byte-offset index, or parser replacement is added.
- Standalone scripts using multiple workers must call analysis inside
  `if __name__ == "__main__":`. Notebook cells can call the imported function
  directly. Restart the kernel after updating the package.

## Scientific and compatibility limits

Cached analysis assumes **fixed initial connectivity**, consistent with the
existing molecular-statistics model. IDs, elements, masses and existing-bond
reconstruction are checked each frame; this does not detect every new bond.
For per-frame pair topology rebuilding, set `pair_backend="reference"` and
`pair_workers=1`. Neither mode turns the fixed-molecule MSD into a reactive
trajectory analysis.

The cache preserves initial atom order within each molecule. The reference
graph's traversal order can change per frame; molecular identity is the set of
persistent IDs. Candidate ordering and periodic-shift conventions are preserved.
`pair_shifts_by_frame` now accompanies `pairs_by_frame`, and `pair_analysis`
records the backend, worker count, and batch size.

The notebooks expose these as `PAIR_BACKEND`, `PAIR_WORKERS`, and
`PAIR_BATCH_SIZE`. Source-frame selection uses `STARTFRAME`/`ENDFRAME` (inclusive/exclusive); the
default is the full trajectory. Lag and fit intervals remain in ps.

## Reproduce the benchmark

```console
python -m examples.benchmark_pair_analysis RUN_DIR --units metal --timestep-ps 0.0005 --max-frames 100 --workers 1 2
```

This times the reference and cached paths and checks molecular identities,
candidate ordering, image shifts, frame statistics, COMs and pair lifetimes.
Tests separately compare forming bonds, local/frame mappings and angle/distance
measurements, including shuffled sparse IDs and a triclinic cell.
Use filename overrides if discovery is ambiguous. For longer scaling runs,
`--skip-reference` compares cached worker counts only; it is not an independent
check against dynamic topology rebuilding.

On 100 frames of the 1,360-atom `continued_run` example, with `CpdN4tCriteria()`:
reference 23.07 s; cached serial 1.31 s; cached two-process 3.15 s. This includes
reading and process startup. These are local measurements, not guaranteed speedups.

For the complete 2,001-frame trajectory, cached serial took 23.47 s and cached
two-process took 19.88 s. Those full-run results matched each other; the dynamic
reference comparison above covered the first 100 frames, not the full trajectory.
