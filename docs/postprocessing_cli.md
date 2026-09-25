# Post-simulation CLI

Install the package in the environment used on the compute node:

```console
python -m pip install -e .
martini-postprocess --help
```

`python -m MartiniFuncs.cli` is equivalent and also works from the repository.
No notebook, GUI, UMA weights, or GPU is required. MD forces are read from the
dump; they are not recomputed. This is one-node postprocessing, not MPI.

## Analyze a finished simulation

```console
martini-postprocess run /scratch/my_run --output /scratch/my_run_analysis --units metal --timestep-ps 0.0005 --temperature-dof 3N-3 --workers 2
```

The run directory supplies the input, dump, and data file through shared discovery.
Use `--input-file`, `--dump-file`, or `--data-file` when several files match.
Units and timestep are mandatory; temperature DOF is explicit and optional.
`--type-symbols types.json` supplies mappings when required by the dump format.

By default, this command:

1. Streams all frames, computes molecular arrays/scalars, and identifies accepted
   CPD/dienophile pairs using the existing 3.25 Angstrom center-distance screen.
   `PairCriteria()` checks valid site geometry but imposes no angle thresholds.
2. Saves a reusable molecular-data cache.
3. Reports the entire cached trajectory, up to 100 ps lag.
4. Re-reads the trajectory once to export **all cached accepted pair encounters**,
   geometries and stored atomic forces. Explicit frame bounds apply to both
   the report and these exports; without bounds, all cached frames are used.

`--max-axis-angle 45`, `--max-plane-angle`, `--max-approach-angle`,
`--max-forming-distance`, and `--max-distance-imbalance` enable optional thresholds.
Angles are degrees; distances are Angstrom. These are screening choices, not TS
predictions. Automatic pair chemistry is specific to CPD/the supported dienophile;
use **`--no-pairs`** for generic molecular statistics on other systems.

Add `--shell-radius 3.0` to also export whole-molecule environments around each
pair. This is an atom-to-pair cutoff; completed molecules can extend beyond it.
Use `--no-export-pairs` when only statistics and pair lifetimes are needed: this
avoids the second pass and potentially large structure files. Exports require
force columns. Missing velocities/forces otherwise disable only their statistics.

```text
my_run_analysis/
  cache/
    manifest.json             # settings, source fingerprints, code/version, units
    analysis.json             # frame records, pair identities/shifts, notices
    frames.csv                # scalar history across all retained frames
    molecules.json            # formula and persistent atom IDs for each molecule
    *.npy                     # molecular COM, velocity, force and box arrays
  reports/default/
    manifest.json             # selected frame range, actual times, and species/file mapping
    frames.csv
    species_000_lags.csv       # lag, origin counts, MSD, VACF, force ACF when available
    pair_lifetimes.csv
  pairs/
    manifest.json
    pairs.extxyz
    index.jsonl               # structure ordinal, frame/time, IDs, image shift, geometry
    clusters.extxyz           # only when a shell radius was requested
```

There is one structure per accepted pair **per retained frame**, not one per
lifetime. The index includes persistent forming bonds and local/frame-local
indices. `structure_index` is zero-based and addresses the same record in both
Extended XYZ files. Files may be empty when no pairs are accepted.

Positions are Angstrom and source forces eV/Angstrom. Exported `velocities` retain
ASE internal units; they are not `momenta`. Report/cache velocities are Angstrom/ps.
Use `ase.io.iread(..., format="extxyz")` to stream exported structures.

## Select frames without rereading the dump

All three commands use `--startframe` (inclusive) and `--endframe` (exclusive),
with zero-based **source dump frame indices**. `[1500, 2000)` selects frames 1500
through 1999, not MD steps 1500 through 1999. Without either flag, the entire
available trajectory is selected. End bounds cannot exceed the number of source
frames in the cache; `None` in Python, or omission in the CLI, uses the end.

These are source indices even after striding: if `--frame-stride 2`, selecting
`--startframe 3 --endframe 9` keeps frames 4, 6, 8. Invalid/empty ranges are errors;
lag statistics require at least three retained frames. `--max-frames` limits the
available source range for previews. Lag and fit arguments still use ps.

`run` builds/reuses the full cache allowed by stride and preview settings, then
applies the frame selection to reports and exports. Selected exports are named
`pairs_frames_START_END` (or `pairs_frames_START_end` for an open end); a default
full export remains `pairs/`. This preserves reusable source data and avoids
confusing a partial export with the full trajectory.

```console
martini-postprocess report /scratch/my_run_analysis/cache --output /scratch/my_run_analysis/reports/frames_1500_2000 --startframe 1500 --endframe 2000 --max-lag-ps 50
martini-postprocess report /scratch/my_run_analysis/cache --output /scratch/my_run_analysis/reports/late_fit --startframe 1500 --endframe 2000 --max-lag-ps 100 --fit-ps 25 75
```

`report` needs only the cache, even if the simulation files are offline. It
memory-maps NPY arrays with pickle disabled. Changing the selected source-frame window, maximum
lag, fit interval, or lifetime gap policy does not rerun pair detection.
Lifetimes are recomputed within the selected window, including boundary censoring.
`--fit-ps` writes `diffusion.csv`; no fit interval is guessed.

`--stability` adds scalar ACF/batch summaries; `--rdf-bins 60` adds species COM
RDF tables. They are opt-in because scalar ACF currently costs quadratic time in
frame count and RDF costs quadratic time in molecule count per frame. They do
not determine that a simulation is equilibrated. Output tables can be plotted
locally with Pandas/Matplotlib; the CLI does not require plotting dependencies.

Python access uses the same package APIs:

```python
from MartiniFuncs.postprocessing import load_analysis_cache
from MartiniFuncs.trajectory_analysis import molecular_window_statistics
result, provenance = load_analysis_cache("/scratch/my_run_analysis/cache")
window = molecular_window_statistics(result, startframe=1500, endframe=2000, max_lag_ps=50)
```

Changing pair criteria, bond scale, units, timestep, or frame sampling requires a
new cache/output directory. A stricter or looser criterion can change lifetimes;
this implementation does not silently filter previously filtered candidates.
An identical `run` command reuses a matching completed cache. For another report
through `run`, supply a new `--report-name`. `report` is the simpler route.

## Export another environment/window

```console
martini-postprocess extract /scratch/my_run_analysis/cache --output /scratch/my_run_analysis/shell_5A_late --startframe 1750 --endframe 2000 --shell-radius 5
```

This streams the source dump again, keeping only cached accepted pairs in the
requested interval. New topology is checked against cached pair memberships.
It does not recompute the statistical analysis or read model weights. Relative
or absolute trajectories are not relocated automatically: extraction uses the
absolute source paths recorded in the cache.

## Slurm

Adapt the account/partition, memory/time limits and Python environment in
[`postprocess.slurm`](../examples/slurm/postprocess.slurm) to your site. Submit
after the MD job succeeds:

```bash
sbatch --dependency=afterok:123456 examples/slurm/postprocess.slurm /scratch/my_run /scratch/my_run_analysis metal 0.0005 --temperature-dof 3N-3
```

Replace `123456` with your actual MD job ID. This uses Slurm's documented
[`afterok` dependency](https://slurm.schedmd.com/sbatch.html), one task and several
CPUs per task. Workers cannot exceed `SLURM_CPUS_PER_TASK`; numerical-library
threads are limited to one per process in the template. Benchmark worker counts:
more workers can be slower, and reading/unwrapping/exports remain sequential.

For many independent simulations, use
[`postprocess_array.slurm`](../examples/slurm/postprocess_array.slurm). Create a
tab-separated `runs.tsv` without a header: simulation directory, unique output
directory, units, timestep in ps. Submit `sbatch --array=1-100%10
examples/slurm/postprocess_array.slurm runs.tsv` for 100 entries with at most ten
running concurrently. Array indexing follows [Slurm's job-array interface](https://slurm.schedmd.com/job_array.html).
Each entry must have its own output directory. The scripts are templates, not
submitted jobs; their resource requests need measurement on your cluster.

## Reproducibility and current limits

- Completed products have a final `manifest.json`. Existing outputs are never
  overwritten; identical complete requests are reused, conflicting/incomplete
  directories raise an error. Choose a fresh product directory after interruption.
  This is restart-by-stage, not a checkpoint inside a partially read trajectory.
- Source checks record path, size and nanosecond modification time, not a full
  trajectory content hash. Files must be closed/stable after MD. Cache reuse
  also checks a package-source SHA256; reports record their current code version.
- Atomic frames are streamed, but the first analysis retains molecular arrays
  and pair lists in memory. Cache reports memory-map arrays but window calculations
  still allocate selected data. Thousands of CPUs do not remove this limitation.
  Use job arrays across simulations and appropriate memory/frame-stride settings.
- `--max-frames` is a preview cap on source frames, not an equilibrated tail.
  Frame selections apply after analysis; defaults select all cached frames. Fixed-topology and TOR sampling/cell
  assumptions remain those of `analyze_run`; source forces are not external-load
  corrections and this CLI does not generate transition states or barriers.
