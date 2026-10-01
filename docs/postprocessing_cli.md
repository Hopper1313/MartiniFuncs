# Post-simulation CLI

For every-frame pair and pair+shell Extended XYZ collection without statistics
reports, use [the collect command](collect_pairs.md). It preserves source forces,
roles and explicit local/global atom mappings and supports indexed frame workers.

Install the package in the environment used on the compute node:

```console
python -m pip install -e .
martini-postprocess --help
```

`python -m MartiniFuncs.cli` is equivalent and also works from the repository.
No notebook, GUI, UMA weights, or GPU is required. MD forces are read from the
dump; they are not recomputed. This is one-node postprocessing, not MPI.

## Pair lifetimes, pairs and shells without lag statistics

```bash
martini-postprocess run ./ --output ./postprocess --units metal --timestep-ps 0.0005 --shell-radius 3.0 --no-lag-statistics --workers 8 --batch-size 32
```

This processes the entire trajectory, writes frame summaries and pair lifetimes,
and exports pairs and shells. It skips MSD, VACF and force-correlation calculations
entirely, not just their CSV files. Add `--reaction-atom-counts 11 23` to enable
the possible-product screen in `run`. The shell radius is your physical choice.

Report choices are independent:

| Option | Effect |
|---|---|
| `--no-lag-statistics` | Skip all three lag calculations and species lag CSVs |
| `--lag-statistics msd` | Calculate only molecular MSD |
| `--lag-statistics vacf force-acf` | Calculate those correlations only |
| `--no-pair-lifetimes` | Skip the lifetime report |
| `--no-frame-summary` | Skip report `frames.csv`; cache frame metadata remains |
| `--fit-ps A B` | Fit diffusion over lag interval [A,B] ps; requires MSD enabled |
| `--stability` | Add scalar stability summaries |
| `--rdf-bins N` | Add RDFs, independently of lag statistics |
| `--no-export-pairs` | Skip pair and shell exports (`run` only) |
| `--shell-radius R` | Include pair+shell exports with cutoff R Angstrom |

For compatibility, `run` and `report` still default to all available lag statistics,
pair lifetimes and frame summaries. Selection flags affect reports, not cache
construction: molecular arrays are still cached for subsequent analysis.

### Reuse an already completed cache

After updating the package, use these commands to avoid rebuilding existing data:

```bash
martini-postprocess report ./postprocess/cache --output ./postprocess/reports/pair_lifetimes --no-lag-statistics
martini-postprocess extract ./postprocess/cache --output ./postprocess/pairs_restarted --shell-radius 3.0 --workers 8 --batch-size 32
```

Use fresh report/export directories if previous outputs are partial or were made
with different settings/code. `report` and `extract` can read the existing cache
schema; `run` intentionally rejects cache reuse after package source changes.
Extract still requires unchanged source simulation files. Completed caches have
`cache/manifest.json`. No frame-range options are needed for full trajectories.

### Logging

Every CLI stage logs its start and completion (or failure), with timestamps and
elapsed seconds. While a stage has not returned, a background thread emits a
heartbeat every 30 seconds. A heartbeat reports elapsed time, not a completion
percentage or proof of CPU activity. Detailed logs additionally identify:

- The selected report quantities and cache loading.
- Species, observable and completed lag counts during correlations/MSD.
- Completed frame counts during lifetime tracking, then encounter sorting/writing.
- The scalar or species pair being processed for stability/RDF calculations.
- Completed frame chunks during analysis and pair/shell exports.

Output is flushed for Slurm logs. `--quiet` disables progress/heartbeat messages.
Python APIs accept `progress=callback`; the CLI supplies the timed stage wrapper.

## Analyze a finished simulation

```console
martini-postprocess run /scratch/my_run --output /scratch/my_run_analysis --units metal --timestep-ps 0.0005 --temperature-dof 3N-3 --workers 2
```

The run directory supplies the input, dump, and data file through shared discovery.
Use `--input-file`, `--dump-file`, or `--data-file` when several files match.
Units and timestep are mandatory; temperature DOF is explicit and optional.
`--type-symbols types.json` supplies mappings when required by the dump format.

For CPD/MAL reaction screening add `--reaction-atom-counts 11 23`. This rebuilds
connectivity on every source frame, before striding, and stops on a 34-atom
connected component rather than continuing fixed-species statistics. It is an
optional, distance-based possible-product screen, not proof of reaction. It runs
inside the frame workers when parallel processing is enabled.
Notebook 08 requires a cache with this screen (or a fresh raw-data analysis).
See [reaction screening and diffusion](diffusion_assessment.md#possible-reactions).

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
    analysis.json             # metadata and notices (schema 2: no per-frame data)
    frames.npz                # per-frame records, one array per column
    frames.csv                # the same scalar history, human-readable
    molecules.json            # formula and persistent atom IDs for each molecule
    pair_*.npy                # pairs as molecule indices + periodic shifts
    frame_offsets.npy         # dump byte offsets, written by the first export
    *.npy                     # molecular COM, velocity, force, box and cell arrays
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
RDF tables. They are opt-in; scalar ACFs use an FFT (O(N log N) in frame count)
and RDF costs quadratic time in molecule count per frame. They do
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
threads are limited to one per process in the template. Workers now parse and
prepare indexed frame chunks and construct pair/shell exports. Benchmark worker
counts: shared-storage bandwidth, ordered writing and serial reports limit scaling.
See [parallel processing](parallel_postprocessing.md).

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
