# Workflow guide: from a LAMMPS trajectory to statistics and structures

This guide is the recommended order of work for a CPD + N4T trajectory, with
copy-paste examples for every step. The short version:

1. **`run` once per trajectory** (expensive, Slurm, many workers). It reads the
   dump, collects every CPD/N4T pair closer than the collection radius (set it to
   the largest radius you will ever select), and writes a reusable **cache**.
2. **Explore the cache in Python** (cheap). Pick the equilibrated frame window
   and the pair thresholds from the data.
3. **`report` as often as you like** (about a minute each). Statistics,
   lifetimes and histograms for any selection and frame window.
4. **`extract` once you know what you need** (reads the dump again). Pair and
   pair+shell structures with forces, for the final selection only.

Everything else (diffusion checks, rotational dynamics, the pair builder,
force validation) hangs off these four steps.

Contents:

- [The commands at a glance](#the-commands-at-a-glance)
- [Where to run each command (not the login node)](#where-to-run-each-command-not-the-login-node)
- [Before you start](#before-you-start)
- [Step 1: `run`, collect once](#step-1-run-collect-once)
- [Step 2: explore the cache in Python](#step-2-explore-the-cache-in-python)
- [Step 3: `report`, statistics for any selection](#step-3-report-statistics-for-any-selection)
- [Step 4: `extract`, structures for the final selection](#step-4-extract-structures-for-the-final-selection)
- [Step 5: `recompute`, isolated forces, F_env and energies](#step-5-recompute-isolated-forces-f_env-and-energies)
- [`collect`: structures without a report](#collect-structures-without-a-report)
- [Designed pairs: the pair builder](#designed-pairs-the-pair-builder)
- [Selection reference](#selection-reference)
- [When do I need a new `run`?](#when-do-i-need-a-new-run)
- [Common errors and what they mean](#common-errors-and-what-they-mean)
- [A complete session, start to finish](#a-complete-session-start-to-finish)

Related references: [CLI details and output schemas](postprocessing_cli.md),
[pair models, collection and criteria](pair_models.md),
[dynamics diagnostics](dynamics_diagnostics.md),
[diffusion assessment](diffusion_assessment.md),
[thermo output](thermo_output.md).

---

## The commands at a glance

All commands are `python -m MartiniFuncs <command> ...` (the installed
`martini-postprocess` command is the same program).

| Command | Reads the dump? | Typical cost | Use it for |
|---|---|---|---|
| `run` | yes, every frame | hours; Slurm job with many workers | once per trajectory: builds the cache, writes a default report |
| `report` | no, cache only | about 1 minute, 1 CPU | statistics for any selection and frame window; repeat freely |
| `extract` | yes, selected frames only | minutes to hours, depending on the selection; `--workers` | `.extxyz` pair and pair+shell structures with forces |
| `collect` | yes, every frame | like `run` + `extract` | structures straight from the dump in one pass, no report; rarely needed |

Measured on the NVT_noshear trajectory (300,001 frames, 1,360 atoms, 43 GB
dump, 3.75 Angstrom collection with about 4.8 million pair detections): a full
`report` took **49 s** on one laptop CPU (cache load and selection 2 s, MSD/VACF/
force ACF for both species 31 s, lifetimes 12 s).

### Two stages for pairs

Pairs are handled in two stages (details in [pair models](pair_models.md)):

| Stage | Class (cpd_n4t) | Set by | Effect |
|---|---|---|---|
| 1. Collection | `CpdN4tCollection(reactive_radius=3.25)` | `run --reactive-radius` | which pairs are measured and **stored in the cache** |
| 2. Selection | `CpdN4tCriteria(...)` | `--max-*` flags, `--geometry-required`, `select_pairs(...)` | which collected pairs a report, export or calculation **uses** |

The collection radius is the CPD center of mass to the dienophile C=C center of
mass distance (nearest periodic image). A cache holds **only** the pairs below
its collection radius: a 3.75 Angstrom cache contains every pair closer than
3.75 Angstrom and nothing beyond. From it you can select any radius up to 3.75
(`--max-screen-distance 3.25`, `3.5`, `3.75`), but never a larger one; a 4.0
Angstrom selection needs a new `run` at 4.0 Angstrom or more. Selection never
changes the cache, so every other threshold can be tried without rereading the
dump.

---

## Where to run each command (not the login node)

Login nodes are shared by everyone on the cluster and are only for editing,
installing, and submitting jobs. **Nothing in this guide that reads a dump, loads
a cache or computes statistics belongs on the login node**, not even the
one-minute `report`: it memory-maps arrays of 0.5-1 GB each and keeps a CPU busy.
Login-node jobs are typically throttled or killed, and they slow the node down
for everyone.

| Task | Where | How |
|---|---|---|
| `git fetch/merge`, `pip install -e`, editing scripts, `--help`, `sbatch`, `squeue` | login node | directly |
| `run`, `collect` | compute node, batch | `sbatch` script with many CPUs (Step 1) |
| `extract` | compute node, batch | `sbatch` script with `--workers` (Step 4) |
| `report`, `examples/diffusion_from_report.py` | compute node, small | batch job, or an interactive session (below) |
| Python exploration, notebooks 04/05/08/09 | compute node | interactive session, or the cluster's Jupyter service on a compute node |
| Force validation (UMA), notebook 06 | GPU compute node | batch job requesting a GPU |
| `recompute` (UMA single points and optimizations) | GPU compute node | batch job requesting a GPU (Step 5) |

**Batch jobs** run unattended: write the commands into a Slurm script and
submit it with `sbatch` (Steps 1, 3 and 4 show complete scripts).

**An interactive session** gives you a shell on a compute node; every command
typed in it runs there, not on the login node:

```bash
srun --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

Once the prompt changes to a compute node, activate the environment and work
normally (the `report` commands in Step 3 and the bare `python ...` commands in
this guide are meant for such a session or a batch script):

```bash
source /scratch/user/nchopper_tamu.edu/conda/minicondainstall/miniconda3/etc/profile.d/conda.sh
conda activate martini
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache --output postprocess_v5_3.75cutoff/reports/r3.25 \
    --startframe 100000 --max-screen-distance 3.25
exit                                                # leave the compute node and free the allocation
```

A single command can also be sent to a compute node without opening a shell, by
prefixing it with `srun` (as in the preview and report examples below).

For notebooks, use the cluster's web portal or Jupyter service if it offers one
(it starts the notebook kernel on a compute node), or start Jupyter inside an
interactive session and connect through an SSH tunnel. Do not start a Jupyter
server on the login node.

---

## Before you start

**Update the code and the environment** (on the login node; these are light):

```bash
cd /scratch/user/nchopper_tamu.edu/MartiniFuncs
git fetch origin
git merge --ff-only origin/claude_refactor
```

If the conda environment does not use an editable install of this clone,
reinstall it:

```bash
conda activate martini
pip install -e /scratch/user/nchopper_tamu.edu/MartiniFuncs
```

Check which copy Python imports:

```bash
python -c "import MartiniFuncs; print(MartiniFuncs.__file__)"
```

**Check the CLI options** for any command:

```bash
python -m MartiniFuncs run --help
```

```bash
python -m MartiniFuncs report --help
```

**Validate forces once per model setup** (before any reaction or UMA work on
extracted structures). This recomputes forces for a few full periodic frames
with your local UMA checkpoint and compares them with the forces in the dump.
It needs a GPU: run it in a batch job or interactive session that requests one
(for example `--gres=gpu:1`; your cluster may require a GPU type or partition),
never on the login node:

```bash
python -m MartiniFuncs.workflows.force_validation nve_2p5.lammpstrj \
    --checkpoint /path/to/uma-s-1p2p1.pt --units metal --charge 0 --spin 1 \
    --atol 0.05 --indices 0 150000 300000 --device cuda --output force_check.json
```

The report lists RMSE, maximum component error and the worst atom per frame;
`passed` requires every force component within `--atol` (eV/Angstrom).
Notebook `examples/workflows/06_reproduce_uma_forces.ipynb` does the same
interactively.

---

## Step 1: `run`, collect once

### The Slurm script

```bash
#!/usr/bin/env bash
#SBATCH --job-name=martini-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=martini-analysis-%j.out

set -euo pipefail

# Keep numerical libraries from starting extra threads in every worker.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
source /scratch/user/nchopper_tamu.edu/conda/minicondainstall/miniconda3/etc/profile.d/conda.sh
conda activate martini

cutoff='3.75'                                     # collection radius (Angstrom)
run_dir='./'
output_dir="./postprocess_v5_${cutoff}cutoff/"    # new folder for new package code
units='metal'
timestep_ps='0.0005'

python -m MartiniFuncs run "$run_dir" \
    --output "$output_dir" \
    --units "$units" \
    --timestep-ps "$timestep_ps" \
    --workers "${MARTINI_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}" \
    --batch-size 32 \
    --reactive-radius "$cutoff" \
    --temperature-dof 3N-3 \
    --no-export-pairs \
    --orientations \
    "$@"
```

Submit it from the simulation directory (submitting is the only part that
happens on the login node; the job itself runs on a compute node):

```bash
sbatch postprocess.slurm
```

Extra flags after the script name are passed through (`"$@"`):

```bash
sbatch postprocess.slurm --max-ring-dihedral-deviation 30 --report-name ring30
```

### What each setting does

| Flag | Meaning | Advice |
|---|---|---|
| `--units metal` | LAMMPS unit style of the dump | required; only `metal` is supported for analysis |
| `--timestep-ps 0.0005` | ps per MD step (not per saved frame) | must match the simulation input |
| `--workers N --batch-size B` | N local processes, B frames per task | `--workers` may not exceed `SLURM_CPUS_PER_TASK` |
| `--reactive-radius R` | collection radius, Angstrom (default 3.25); the cache holds only pairs below it | set it to the largest radius you will ever want to select |
| `--temperature-dof 3N-3` | degrees of freedom for the temperature column | omit to skip temperature |
| `--no-export-pairs` | skip the structure export pass | recommended; use `extract` later |
| `--orientations` | also cache each molecule's long axis and ring normal | needed for rotational dynamics; doubles the COM storage |
| `--frame-stride K` | keep every K-th frame in the cache | reading and unwrapping still use every frame |
| `--max-frames N` | read only the first N frames | quick previews of a new trajectory |
| `--reaction-events record\|stop\|off` | topology-change (reaction) detection | default `record`; `stop` ends the analysis before the first event |
| `--pair-backend cached\|reference` | reuse first-frame topology, or rebuild every frame | keep `cached` for nonreactive runs |
| `--no-pairs` | molecular statistics only | for systems without CPD/N4T pairs |
| `--max-*`, `--geometry-required` | selection for this run's own report/export only | the cache keeps every collected pair regardless |
| `--startframe`, `--endframe` | frame window for this run's report/export only | the cache always covers all frames read |

### Choosing the collection radius

Pick the **largest** radius you might want to select, because the cache cannot
hold anything beyond it. The script above collects at 3.75 Angstrom, so its
cache supports selections at 3.75 Angstrom and below. If you might later want
4.0 Angstrom, collect at 4.0 Angstrom in the first place. A larger radius only
costs a bigger candidate table (the 3.75 Angstrom NVT cache held about 4.8
million pair detections), not a slower `report`.

To go beyond a radius you already collected, run again at the larger radius.
This is a separate full pass over the dump into its own folder; it does not
add to the existing cache:

```bash
python -m MartiniFuncs run ./ --output ./postprocess_v5_4.0cutoff \
    --units metal --timestep-ps 0.0005 --workers 64 --reactive-radius 4.0 \
    --temperature-dof 3N-3 --orientations --no-export-pairs
```

### Variations

The full-size commands in this section use 64 workers: put them in the Slurm
script in place of its `python -m MartiniFuncs run ...` line and submit with
`sbatch`; do not run them on the login node.

A quick preview of a new trajectory (first 2,000 frames, few workers), sent to
a compute node with `srun`:

```bash
srun -n1 -c8 --mem=16G -t 00:30:00 python -m MartiniFuncs run ./ --output ./preview \
    --units metal --timestep-ps 0.0005 --workers 8 --max-frames 2000 --no-export-pairs
```

Keep every 10th frame (smaller cache, coarser lag statistics):

```bash
python -m MartiniFuncs run ./ --output ./postprocess_v5_stride10 \
    --units metal --timestep-ps 0.0005 --workers 64 --frame-stride 10 --no-export-pairs
```

Stop cleanly before the first reaction, so every statistic is pre-reaction:

```bash
python -m MartiniFuncs run ./ --output ./postprocess_v5_prereaction \
    --units metal --timestep-ps 0.0005 --workers 64 --reaction-events stop --no-export-pairs
```

The same run again with identical collection settings but different selection
flags reuses the cache (it prints `Reusing completed cache`) and only writes a
new report. Every setting that is not a selection or frame-window flag must
match the original run, or `run` refuses to reuse the cache:

```bash
python -m MartiniFuncs run ./ --output ./postprocess_v5_3.75cutoff \
    --units metal --timestep-ps 0.0005 --workers 64 --reactive-radius 3.75 \
    --temperature-dof 3N-3 --orientations --no-export-pairs \
    --report-name forming4 --max-forming-distance 4.0
```

### What the cache contains

`OUTPUT/cache/` holds everything later steps need; the dump is not copied.

| File | Contents |
|---|---|
| `manifest.json` | configuration (including `pair_collection`), source file fingerprints, code digest |
| `frames.npz`, `frames.csv` | per-frame `frame`, `timestep`, `time_ps`, density, volume, temperature, `candidate_pairs` |
| `com_angstrom.npy` | TOR-unwrapped molecular COMs (frames, molecules, 3), for displacements and MSD |
| `wrapped_com_angstrom.npy` | instantaneous COMs relative to the cell center, for RDFs |
| `com_nonaffine_angstrom.npy` | COMs with the cell's affine motion removed (barostat, shear) |
| `com_velocity_angstrom_ps.npy`, `net_force_eV_angstrom.npy` | molecular COM velocities and net forces |
| `orientation_axes.npy` | with `--orientations`: (frames, molecules, 2, 3) long axis and ring normal |
| `cell_angstrom.npy`, `cell_origin_angstrom.npy`, `box_lengths_angstrom.npy` | cell per frame |
| `species.npy`, `molecules.json` | formula and atom IDs of each COM column |
| `candidate_*.npy` | every collected pair with its metrics (screen distance, forming distances, angles, ring dihedral) |
| `pair_*.npy` | the collected pairs per frame, compactly |

All positions are Angstrom, times ps, velocities Angstrom/ps, forces eV/Angstrom.
Molecule columns follow `molecules.json`; atom IDs are LAMMPS IDs minus one.

---

## Step 2: explore the cache in Python

Work in a notebook or an interactive session **on a compute node** (see
[Where to run](#where-to-run-each-command-not-the-login-node)); nothing here
reads the dump, but loading and histogramming millions of rows takes several GB
of memory. The examples assume:

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MartiniFuncs.workflows.postprocess import load_analysis_cache

CACHE = Path("postprocess_v5_3.75cutoff/cache")
result, manifest = load_analysis_cache(CACHE)

frames = pd.DataFrame(result["frames"])
species = np.asarray(result["species"])
times = frames["time_ps"].to_numpy()
print(manifest["configuration"]["pair_collection"], len(frames), "frames,",
      dict(zip(*np.unique(species, return_counts=True))))
```

`result` is a dictionary of memory-mapped arrays, so loading is fast even for
hundreds of thousands of frames.

### 2a. Choose the analysis window (equilibration)

Plot the per-frame scalars and decide where the system has settled:

```python
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, layout="constrained")
for ax, column in zip(axes, ["density_g_cm3", "temperature_K", "candidate_pairs"]):
    if column in frames:
        ax.plot(frames["time_ps"], frames[column], linewidth=.5)
        ax.set_ylabel(column)
axes[-1].set_xlabel("time (ps)")
plt.show()
```

Add the LAMMPS thermo log (potential energy, pressure) on the same time axis:

```python
from MartiniFuncs.io.thermo import attach_thermo, discover_thermo_file, read_lammps_thermo

thermo_path = discover_thermo_file(".")            # the run directory; None if there is no thermo log
if thermo_path is not None:
    thermo = read_lammps_thermo(thermo_path, units="metal")
    joined, coverage = attach_thermo(frames, thermo)
    print(coverage)
    joined.plot(x="time_ps", y=["potential_energy_eV", "pressure_bar"], subplots=True, figsize=(8, 5))
```

Screen a scalar for stationarity with block statistics (absolute tolerance in
the observable's units):

```python
from MartiniFuncs.statistics.diffusion import scalar_stationarity

START = 100_000                                     # candidate first frame of the analysis window
window = frames["frame"] >= START
check = scalar_stationarity(frames.loc[window, "density_g_cm3"].to_numpy(), times[window],
                            block_frames=20_000, tolerance=0.002)
print(check["status"], check["reasons"])
```

Use the chosen first frame as `--startframe` in every `report` and `extract`
below. Frames are zero-based positions in the dump, not LAMMPS timesteps.

### 2b. Look at the collected pairs before choosing thresholds

```python
from MartiniFuncs.pairs.candidates import candidate_table

table = candidate_table(result)                     # one row per collected pair per frame
table = table[table["frame"] >= START]
print(len(table), "pair detections;", table["geometry_ok"].mean(), "fraction with measured geometry")
print(table[["screen_distance", "max_forming_distance", "axis_angle", "plane_angle"]].describe())
```

Histograms of every metric, to pick thresholds from the data:

```python
metrics = [c for c in ["screen_distance", "max_forming_distance", "distance_imbalance", "axis_angle",
                       "plane_angle", "approach_angle", "ring_dihedral"] if c in table]
fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
for ax, name in zip(axes.flat, metrics):
    ax.hist(table[name].dropna(), bins=60)
    ax.set_xlabel(name)
plt.show()
```

How many pairs survive a candidate selection, without writing anything:

```python
from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCriteria

for criteria in [CpdN4tCriteria(max_screen_distance=3.25),
                 CpdN4tCriteria(max_screen_distance=3.25, max_forming_distance=4.0),
                 CpdN4tCriteria(max_screen_distance=3.25, max_forming_distance=4.0, max_plane_angle=30)]:
    selected = candidate_table(result, criteria)
    print(criteria.active(), len(selected[selected["frame"] >= START]))
```

Exo versus endo (needs a cache made after the ring-dihedral metric was added):

```python
if "ring_dihedral" in table:
    near = CpdN4tCriteria(max_screen_distance=3.25, max_ring_dihedral_deviation=30)
    aligned = candidate_table(result, near)
    aligned = aligned[aligned["frame"] >= START]
    kind = np.where(aligned["ring_dihedral"].abs() < 90, "exo", "endo")
    print(pd.Series(kind).value_counts())
```

A free-energy profile of one metric (conditional on being collected):

```python
from MartiniFuncs.statistics.dynamics import pmf_1d

pmf = pmf_1d(table["plane_angle"].dropna(), temperature_K=400, edges=np.linspace(0, 90, 31), jacobian="sin")
plt.plot(pmf["x"], pmf["pmf_kJ_mol"])
plt.xlabel("plane angle (deg)"); plt.ylabel("W (kJ/mol)")
plt.show()
```

### 2c. Pairs and lifetimes in Python

`select_pairs` returns per-frame pair lists; `get_pair_lifetime` turns them into
encounter episodes:

```python
from dataclasses import asdict
from MartiniFuncs.pairs.candidates import select_pairs
from MartiniFuncs.pairs.lifetimes import get_pair_lifetime
from MartiniFuncs.trajectory.analysis import select_frame_window

criteria = CpdN4tCriteria(max_screen_distance=3.25, max_forming_distance=4.0)
pairs, shifts = select_pairs(result, criteria)
keep = select_frame_window(result, startframe=START)
in_window = [p for p, k in zip(pairs, keep) if k]
episodes = get_pair_lifetime(in_window, times[keep], max_gap_frames=0)
lifetimes = pd.DataFrame([asdict(e) for e in episodes])
complete = lifetimes[~lifetimes.left_censored & ~lifetimes.right_censored]
print(len(lifetimes), "encounters;", complete["observed_span_ps"].describe())
```

`max_gap_frames=2` bridges up to two missing frames inside one encounter.

### 2d. Molecular displacements and diffusion

Lag statistics for the window (the same numbers `report` writes):

```python
from MartiniFuncs.trajectory.analysis import molecular_window_statistics

window_stats = molecular_window_statistics(result, startframe=START, max_lag_ps=100.)
for name, entry in window_stats["species"].items():
    plt.loglog(entry["lag_ps"][1:], entry["msd_angstrom2"][1:], label=name)
plt.xlabel("lag (ps)"); plt.ylabel("MSD (Angstrom$^2$)"); plt.legend()
plt.show()
```

A log-log MSD shows the regimes directly: slope 2 at sub-ps lags (ballistic),
a plateau (caging), and slope 1 only if the molecules diffuse.

Squared displacement from one reference frame, per molecule:

```python
from MartiniFuncs.statistics.md import get_molecular_reference_msd

rows = np.flatnonzero(keep)
for name in sorted(set(species)):
    columns = np.flatnonzero(species == name)
    reference = get_molecular_reference_msd(result["com_angstrom"][np.ix_(rows, columns)], times[rows],
                                            unwrapped=True, selection=name)
    plt.semilogx(reference["elapsed_time_ps"][1:], reference["msd_angstrom2"][1:], label=name)
plt.xlabel("time since reference frame (ps)"); plt.ylabel("MSD from t0 (Angstrom$^2$)"); plt.legend()
plt.show()
```

Diffusion with temporal-block uncertainty and explicit screens (it reports
`not supported` rather than a number when the MSD is not diffusive):

```python
from MartiniFuncs.statistics.diffusion import diffusion_diagnostics

for name in sorted(set(species)):
    positions = result["com_angstrom"][np.ix_(rows, np.flatnonzero(species == name))]
    check = diffusion_diagnostics(positions, times[rows], fit_intervals_ps=[(20, 50), (50, 100), (100, 200)],
                                  block_frames=40_000)
    for fit in check["fits"]:
        print(name, (fit["fit_start_ps"], fit["fit_end_ps"]), fit["status"], fit.get("alpha"),
              fit.get("D_pooled_cm2_s"), fit["reasons"][:1])
```

Caging diagnostics: non-Gaussian parameter and whether the frame spacing
resolves ballistic motion:

```python
from MartiniFuncs.statistics.dynamics import displacement_moments, sampling_diagnostics

name = "C5H6"
columns = np.flatnonzero(species == name)
positions = result["com_angstrom"][np.ix_(rows, columns)]
velocities = result["com_velocity_angstrom_ps"][np.ix_(rows, columns)]
moments = displacement_moments(positions, times[rows])
plt.semilogx(moments["lag_ps"], moments["alpha2"]); plt.xlabel("lag (ps)"); plt.ylabel("alpha_2")
plt.show()
print(sampling_diagnostics(positions, velocities, times[rows])["notes"])
```

### 2e. Rotational dynamics (needs `--orientations`)

```python
from MartiniFuncs.statistics.dynamics import correlation_time, orientation_correlation

axes = result["orientation_axes"]                   # (frames, molecules, 2, 3): long axis, ring normal
if axes is not None:
    for name in sorted(set(species)):
        normals = axes[np.ix_(rows, np.flatnonzero(species == name))][:, :, 1]
        c2 = orientation_correlation(normals, times[rows], order=2, nlag=20_000)
        tau = correlation_time(c2["lag_ps"], c2["correlation"])
        print(name, "ring-normal P2 correlation time", tau["tau_ps"], "ps; reached zero:", tau["reached_zero"])
```

If the ring normals decorrelate quickly while the COMs stay caged, the
molecules rotate in place (plastic-crystal-like) rather than being frozen.

---

## Step 3: `report`, statistics for any selection

`report` reads only the cache and needs only one CPU, but it is still compute:
run the examples below in an interactive session on a compute node or put them
in a batch script (as in the radius scan below), not on the login node. Give
every report its own `--output` folder;
re-running an identical command reuses the finished report, and a different
selection in an existing folder is refused (nothing is overwritten).

### Examples

All collected pairs, after equilibration:

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/all_from100k --startframe 100000
```

Narrow the radius to 3.25 Angstrom:

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25 --startframe 100000 --max-screen-distance 3.25
```

Exo/endo-aligned pairs only (within 30 degrees of 0 or 180):

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25_ring30 --startframe 100000 \
    --max-screen-distance 3.25 --max-ring-dihedral-deviation 30
```

A stricter, reaction-like geometry:

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/close_aligned --startframe 100000 \
    --max-screen-distance 3.25 --max-forming-distance 3.8 --max-plane-angle 30 \
    --max-axis-angle 30 --max-ring-dihedral-deviation 30
```

Pairs only (skip MSD and correlations; much faster):

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25_pairs_only --startframe 100000 \
    --max-screen-distance 3.25 --no-lag-statistics
```

Molecular statistics only (no pair lifetimes):

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/msd_only --startframe 100000 \
    --lag-statistics msd vacf --no-pair-lifetimes --max-lag-ps 200
```

An explicit diffusion fit (only after checking the MSD is linear in that range):

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/fit_50_100 --startframe 100000 --fit-ps 50 100
```

Lifetimes that tolerate brief gaps, plus scalar stability and COM RDFs:

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25_gap2_rdf --startframe 100000 \
    --max-screen-distance 3.25 --max-gap-frames 2 --stability --rdf-bins 100
```

Early versus late halves of the trajectory, for a drift check:

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25_early --startframe 100000 --endframe 200000 \
    --max-screen-distance 3.25
```

```bash
python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25_late --startframe 200000 \
    --max-screen-distance 3.25
```

A radius scan in one Slurm job (one CPU is enough):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=martini-reports
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=martini-reports-%j.out
set -euo pipefail
source /scratch/user/nchopper_tamu.edu/conda/minicondainstall/miniconda3/etc/profile.d/conda.sh
conda activate martini

cache=postprocess_v5_3.75cutoff/cache
for radius in 3.0 3.25 3.5 3.75; do
    python -m MartiniFuncs report "$cache" --output "postprocess_v5_3.75cutoff/reports/r${radius}" \
        --startframe 100000 --max-screen-distance "$radius" --no-lag-statistics
done
```

Send a single report to a compute node without opening a session:

```bash
srun -n1 -c1 --mem=16G -t 00:30:00 python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/reports/r3.25 --startframe 100000 --max-screen-distance 3.25
```

Start the reports automatically when the `run` job finishes:

```bash
jobid=$(sbatch --parsable postprocess.slurm)
sbatch --dependency=afterok:$jobid reports.slurm
```

### What a report contains

| File | Contents |
|---|---|
| `manifest.json` | the cache, frame window, selection (`criteria`), time window, per-species files, notices |
| `frames.csv` | per-frame scalars in the window; `selected_pairs` per frame when a selection is given |
| `species_000_lags.csv`, `species_001_lags.csv`, ... | `lag_ps`, `n_origins`, `msd_angstrom2`, `vacf`, `force_acf` (manifest maps file to species) |
| `pair_lifetimes.csv` | one row per encounter: frames, times, span, detections, censoring, source frames |
| `pair_metric_histograms.csv` | histograms of the screen distance and every metric for the selected pairs |
| `diffusion.csv` | with `--fit-ps`: D per species with slope, intercept and residual |
| `topology_events.csv` | reaction/topology events recorded by `run`, if any |
| `scalar_stability.json` | with `--stability`: ACF, inefficiency and batch errors of each scalar column |
| `rdf_000.csv`, ..., `rdf_species.json` | with `--rdf-bins`: species-pair COM RDFs |

### Reading report output

```python
import json
from pathlib import Path
import pandas as pd

report = Path("postprocess_v5_3.75cutoff/reports/r3.25")
manifest = json.loads((report / "manifest.json").read_text())
lags = {name: pd.read_csv(report / entry["file"]) for name, entry in manifest["species"].items()}
lifetimes = pd.read_csv(report / "pair_lifetimes.csv")
histograms = pd.read_csv(report / "pair_metric_histograms.csv")
print(manifest["time_window_ps"], manifest["configuration"]["criteria"])
print(lifetimes[["observed_span_ps", "detected_frames"]].describe())
print(histograms[histograms["metric"] == "max_forming_distance"].head())
```

Diffusion coefficients (Einstein and Green-Kubo, with the MSD scaling exponent)
from a report's lag files:

```bash
python examples/diffusion_from_report.py postprocess_v5_3.75cutoff/reports/msd_only --fit-ps 50 100
```

Trust a D value only when `alpha` is close to 1 and the two estimates agree;
otherwise the MSD is still caged or ballistic in that range.

---

## Step 4: `extract`, structures for the final selection

`extract` rereads the 43 GB dump and should always run as a **batch job** on
a compute node (the Slurm script below), with `--workers` matching the CPUs you
request. It reads the selected frames from the dump again and writes one
structure per selected pair per frame, with the dump's forces (eV/Angstrom).
Each export becomes many structures, so narrow the selection and the frame
window first: the same selection through `report` tells you how many pairs to
expect (`selected_pairs` in `frames.csv`).

### Examples

Pairs only, one frame window, exo/endo-aligned and close:

```bash
python -m MartiniFuncs extract postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/exports/close_aligned_100k_110k \
    --startframe 100000 --endframe 110000 \
    --max-screen-distance 3.25 --max-forming-distance 3.8 --max-ring-dihedral-deviation 30
```

With whole neighboring molecules within 3 Angstrom of the pair:

```bash
python -m MartiniFuncs extract postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/exports/close_aligned_shell3 \
    --startframe 100000 --endframe 110000 \
    --max-screen-distance 3.25 --max-forming-distance 3.8 --max-ring-dihedral-deviation 30 \
    --shell-radius 3.0
```

A larger export as a Slurm job with workers:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=martini-extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=martini-extract-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source /scratch/user/nchopper_tamu.edu/conda/minicondainstall/miniconda3/etc/profile.d/conda.sh
conda activate martini

python -m MartiniFuncs extract postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/exports/ring30_shell3 \
    --startframe 100000 --max-screen-distance 3.25 --max-ring-dihedral-deviation 30 \
    --shell-radius 3.0 --workers "$SLURM_CPUS_PER_TASK" --batch-size 32
```

Every collected pair in a short window (no selection), e.g. for a quick look:

```bash
python -m MartiniFuncs extract postprocess_v5_3.75cutoff/cache \
    --output postprocess_v5_3.75cutoff/exports/all_100k_100100 --startframe 100000 --endframe 100100
```

### What an export contains

| File | Contents |
|---|---|
| `pairs.extxyz` | one frame per selected pair: CPD rows, then N4T rows; `pair_role`, `global_index`, `lammps_id`, `lammps_forces` |
| `clusters.extxyz` | with `--shell-radius`: the pair plus whole neighboring molecules (`pair_role` = `shell`) |
| `index.jsonl` | one line per structure: frame, time, atom IDs, image shift, full measured geometry |
| `frames.csv` | selected frames with their pair counts (including zero) |
| `manifest.json` | cache, window, selection, structure count, force units |

`structure_index` links the three files row for row.

### Reading an export

```python
import json
from pathlib import Path
from ase.io import read

export = Path("postprocess_v5_3.75cutoff/exports/close_aligned_shell3")
pairs = read(export / "pairs.extxyz", index=":")
index = [json.loads(line) for line in (export / "index.jsonl").read_text().splitlines()]
first = pairs[0]
print(len(pairs), "structures;", first.get_chemical_formula(), set(first.arrays["pair_role"]))
print(first.arrays["lammps_forces"][:3])             # eV/Angstrom, as written by LAMMPS
print(index[0]["frame"], index[0]["geometry"]["forming_distances"], index[0]["geometry"]["ring_dihedral"])
```

Distributions of the exported geometries:

```python
import numpy as np
from MartiniFuncs.chemistry.cpd_n4t.pairs import load_pair_geometries

geometry = load_pair_geometries(export / "index.jsonl")
print({name: float(np.nanmean(values)) for name, values in geometry.items()})
```

Next steps: Step 5 recomputes these structures with UMA (isolated-fragment forces,
the environmental force F_env, energies, optimizations).
`examples/workflows/07_prepare_reaction_inputs.ipynb`
turns a `clusters.extxyz` structure into a frozen-shell input and a row-identified
force archive.

---

## Step 5: `recompute`, isolated forces, F_env and energies

The forces exported by `extract` (`lammps_forces`) are the **total** MD forces in
the periodic system. `recompute` evaluates parts of each exported structure in
isolation with the same UMA model, so the total force can be split into an
internal part and the force exerted by the surroundings:

F_env = F_MD - F_isolated (per atom, eV/Angstrom)

It reads `pairs.extxyz` or `clusters.extxyz` and writes a new `.extxyz` with the
input's arrays and info kept and the new quantities added. It needs a GPU and
the UMA checkpoint used for the MD, and always runs as a batch job.

**Validate forces first.** F_env is only meaningful if the calculator reproduces
the MD forces on the full periodic frame (see
[Before you start](#before-you-start)). Otherwise F_env also contains model
differences.

### The three tasks

| `--task` | What it does | Adds |
|---|---|---|
| `single-point` (default) | evaluates the pair as one isolated fragment, and each shell molecule alone, at the MD geometry | arrays `isolated_forces`, `environment_forces`, `isolated_fragment` (0 = pair, k = shell molecule); info `energy_pair`, `energy_shell` |
| `single-point --molecules` | also evaluates each pair member alone | arrays `molecule_isolated_forces`, `molecule_environment_forces`; info `energy_CPD`, `energy_MAL`, `interaction_energy` = E_pair - E_CPD - E_MAL |
| `relax-fragments` | relaxes the isolated pair (and with `--molecules` each member) from the MD geometry | info `energy_L`, `relaxed_energy_L`, `deformation_energy_L` = energy_L - relaxed_energy_L, `relaxation_converged_L`, `relaxation_steps_L` for L = pair, CPD, MAL; relaxed fragments in `OUTPUT.relaxed.extxyz` |
| `frozen-shell` | optimizes the pair with every shell atom fixed (clusters only) | the cluster at the optimized geometry with `md_positions`, `frozen`, `optimized_forces`; info `energy_md_geometry`, `energy_optimized`, `optimization_converged` |

What the two force decompositions mean:

- **Pair mode** (default): F_isolated is the force on each atom of the
  CPD + N4T pair evaluated together without anything else. F_env on a pair
  atom is then the force from everything beyond the pair. The CPD-N4T
  interaction counts as internal.
- **Molecule mode** (`--molecules`, added on top): each molecule is evaluated
  alone, so `molecule_environment_forces` on a CPD atom also includes the force
  from its N4T partner. The difference between the two decompositions on a pair
  atom is exactly the partner's contribution.
- Shell molecules (clusters) are always evaluated one molecule at a time, so
  their F_env is the force from all other molecules in the MD frame.

Energies are eV; every fragment is neutral and a singlet by default
(`--charge`, `--spin` apply to all fragments). `relax-fragments` and
`frozen-shell` change the geometry, so their structures no longer correspond to
the exported MD forces (kept for reference, with `md_positions` recording the
geometry they belong to).

### Examples

Single points on the final clusters, pair mode:

```bash
python -m MartiniFuncs recompute postprocess_v5_3.75cutoff/exports/final/clusters.extxyz \
    --output postprocess_v5_3.75cutoff/exports/final/clusters_fenv.extxyz \
    --checkpoint /path/to/uma-s-1p2p1.pt --device cuda
```

Pair and molecule decompositions with interaction energies:

```bash
python -m MartiniFuncs recompute postprocess_v5_3.75cutoff/exports/final/pairs.extxyz \
    --output postprocess_v5_3.75cutoff/exports/final/pairs_fenv_molecules.extxyz \
    --checkpoint /path/to/uma-s-1p2p1.pt --device cuda --molecules
```

Relax the isolated pair and both molecules (deformation energies):

```bash
python -m MartiniFuncs recompute postprocess_v5_3.75cutoff/exports/final/pairs.extxyz \
    --output postprocess_v5_3.75cutoff/exports/final/pairs_relaxed.extxyz \
    --checkpoint /path/to/uma-s-1p2p1.pt --device cuda --task relax-fragments --molecules \
    --optimizer BFGS --fmax 0.05 --steps 500
```

Optimize each pair inside its frozen shell:

```bash
python -m MartiniFuncs recompute postprocess_v5_3.75cutoff/exports/final/clusters.extxyz \
    --output postprocess_v5_3.75cutoff/exports/final/clusters_frozen_shell.extxyz \
    --checkpoint /path/to/uma-s-1p2p1.pt --device cuda --task frozen-shell --fmax 0.05 --steps 1000
```

The default inference settings are the MD settings with `merge_mole` and
`compile` turned off. Both are UMA's fixed-composition MD fast path; recompute
evaluates fragments of changing composition, and the unmerged model gives the
same forces (agreement about 1e-6 eV/Angstrom). These defaults also run on CPU
and on machines without a C++ compiler, e.g. Windows (`--device cpu`; about
0.1-2 s per fragment). Further overrides go in a JSON file passed with
`--inference-settings settings.json`.

Testing on a laptop or workstation without a GPU works the same way:

```bash
python -m MartiniFuncs recompute EXPORT/clusters.extxyz --output EXPORT/clusters_fenv_cpu.extxyz     --checkpoint /path/to/uma-s-1p2p1.pt --device cpu --molecules
```

As a GPU batch job (adjust the GPU request to your cluster):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=martini-recompute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=martini-recompute-%j.out
set -euo pipefail
source /scratch/user/nchopper_tamu.edu/conda/minicondainstall/miniconda3/etc/profile.d/conda.sh
conda activate martini

export_dir=postprocess_v5_3.75cutoff/exports/final
python -m MartiniFuncs recompute "$export_dir/clusters.extxyz" --output "$export_dir/clusters_fenv.extxyz" \
    --checkpoint /path/to/uma-s-1p2p1.pt --device cuda --molecules
```

Outputs are never overwritten (choose a new `--output`), and a run that fails
partway leaves no partial file.

### From Python

The same tasks are classes in `MartiniFuncs.recompute` and work with any ASE
calculator:

```python
from ase.io import read
from MartiniFuncs.recompute.driver import recompute_file
from MartiniFuncs.recompute.evaluation import FragmentEvaluator, FragmentRelaxer, uma_calculator
from MartiniFuncs.recompute.tasks import ForceDecomposition, FragmentRelaxation, FrozenShellOptimization

calculator = uma_calculator("/path/to/uma-s-1p2p1.pt", task="omol", device="cuda")   # or device="cpu"
evaluator = FragmentEvaluator(calculator, charge=0, spin=1)

export = "postprocess_v5_3.75cutoff/exports/final"
recompute_file(f"{export}/clusters.extxyz", f"{export}/clusters_fenv.extxyz",
               ForceDecomposition(evaluator, molecules=True))

relaxer = FragmentRelaxer(evaluator, optimizer="BFGS", fmax=0.05, steps=500)
recompute_file(f"{export}/pairs.extxyz", f"{export}/pairs_relaxed.extxyz", FragmentRelaxation(relaxer, molecules=True))
recompute_file(f"{export}/clusters.extxyz", f"{export}/clusters_frozen.extxyz", FrozenShellOptimization(relaxer))

# One structure at a time:
cluster = read(f"{export}/clusters.extxyz", index=0)
result = ForceDecomposition(evaluator).apply(cluster).structure
print(result.info["energy_pair"], result.arrays["environment_forces"][:3])
```

### Reading the results

```python
import numpy as np
import pandas as pd
from ase.io import read

structures = read("postprocess_v5_3.75cutoff/exports/final/clusters_fenv.extxyz", index=":")
rows = []
for s in structures:
    roles = s.arrays["pair_role"]
    pair = roles != "shell"
    f_env = s.arrays["environment_forces"][pair]
    rows.append(dict(frame=s.info.get("frame_index"), structure=s.info.get("structure_index"),
                     interaction_energy_eV=s.info.get("interaction_energy"),
                     net_env_force_CPD=np.linalg.norm(s.arrays["environment_forces"][roles == "CPD"].sum(axis=0)),
                     net_env_force_MAL=np.linalg.norm(s.arrays["environment_forces"][roles == "MAL"].sum(axis=0)),
                     rms_env_force_pair_atoms=np.sqrt((f_env ** 2).sum(axis=1).mean())))
print(pd.DataFrame(rows).describe())
```

---

## `collect`: structures without a report

`collect` is `run` and `extract` in one pass, with no statistical report and
`--shell-radius` required. Prefer `run` followed by `extract`: you keep the
reusable cache, and you choose the selection after looking at the data. Use
`collect` only when you already know the selection and only want structures.
Like `run`, it reads the whole dump: put the command in a batch script.

```bash
python -m MartiniFuncs collect ./ --output ./collect_ring30 --units metal --timestep-ps 0.0005 \
    --workers 64 --reactive-radius 3.25 --shell-radius 3.0 --max-ring-dihedral-deviation 30 \
    --startframe 100000
```

---

## Designed pairs: the pair builder

`MartiniFuncs.chemistry.cpd_n4t.build` places a rigid CPD against a fixed N4T at
a requested geometry, independent of any trajectory. Use it for controlled
scans (distance, exo/endo, tilt) and to compare sampled encounters with
designed ones. Notebook `examples/workflows/09_build_pairs.ipynb` does the same
with plots and export. It is light (seconds per structure) but still belongs in
an interactive session or notebook on a compute node.

```python
from ase.io import write
from MartiniFuncs.chemistry.cpd_n4t.build import build_pair, rdkit_templates

cpd, n4t = rdkit_templates()                        # MMFF-relaxed templates; or your own relaxed Atoms
atoms, geometry = build_pair(cpd, n4t, distance=3.0, ring_dihedral=180, plane_angle=15)
print(geometry.approach_type, geometry.forming_distances, atoms.info["min_intermolecular_distance"])
write("endo_3A.xyz", atoms)
```

A scan over distance and exo/endo, skipping overlapping structures:

```python
structures = []
for distance in (2.6, 3.0, 3.4, 3.8):
    for dihedral in (0, 180):
        try:
            atoms, geometry = build_pair(cpd, n4t, distance=distance, ring_dihedral=dihedral,
                                         plane_angle=15, min_contact=1.0)
        except ValueError as exc:
            print(distance, dihedral, "skipped:", exc)
            continue
        structures.append(atoms)
write("designed_pairs.extxyz", structures, format="extxyz")
```

Other distance definitions (`distance_kind="max_forming"` or `"screen"`) and
templates from a trajectory frame:

```python
from MartiniFuncs.chemistry.cpd_n4t.build import templates_from_frame
from MartiniFuncs.io.lammps import read_lammps_frame

frame_cpd, frame_n4t = templates_from_frame(read_lammps_frame("nve_2p5.lammpstrj", index=0, units="metal").atoms)
atoms, geometry = build_pair(frame_cpd, frame_n4t, distance=3.2, distance_kind="max_forming", ring_dihedral=180)
print(max(geometry.forming_distances))               # 3.2
```

---

## Selection reference

| CLI flag | Python (`CpdN4tCriteria(...)`) | Selects pairs with | Units |
|---|---|---|---|
| `--max-screen-distance` | `max_screen_distance` | CPD COM to C=C COM at most this (at most the collected radius) | Angstrom |
| `--max-forming-distance` | `max_forming_distance` | the longer of the two forming C...C distances at most this | Angstrom |
| `--max-distance-imbalance` | `max_distance_imbalance` | the two forming distances differ by at most this | Angstrom |
| `--max-axis-angle` | `max_axis_angle` | diene termini axis within this of the C=C axis | degrees (0-90) |
| `--max-plane-angle` | `max_plane_angle` | the two reactive ring planes within this of parallel | degrees (0-90) |
| `--max-approach-angle` | `max_approach_angle` | site separation within this of the diene-plane normal | degrees (0-90) |
| `--max-ring-dihedral-deviation` | `max_ring_dihedral_deviation` | CH2-termini-C=C-N dihedral within this of 0 (exo) or 180 (endo) | degrees (0-90) |
| `--geometry-required` | `require_geometry=True` | only pairs whose geometry could be measured | |

Rules:

- Each threshold is an upper bound; equality passes.
- Any geometric threshold also requires a measured geometry.
- No selection at all uses every collected pair.
- The same rules apply everywhere: `report`, `extract`, `run`'s own report,
  `select_pairs`, `candidate_table`, `screen_frame`.

The signed `ring_dihedral` (stored in candidate tables and exports) is near 0
for exo and near +/-180 for endo; its sign is the twist direction (mirror
images), not exo/endo.

---

## When do I need a new `run`?

| You want to... | New `run`? |
|---|---|
| change thresholds or narrow the radius | no: `report` / `extract` with selection flags |
| change the frame window | no: `--startframe` / `--endframe` |
| select a radius larger than the collected one (e.g. 4.0 from a 3.75 cache) | yes, at the larger radius |
| change `--bond-scale`, `--frame-stride`, `--temperature-dof`, `--orientations` | yes |
| use a metric added to the code after the cache was made (e.g. ring dihedral) | yes |
| analyze a new or extended trajectory | yes |
| `report`/`extract` an old cache after updating the code | no: they work on old caches (without the newer metrics) |
| another `run` after updating the code | yes, into a new `--output` folder: the cache records the code digest |

---

## Common errors and what they mean

| Message | Meaning and fix |
|---|---|
| `Cache does not match source files, analysis settings, or package code; use a new --output directory` | a `run` into an existing cache with different settings, an updated package, or a changed dump. Use a new `--output`. |
| `Output exists with different settings or is incomplete: ...; choose a new name` | a `report`/`extract` folder already holds a different product. Use a new `--output`. |
| `max_screen_distance must be at most 3.75 Angstrom, the reactive_radius collected by the analysis` | the selection is wider than the collection. Narrow it, or `run` again with a larger `--reactive-radius`. |
| `CpdN4tCriteria thresholds metric 'ring_dihedral_deviation', which these candidates do not have` | the cache predates that metric. `run` again with the current code. |
| `pair model 'cpd_n4t' has no threshold --max-...` | a threshold flag of another pair model. |
| `--workers exceeds SLURM_CPUS_PER_TASK` | lower `--workers` or request more CPUs. |
| `Source files changed since caching; refuse to export mismatched structures/forces` | the dump was modified or moved after `run`; `extract` refuses rather than mixing frames. |
| `Incomplete or missing analysis cache` | the path is not a finished cache (point at `OUTPUT/cache`, and let `run` finish). |

---

## A complete session, start to finish

Steps marked *login* run on the login node; everything else runs on a compute
node, as a batch job or inside an interactive session (`srun ... --pty bash`).

1. *login*, then compute: update the code (Before you start), then preview on a
   compute node (about 10 minutes):

   ```bash
   srun -n1 -c8 --mem=16G -t 00:30:00 python -m MartiniFuncs run ./ --output ./preview \
       --units metal --timestep-ps 0.0005 --workers 8 --max-frames 2000 --no-export-pairs
   ```

2. *login*: submit the full collection as a batch job:

   ```bash
   jobid=$(sbatch --parsable postprocess.slurm)      # 3.75 Angstrom, --orientations, --no-export-pairs
   ```

3. Compute node: explore in a notebook (Step 2): choose `START = 100_000` from density,
   energy and the candidate histograms; choose thresholds, e.g.
   `max_screen_distance=3.25, max_forming_distance=3.8, max_ring_dihedral_deviation=30`.

4. Compute node (interactive session or batch): reports for the decisions (cheap, repeat freely):

   ```bash
   python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache --output postprocess_v5_3.75cutoff/reports/all \
       --startframe 100000
   ```

   ```bash
   python -m MartiniFuncs report postprocess_v5_3.75cutoff/cache --output postprocess_v5_3.75cutoff/reports/final \
       --startframe 100000 --max-screen-distance 3.25 --max-forming-distance 3.8 \
       --max-ring-dihedral-deviation 30 --no-lag-statistics
   ```

5. Compute node: check diffusion honestly:

   ```bash
   python examples/diffusion_from_report.py postprocess_v5_3.75cutoff/reports/all --fit-ps 50 100
   ```

6. Batch job: structures for the final selection (put this in a Slurm script with 32 CPUs):

   ```bash
   python -m MartiniFuncs extract postprocess_v5_3.75cutoff/cache --output postprocess_v5_3.75cutoff/exports/final \
       --startframe 100000 --max-screen-distance 3.25 --max-forming-distance 3.8 \
       --max-ring-dihedral-deviation 30 --shell-radius 3.0 --workers 32
   ```

7. GPU compute node: validate forces once, then recompute the final structures (Step 5):

   ```bash
   python -m MartiniFuncs recompute postprocess_v5_3.75cutoff/exports/final/clusters.extxyz \
       --output postprocess_v5_3.75cutoff/exports/final/clusters_fenv.extxyz \
       --checkpoint /path/to/uma-s-1p2p1.pt --device cuda --molecules
   ```

8. Prepare reaction inputs (notebook 07);
   compare with designed pairs from notebook 09.
