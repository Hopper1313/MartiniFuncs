# Assess diffusion and equilibration

Read notebook 08 in order: possible reactions, scalar stability, diffusive motion,
structural stability, then conclusions. Each main result table now includes a
finding and next step. `SHOW_DETAILS=False` hides the full diagnostic tables and
block plots; set it to True for inspection. All detailed arrays remain in the kernel.
The sampling-budget table explains whether the requested lag/block choices fit
the available trajectory. Too few blocks means uncertainty is unresolved; it
does not by itself mean the system is out of equilibrium.

`scalar_summary` reports changes in the observable's own units and explains missing
tolerances, correlation limits, or disagreement. `diffusion_summary` distinguishes
unavailable lag ranges, unresolved diffusion, mixed evidence and candidate fits.
It does not choose the best-looking fit. Export includes these findings and the
sampling budget alongside the detailed tables.

Open [notebook 08](../examples/workflows/08_diffusion_and_equilibration.ipynb).
Set `RUN_DIR`, `UNITS` and `TIMESTEP_PS`; default discovery finds the input,
data and dump. Alternatively set `CACHE_DIR` to your CLI cache. A cache supplies
its original physics and sampling settings; changing the notebook's loading
settings does not change cached data.

`WINDOWS` contains explicit source-frame intervals `(startframe, endframe)`.
Indices are zero-based, end-exclusive; `(0, None)` selects all available frames.
Run the loading cell once, then change windows and rerun diagnostic cells.
Check the full history before choosing later windows. No automatic burn-in
selection is performed.

## Possible reactions

Notebook 08 enables `REACTION_ATOM_COUNTS=(11, 23)` for current CPD/MAL, so a
34-atom connected component is a possible product. Change the counts for other
reactants. Connectivity is rebuilt from distances with `BOND_SCALE` on every
source frame before striding. At the first hit, `analyze_run` raises
`PossibleReactionError`; the notebook displays its atom IDs, frame-local indices,
formula, frame index and timestep. Fixed-species diffusion must not continue
silently after membership changes. A frame-zero hit is a preexisting candidate,
not a measured formation event.

The general `analyze_run` API leaves this chemistry-specific screen off unless
`reaction_atom_counts` is supplied. For a compatible CLI cache, use
`--reaction-atom-counts 11 23`; old unchecked caches are rejected by notebook 08.
Screening rebuilds connectivity on every source frame (about 3 ms per 1360-atom
frame) and runs in the frame workers when `--workers` > 1.
`MAX_FRAMES` limits the checked range too. This rule cannot catch larger
aggregates, fragmentation, or all other reaction mechanisms.

To inspect all occurrences instead of stopping at the first:

```python
from MartiniFuncs.lammpsio import iter_lammps_frames
from MartiniFuncs.reaction_detection import scan_possible_products

records = scan_possible_products(
    iter_lammps_frames(dump_path, units="metal", input_datafile=data_path),
    reactant_atom_counts=(11, 23), bond_scale=1.3,
)
```

Records retain persistent atom IDs, so repeated appearances can be compared for
persistence or reversibility. Transient close contacts can merge the inferred
graph; count alone does not establish a chemical product. Inspect actual bonds
and composition before deciding whether to truncate analysis before the first
candidate (`MAX_FRAMES=first_candidate_frame`) or implement product-aware statistics.

## What to set and inspect

- `FIT_INTERVALS_PS`: at least three distinct positive lag intervals. Examine
  sustained linear MSD growth and agreement between fits, not just a positive slope.
- `BLOCK_DURATIONS_PS`: compare at least two increasing durations. Each temporal
  block should span at least five times the longest fitted lag, and each assessment
  needs at least eight complete blocks. A short trajectory can legitimately fail.
- `SCALAR_TOLERANCES`: physically meaningful absolute differences, in each scalar's
  units. Defaults are unset, so no scalar equivalence is claimed automatically.
- `LOAD_THERMO=True`: automatically read a unique LAMMPS .out/.log; use
  `THERMO_FILE` and `THERMO_BLOCK` to resolve ambiguity. Match exact timesteps,
  use each scalar at its observed spacing, and never interpolate. See
  [thermo output](thermo_output.md).
- `THERMO_CSV`: alternative manual scalar data with unique integer source `frame` indices
  and columns such as `potential_energy_eV`, `pressure_bar`, `temperature_K`.
  Produce this from your simulation output, mapping timesteps to source frames;
  This CSV option bypasses automatic output parsing. Missing samples remain inconclusive. Density
  and volume are computed directly from the dump and are always assessed when available.
- Inspect molecular RDF sections for structural evolution and block-D histories
  for drift or persistent correlations. Add slower system-specific observables.

The notebook omits cached atomic temperature because its COM-bias convention may
not match the simulation. Thermo temperature must use a documented LAMMPS compute.
Pressure/energy come from the output, not reconstruction from forces. Missing observables are printed.

## Package interfaces and uncertainty

`diffusion_diagnostics.diffusion_diagnostics(positions, times_ps, ...)` takes
one species' TOR COM positions, shape `(frames, molecules, 3)`, in Angstrom.
It computes all-molecule MSD, free-intercept slope/6 estimates, log-scaling
exponents, and independent fits within nonoverlapping temporal blocks. Bootstrap
resampling uses whole blocks, preserving simultaneous molecular correlations.
Its 95% intervals assume blocks are sufficiently independent; they are not
ordinary least-squares errors from correlated MSD points. Values in m²/s are
Angstrom²/ps multiplied by `1e-8`. Fit-lag sampling is capped at 100 points.

`scalar_stationarity(values, times_ps, ...)` compares chronological groups of
block means. Its bootstrap difference interval must fit entirely inside the
chosen tolerance. An FFT first-zero ACF estimate screens whether blocks span
ten correlation times. This estimate can miss slow modes. Constant traces or
too few blocks do not pass. Density and volume do not provide independent evidence.

Numerical thresholds (alpha 0.8–1.2, 25% sensitivity/temporal shift, 50% relative
CI half-width) are configurable **heuristics**, not universal scientific limits.
A candidate requires all checks for that row; agreement across block sizes and
source windows, structural stability and independent replicas require inspection.
No row certifies equilibrium. Finite-system diffusion is reported without a
hydrodynamic finite-size correction.

The module docstrings give equations and variable definitions. Notebook references
describe TOR, correlated MSD errors, and observable-dependent equilibration.
The implementation does not claim to reproduce an optimal covariance-aware
diffusion estimator or an automatic equilibration-detection algorithm.

## Outputs and validation

The kernel retains `scalar_table`, `diffusion_table`, and detailed dictionaries
with block estimates/curves. `WRITE_FILES=True` creates a **new** output directory
containing both tables, frame observables and settings. Existing output directories
are rejected. Pair screening and UMA inference are not needed.

Tests cover known Brownian diffusion and units, confined motion, changing mobility,
scalar stationarity/drift, inadequate data, and notebook execution on the dump fixture
and synthetic data. These establish software behavior, not convergence of your MD.
