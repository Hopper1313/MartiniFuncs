# Molecular simulation → reaction inputs

The scientific aim is to estimate reaction barriers for molecular encounters
sampled in UMA/fairchem MD, comparing an applied-force treatment with an explicit
frozen environment. The software aim is a clear, testable package that preserves
identity, units and geometry when moving between those stages.

These notebooks are the general entry points. Older notebooks in `examples/`
remain as historical examples; `02_plot_md_statistics.ipynb` is a saved-report
viewer, not the recommended starting point for new data.

## Start here

For cluster jobs without notebooks, use the [postprocessing CLI and Slurm templates](../../docs/postprocessing_cli.md).
Its saved cache can also be loaded in Python for interactive analysis.

1. Install from the repository: `pip install -e ".[notebooks]"`. Use this
   environment as the notebook kernel; restart after updating the package.
2. Open the notebook you need and edit its **configuration cell**. Replace
   `/path/to/...` with your files. Windows forward slashes or raw strings work.
3. Run all cells. Each notebook has its own imports and configuration; no hidden
   state from another notebook is required. Computations call package functions.
4. Export is opt-in (`WRITE_FILES=True` or `SAVE_PLOTS=True`). Existing structure
   exports are protected. Real UMA inference is separately opt-in.

Each notebook includes visible **Mathematics** Markdown cells next to its
calculations: equations, variable definitions, units, and links between inputs
and outputs. Notebook 08 also maps every numerical configuration setting to its
role, including block rounding, bootstrap percentiles, relative thresholds and
the separate scalar/diffusion block rules. These document current implementation
behavior; a chosen tolerance is a screening assumption, not a physical constant.

| Notebook | Supply | Produces / next step |
|---|---|---|
| [01 Read and inspect](01_read_and_inspect_trajectory.ipynb) | Text dump, units, timestep; optional data file | Frame arrays, molecule inventory, streamed scalar history |
| [02 Initial configurations](02_build_initial_configurations.ipynb) | Whole molecular XYZ files, names/counts, cell | LAMMPS data + identity JSON + NPT/NVT/NVE templates |
| [03 Pairs and environments](03_select_pairs_and_environments.ipynb) | Raw frame, pair criteria or explicit molecule IDs | Pair/shell Extended XYZ, assessment and provenance |
| [04 Molecular statistics](04_molecular_statistics.ipynb) | Raw run directory, units/timestep, selected source-frame window | All-molecule MSD, correlations, optional diffusion and pair lifetimes |
| [05 Stability and structure](05_stability_and_structure.ipynb) | Raw run directory, selected source-frame window | Block statistics, batch errors, species COM RDFs |
| [06 Force reproduction](06_reproduce_uma_forces.ipynb) | Full periodic dump, local model, charge/spin, tolerance | UMA reproduction report; begin with one frame |
| [07 Reaction preparation](07_prepare_reaction_inputs.ipynb) | Your own `cluster.extxyz` from 03 | Frozen-shell ASE structure and row-identified force archive |
| [08 Diffusion and equilibration](08_diffusion_and_equilibration.ipynb) | Run directory or optional CLI cache; explicit source windows | Diffusive-regime checks, temporal-block uncertainty, scalar equivalence and structural comparisons |

Only 07 requires a derived structure, and 03 shows exactly how to generate it.
No notebook requires precomputed reports or bundled research results. Notebook 08
can optionally reuse your own CLI cache. No notebook runs LAMMPS or a reaction
optimization automatically.

Use 08 when assessing whether a diffusion coefficient is supportable. It compares
lag fits, temporal blocks, block sizes and source windows, and reports insufficient
sampling explicitly. Set scientifically meaningful absolute scalar tolerances;
inspect structure and slow observables before claiming equilibration. The notebook
documents missing thermo data, uncertainty assumptions, and all numerical screens.

Notebook 04 now caches pair topology. Start with `PAIR_WORKERS=1`; optional
process workers are not necessarily faster after caching. See
[performance and benchmarking](../../docs/trajectory_performance.md) for the
fixed-connectivity assumption, reference backend, and worker/batch controls.

### Access and save your own results

For notebooks 01, 03, 04, 05 and 06, set `RUN_DIR` to your simulation folder;
leave `INPUT_FILE`, `DUMP_FILE`, and `DATA_FILE` as `None`. The shared helper finds
`.lammpstrj`/`.dump`, `.data`, and `in.*`/`*.in` files:

```python
from pathlib import Path
from MartiniFuncs.trajectory_analysis import resolve_run_files

RUN_DIR = Path(r"C:\path\to\simulation")
files = resolve_run_files(RUN_DIR, require_input=False)
print(files)  # {"input": Path or None, "dump": Path, "data": Path or None}
```

Each file kind is searched directly in the folder first, then recursively if
absent there. Multiple matches require a relative or absolute filename override
(for example `dump_file="production.lammpstrj"`); optional files are not silently
discarded on ambiguity. Missing data/input files are allowed with
`require_input=False`, provided the reader has enough element/mass information
and you supply physical settings. Discovery neither executes input scripts nor
establishes that selected files describe the same run. Keep `UNITS` and
`TIMESTEP_PS` explicit. Notebook 02 instead starts from molecular templates;
07 starts from the cluster exported by 03.

Notebook 04 leaves `result` (molecular arrays and provenance), `frames` (scalar
DataFrame), and `window` (selected per-species lag statistics) in your kernel.
Notebook 05 leaves `rows` (block summaries), `nearest_rows` and each computed
`rdf`. These are ordinary Python/NumPy objects, not opaque saved-report formats.
For example, after running 04:

```python
frames.to_csv("frame_statistics.csv", index=False)
for species, data in window["species"].items():
    columns = {key: data[key] for key in
               ("lag_ps", "n_origins", "msd_angstrom2", "vacf", "force_acf")
               if key in data}
    pd.DataFrame(columns).to_csv(f"{species}_molecular_statistics.csv", index=False)
```

These explicit writes replace files of the same name. Keep the chosen units,
timestep, frame stride and analysis interval with exported results; changing a
plot label is not a substitute for rerunning with corrected physical settings.

## Inputs and conventions

- `UNITS` identifies the source LAMMPS unit style. `TIMESTEP_PS` is picoseconds per
  MD step, not between saved frames. Analysis supports `metal` currently;
  low-level readers support additional physical unit styles. Explicit settings
  bypass input-script parsing; there is no need for a restart header.
- Directory readers discover a unique text dump (`.lammpstrj`/`.dump`). For
  multiple files, choose `DUMP_FILE`. An explicit filename may use another suffix.
  `DATA_FILE` supplies element/mass information when the dump does not; use a
  corresponding data file, not an unrelated configuration with reused IDs.
  Optional `TYPE_SYMBOLS` maps integer types to elements.
- ASE positions are Å, stored source forces eV/Å, reported velocities Å/ps and
  times ps. Raw dump columns retain source units. Psience Molecule construction
  converts Å to Bohr; its figure and our overlays use Å. Normal vectors are
  dimensionless. Existing Psience bond inference remains unchanged.
- `lammps_id` is the source ID; `global_index` normally equals that ID minus one.
  Local indices are rows of a particular Atoms object. Slicing/reordering changes
  rows, not persistent identity. Pair force arrays are exported in geometry order.
- NPT/NVT/NVE inputs use `metal`: pressure bar, damping and timestep ps. Templates
  require the external fairchem launcher with local **uma-s-1p1 / omol** settings.
  The example run inputs are independent starts, not a validated compression
  protocol or exact restart. The packing example cell is intentionally dilute.

## What transfers to a new system?

Generic: reading, initial packing of multiple species, explicit pair/shell
extraction, geometry/vector alignment, molecular statistics, structure diagnostics,
and full-frame force comparison. Inspect inferred molecule sizes/formulas first.
Species statistics currently group by formula, so isomers require extra labeling.

Chemistry-specific: automatic candidate selection, reactive-site identification,
forming-bond pairing, angle criteria, plane overlays and candidate-pair lifetime
screening assume the supported CPD/dienophile system. In 03, use manual complete
molecule IDs plus an explicitly checked periodic shift for different chemistry;
its legacy CPD/MAL roles then mean first/second molecules, not chemical labels.
Automatic selection also rejects unsupported solvent species; manual extraction
does not have that selector limitation.

The MD analysis assumes fixed molecular membership. Existing bonds are checked,
but new reactions are not detected. COM diffusion requires fixed-center,
orthorhombic periodic cells and sufficiently frequent source frames for temporal
unwrapping. A sheared cell is not silently assigned the same diffusion formula.
Missing velocity/force columns omit those statistics. Pressure and potential
energy from thermo logs are not parsed yet.

## How the modules connect

See [module and data-flow guide](../../docs/workflow_api.md) for function inputs,
returns, index/units conventions and the boundary between analysis and proposed
reaction workflows. Functions can also be inspected with `help(function)`.

## Interpretation and validation

Software checks establish ordering, units, periodic reconstruction, packing
separation and interface behavior. They do not establish scientific validity.
For each run, inspect molecular identity, then temperature/density/structure and
sampling. Select source frames with `STARTFRAME`/`ENDFRAME` when needed (default: all), and
set temperature DOF explicitly. Fit diffusion
only after inspecting lag dependence. Candidate persistence refers to sampled
detections, not reaction rates; boundary-censored spans are incomplete lifetimes.

Before reaction work, reproduce forces on the full periodic frame. A finite pair
with total MD forces is not automatically a correct external-force model:
double-counting, force/gradient sign, work convention, and many-body environment
effects need reference calculations. Frozen-shell barriers require shell-size
and boundary checks and confirmation that the optimizer honors constraints.
Neither method supplies a transition state merely by selecting a close pair.

`cmcc-reactions` is an inspected downstream dependency, not modified here. Its
`ForceOptimizer.reoptimize_with_force` interface includes mode/magnitude/unit and
projection choices; an arbitrary source-force array is not assumed compatible.
Adapters and TS/barrier validation remain future work.

## Reproduce the notebook checks

`python -m unittest tests.test_workflow_notebooks tests.test_trajectory_analysis tests.test_stability`

Tests inject temporary user-like inputs at the configuration cell and execute
the notebooks from empty namespaces. They exercise exports and numerical checks.
UMA call wiring is mocked because weights are not distributed; passing tests
does not claim a real UMA reproduction or a reaction-barrier calculation.
