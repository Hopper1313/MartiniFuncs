# MartiniFuncs

Tools for extracting molecular encounters, source forces, and complete molecular
environments from LAMMPS trajectories. Transition-state generation and reaction
barrier validation are future pipeline stages.

## Package layout

| Subpackage | Contents |
|---|---|
| `MartiniFuncs.io` | `lammps` (dump/input reader), `lammps_inputs`, `thermo`, `psience` |
| `MartiniFuncs.structure` | `graph` (bond graphs, periodic neighbours), `periodic` (minimum image, TOR COM unwrapping), `molecules`, `alignment`, `packing` |
| `MartiniFuncs.trajectory` | `analysis` (`analyze_run`, `analyze_segments`), `parallel` (indexed frame chunks), `reactions` (topology-change events) |
| `MartiniFuncs.statistics` | `md` (molecular observables, FFT lag MSD/correlations), `dynamics`, `diffusion`, `stability` |
| `MartiniFuncs.pairs` | chemistry-independent pair machinery: `model` (`PairModel`, `PairCriteria`, registry), `assessment` (per-frame screen/measurement), `candidates` (cached candidates, `select_pairs`), `lifetimes`, `screening`, `structures` (pair/shell clusters, Extended XYZ), `topology` (cached fixed-topology evaluator) |
| `MartiniFuncs.chemistry.cpd_n4t` | the CPD + N4T Diels-Alder model: `model` (`CpdN4tModel`, `CpdN4tCriteria`), `pairs` (molecule classes, pair screen), `selection` (reactive sites and geometry), `visualization` |
| `MartiniFuncs.workflows` | `cli` (`python -m MartiniFuncs`), `postprocess` (cache/report/export), `progress`, `force_validation` |
| `MartiniFuncs.legacy` | superseded helpers kept for reference only |

Pair chemistry is a pluggable [pair model](docs/pair_models.md) (default `cpd_n4t`).
Common entry points are also importable lazily from the top level, e.g.
`from MartiniFuncs import analyze_run, select_pairs, CpdN4tCriteria`.

## Post-simulation CLI and Slurm

See the [CLI guide](docs/postprocessing_cli.md) for automatic directory discovery,
cached molecular statistics, pair/force/shell exports, reusable frame-window reports,
and Slurm job/job-array templates. Run `python -m MartiniFuncs --help`, or use
`martini-postprocess` after reinstalling the package to register the new command.

## Recommended workflow notebooks

Start with the [workflow guide](examples/workflows/README.md) and
[module/data-flow reference](docs/workflow_api.md). Seven independently runnable
notebooks cover raw trajectory inspection, initial packing, pair/shell extraction,
molecular MD statistics, stability/RDFs, local UMA force reproduction and
reaction-input preparation. They use your files and explicit settings, not saved
research reports. The older notebooks below remain available as examples.

## Analyze a simulation directory

Use [04_plot_md_statistics_from_directory.ipynb](examples/04_plot_md_statistics_from_directory.ipynb)
for new runs. Set `SIMULATION_DIR`; unique input and text dump files are discovered
recursively. Set `INPUT_FILE`, `DUMP_FILE`, or `DATA_FILE` when necessary to select
the correct files. This notebook uses `MartiniFuncs.trajectory.analysis` directly,
without any saved NPT reports. It plots scalar histories, species-resolved
**all-molecule COM MSD**, molecular velocity and net-force correlations, optional
diffusion fits, and optional CPD/dienophile pair lifetimes. It has no arbitrary
atomic-subset MSD diagnostic.

The default analysis uses the entire loaded trajectory. Select zero-based source
frames with inclusive `STARTFRAME` and exclusive `ENDFRAME` in the settings cell. `MAX_FRAMES` limits
source frames for a preview; `FRAME_STRIDE` controls retained frames. Unwrapping
still processes every source frame. Molecular arrays remain in memory, but
atomic frames are streamed. Pair screening is optional and chemically specific; the notebooks expose its
cached backend and worker controls. Diffusion fitting requires an explicit lag range.

This version requires metal units and a constant timestep. Set notebook `UNITS`
and `TIMESTEP_PS` directly (defaults: `"metal"`, `0.0005`). The package equivalent is
`analyze_run(directory, units="metal", timestep_ps=0.0005)`. Explicit values take
precedence; supplying both skips input parsing and header inference entirely,
and no input file is required. Optional `type_symbols={1: "C", 2: "H"}` supplies
element mapping when needed. Setting a physical setting to `None` requests
metadata fallback; units can also come from a LAMMPS `write_data` header.
`units_style` remains a compatibility alias. Binary restart files are not parsed
and timestep size is never guessed from dump steps. `timestep_ps` always denotes
ps per MD step, not the interval between saved frames.

Frame readers already accept `units=...`; force validation accepts it through
the Python API and CLI `--units`, bypassing input parsing when provided.
Force comparisons do not require timestep size. Molecular statistics accept
explicit `times_ps`, and run templates accept explicit `timestep_ps`.
Temperature DOF must
match the simulation (the notebook explicitly selects `3N-3` for unconstrained
atoms). Missing velocities/forces are reported. Molecular membership is fixed
from the initial bond graph and species are grouped by formula, so reactions
and isomer distinctions need additional handling. COM unwrapping (TOR) supports
any fully periodic cell: orthorhombic NPT, triclinic and sheared (`fix deform`)
cells, with a fixed or moving box center. Displacements are lab-frame, so an
imposed shear flow is not subtracted. LAMMPS thermo logs are parsed by
`MartiniFuncs.io.thermo` (see docs/thermo_output.md).

## Installation and tests

Use Python 3.10+ and `pip install -e .`. The reader is tested with ASE 3.29.0.
UMA validation additionally requires an appropriate fairchem environment
(tested API: fairchem-core 2.22.0, Python 3.11). Optional dependencies are
available through `pip install -e ".[uma]"`; no model weights are included.

From the repository root:

```sh
python -m unittest discover -s tests -p "test_*.py" -v
python -m tests.lammpsio_tests.test
python -m tests.lammpsio_tests.test --index 1 --max-axis-angle 45
```

The 45-degree example illustrates the filtering API; it is not a validated
reaction-readiness threshold. The example defaults to bond scale 1.3 and prints
diagnostics before optionally writing clusters with `--output clusters.extxyz`.

## Read synchronized frames

```python
from MartiniFuncs.io.lammps import iter_lammps_frames, read_lammps_input_settings

settings = read_lammps_input_settings("tests/lammpsio_tests/infile.in")
for frame in iter_lammps_frames(
    "tests/lammpsio_tests/multiple_frames.lammpstrj",
    units=settings["units"],
):
    atoms = frame.atoms
    forces = atoms.arrays["lammps_forces"]  # eV/Angstrom, source MD forces
    ids = atoms.arrays["lammps_id"]
    # Process and release each frame's results before advancing.
```

`read_lammps_frame(path, index=..., units=...)` returns one frame. Positive
indices scan forward; negative indices keep only the required trailing frames.
No split files are needed. Each bounded frame is read once from disk and parsed
for metadata and ASE geometry. This is not yet a persistent random-access index.

The new reader requires physical LAMMPS units. It preserves:

- Original IDs and `global_index = lammps_id - id_offset` (default offset 1).
- ID-sorted raw columns in `frame.columns`, in **original dump units**.
- Element/type information, explicit masses, molecule IDs and image flags when
  present; arbitrary numeric columns remain available in `frame.columns`.
- Cartesian coordinates, cell, origin, PBC, timestep and source path.
- Source forces in eV/Angstrom and velocities in ASE internal units.

`lammps_forces` is an array, not a live calculator result. It survives wrapping
and slicing but still refers to the original MD snapshot. No placeholder energy
or calculator is attached. Translation by cell vectors does not rotate forces;
any later rotation of geometry must also rotate force/velocity vectors.
Coordinates are not wrapped by default. Image flags are retained, not
automatically applied; molecular extraction unwraps through the bond graph.

Explicit dump elements are preferred. A `type_symbols={1: "C", ...}` mapping is
checked against them. Data-file or nearest-mass inference is a fallback; use
explicit mappings for unusual masses or coarse-grained particles. The input
settings helper reads literal commands and continuations, not executable LAMMPS
logic. It rejects unresolved settings/includes/branches rather than guessing.
Model, task, charge, and spin are not inferred from comments.

## Inspect and filter encounters

```python
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_all_prereactive_pairs
from MartiniFuncs.pairs.structures import (get_pair_shell_clusters, get_cluster_positions_and_forces,
                                           pair_data_to_extxyz)
from MartiniFuncs.chemistry.cpd_n4t import CpdN4tCriteria
from MartiniFuncs.chemistry.cpd_n4t.selection import assess_prereactive_pairs

pairs, shifts = get_all_prereactive_pairs(frame.atoms, scale=1.3)
reports = assess_prereactive_pairs(frame.atoms, pairs, shifts, scale=1.3)
# Inspect reports before choosing thresholds.
criteria = CpdN4tCriteria(max_axis_angle=45, max_forming_distance=3.5)
reports = assess_prereactive_pairs(frame.atoms, pairs, shifts,
                                  criteria=criteria, scale=1.3)
keep = [r.pair_index for r in reports if r.accepted]
clusters = get_pair_shell_clusters(
    frame.atoms, [pairs[i] for i in keep], [shifts[i] for i in keep], scale=1.3,
)
records = get_cluster_positions_and_forces(frame.atoms, frame, clusters)
if records:
    pair_data_to_extxyz(records, "clusters.extxyz")
```

`get_all_prereactive_pairs(..., criteria=criteria)` is a convenience filter with
the original `(pairs, shifts)` return type. Keep the separate assessment step
when rejected candidates and reasons are needed. Default pair selection remains
a distance screen, with a stricter check that the dienophile site has two atoms.

Optional diagnostics validate CPD/dienophile site patterns and report:

- The two forming-bond distances, evaluating both endpoint matchings and choosing
  the one minimizing the larger distance (then their sum, then IDs).
- Distance imbalance, useful for characterizing asynchronous approaches.
- Axis angle between the diene termini and the dienophile C=C.
- Angle between fitted reactive carbon planes.
- Approach angle between site separation and the diene-plane normal.

Angles are unsigned, 0--90 degrees; they do not distinguish endo/exo or establish
stereochemistry. No numerical thresholds are enabled by default. Ambiguous or
degenerate sites produce rejection reasons. This remains specific to the current
11-atom CPD / 23-atom dienophile chemistry and distance-inferred connectivity.
At scale 1.5 the supplied frame has an overconnected CPD bridge in one candidate;
at scale 1.3 both sites are unambiguous. Component counts alone miss this issue.

The exported cluster is nonperiodic; source cell/PBC and units remain metadata.
Stored source forces are **not** finite-cluster predictions or environment-only
forces. Whole shell molecules can extend beyond the nominal shell radius.

## Reproduce UMA forces locally

The user-facing workflow lives in `MartiniFuncs.workflows.force_validation`, as both the
`validate_forces()` Python API and a command-line module. It compares full
periodic snapshots before extraction or relaxation. Tests under
`tests/test_force_comparison.py` only exercise the comparison code with an
analytic calculator; they do not run or require UMA.

Provide a local **uma-s-1p1** checkpoint, a LAMMPS dump containing `fx fy fz`, and
optionally the LAMMPS input and data files. `input_file` supplies literal units
and provenance; `datafile` supplies fallback species mapping. Checkpoint weights
are never included or downloaded. The model has been confirmed as uma-s-1p1;
the uma-s-1p2 comment in the supplied input is not used as configuration.

The confirmed MD inference settings are applied explicitly by default:

```json
{
  "tf32": true,
  "merge_mole": true,
  "compile": true,
  "activation_checkpointing": false,
  "internal_graph_gen_version": 2,
  "external_graph_gen": false
}
```

Other settings retain the installed fairchem defaults. The report records the
full requested settings and dependency versions. Matching these six options
alone does not establish equivalence between different fairchem versions or
CPU/GPU hardware. CUDA is the default to match GPU inference; `--device cpu` is
available, but TF32 is a GPU option and CPU timing/numerics are not equivalent.

For example, **if the full system is neutral and singlet**, from the repository
root:

```sh
python -m MartiniFuncs.workflows.force_validation tests/lammpsio_tests/multiple_frames.lammpstrj --input tests/lammpsio_tests/infile.in --checkpoint /path/outside/repo/uma-s-1p1.pt --charge 0 --spin 1 --atol 0.001 --output force_report.json
```

The default checks only frame 0 using `omol`. Add `--indices 0 4 8` to check
several frames, or `--index 4` for a single different frame. Indices are zero-based,
nonnegative frame numbers, not timesteps. Frames are processed in file order,
using one calculator and bounded trajectory memory. A missing requested frame
raises an error; duplicate indices are rejected.

The same operation is available from Python:

```python
from MartiniFuncs.workflows.force_validation import validate_forces

report = validate_forces(
    dump="tests/lammpsio_tests/multiple_frames.lammpstrj",
    checkpoint="/path/outside/repo/uma-s-1p1.pt",
    input_file="tests/lammpsio_tests/infile.in",
    indices=[0],  # start with one; increase after inspecting timing
    charge=0, spin=1,  # use the actual full-system charge and multiplicity
    atol=0.001,       # illustrative absolute tolerance in eV/Angstrom
)
print(report["passed"], report["frames"])
```

Charge/spin and the tolerance above are examples, not inferred from the input.
An optional `inference_settings` mapping overrides the confirmed defaults.
The CLI accepts a JSON overrides file or fairchem preset name with
`--inference-settings`. A preset explicitly replaces the MD defaults; a mapping
merges with them. JSON may wrap the options under `if_settings` and contain the
standard `InferenceSettings` `_target_`; no Hydra code is executed.

The report contains per-frame RMSE/MAE, maximum component and atom-vector errors,
worst atom ID, net forces, and pass/fail. It also records model-load time,
per-frame evaluation time, total elapsed time, and checkpoint SHA256. The first
evaluation can include compilation and warm-up, so it may be much slower than
subsequent frames. Successful completed runs return status 0 only if every
component in every selected frame meets the tolerance; mismatches return 1.

No force projection or geometry relaxation is performed. Dump rounding, model
version, task, inference precision, and force-modifying LAMMPS fixes can affect
agreement. Energy offsets are not loaded because only forces are compared.
Actual UMA force reproduction still requires the local checkpoint and physical
configuration; analytic unit tests do not establish it. A matching result also
does not validate barriers or justify adding total MD forces to a pair.

## Extraction and visualization notebook

Open `examples/01_extract_and_visualize.ipynb` with a kernel where this package
is installed (`pip install -e ".[notebooks]"`). Start Jupyter from the repository root or
`examples/`. The notebook reads the supplied multi-frame fixture, reports pair
geometry and rejection reasons, extracts one complete molecular shell, and
displays role colors and source-force arrows. It needs no UMA checkpoint and
writes no output files. X3D requires a trusted notebook and access to its viewer
JavaScript assets. The notebook kernel must have IPython/Jupyter available.

Set `DATASET = "shear"` in the notebook configuration cell to use
`tests/lammpsio_tests/more_test_data`. That dump has type IDs but no elements or
masses, so the example passes `2_337_before.data` as `input_datafile` for species
and MD masses. The settings reader accepts an initial `clear`; a `clear` after
simulation settings still requires explicitly resolved, frame-specific input.
For a quick check on the large shear dump, use frame 0: later positive indices
scan forward and negative indices scan to the end. The notebook regression test
exercises frame 0 when this optional dataset is available.

`chemistry.cpd_n4t.visualization.to_display_molecule(atoms)` converts geometry to Psience Bohr
coordinates without an energy evaluator: 1 Angstrom is approximately 1.889726
Bohr. ASE objects retain their original Angstrom coordinates. Conversion back
with `io.psience.molecule_to_ase_atoms` applies Bohr to Angstrom. The legacy
`io.psience.ase_atoms_to_molecule` retains its existing evaluator behavior.
`chemistry.cpd_n4t.visualization.plot_cluster` colors CPD green, MAL orange, and shell atoms gray;
optional red arrows represent `lammps_forces` in eV/Angstrom. `force_scale=0.2`
means a 1 eV/Angstrom force is drawn as a 0.2 Angstrom displacement. Forces are
never normalized, evaluated, or interpreted as environment-only forces.
The arrow displacement is then converted to Bohr (approximately 0.377945 Bohr
in this example) before passing it to Psience `mode_vectors`. This is a display
length conversion, not a physical force conversion to Hartree/Bohr.

Bonds use Psience `Molecule` inference and plotting defaults. The extraction
bond graph is not substituted for `mol.bonds`. Use `bonds=False` for full
simulation-cell views. For guessed reaction-forming bonds, use
`to_display_molecule`, append the local-index bonds to `mol.bonds`, and call
`mol.plot`; retain its existing inferred bonds.

`structure.alignment.align_geometry` returns `(aligned, rmsd, transform)` and matches fit
atoms by `global_index` when both structures provide it. Its transform uses
`positions @ rotation + translation`; vectors use only `@ rotation`.
`lammps_forces`, `forces`, `velocities`, and ASE `momenta` arrays rotate by default.
Additional Cartesian vector arrays must be named explicitly. Source metadata
stays in the original frame; live calculator results are not copied. Alignment
requires finite, unwrapped structures, not periodic frames. Collinear fits
cannot determine a unique orientation.

The existing `legacy.geom_utils.align_atoms` keeps row correspondence and its two-value
return, but now rotates stored vectors and rejects periodic inputs. Extract a
finite cluster before using it. Neither alignment API reorders atoms.

## Legacy I/O compatibility

`make_ase_obj_and_fix_atomtypes()` now selects geometry and metadata from the same
frame. Its old implicit `metal` default emits a warning; pass `units` explicitly.
`my_readlammps()` retains its raw four-item tuple and last-frame default, with a
new `index` keyword. Raw tuples preserve raw units; new `LammpsFrame` extraction
uses normalized vectors. Mixing frame timesteps is rejected. Velocities are
optional in normalized cluster/extxyz extraction; the legacy custom XYZ writer
still expects them. `split_lammpstrj()` is retained and now writes every frame.

`read_input_datafile()` retains its six-item return structure using ASE parsing;
its coordinates/masses use ASE units and type metadata covers populated types.
No changes to `cmcc-reactions` are part of this milestone.


## MD statistics

`MartiniFuncs.statistics.md` separates spatial scope in its public names:

- Molecule: `get_molecule_center_of_mass`, `get_molecule_com_velocity`,
  `get_molecule_net_force`, `get_molecule_orientation`, `get_molecule_rmsd`,
  `get_molecule_displacement`.
- Pair: `get_pair_relative_velocity`, `get_pair_force_projections`,
  `get_pair_geometry`, `get_pair_geometry_timeseries`, `get_pair_lifetime`.
- Whole frame: `get_frame_kinetic_energy`, `get_frame_temperature`,
  `get_frame_density`, `get_frame_force_statistics`, `get_frame_rmsd`,
  `get_cell_deformation`.
- Selected ensembles over a trajectory/window: `get_molecular_msd`,
  `get_atomic_msd`, `get_molecular_diffusion_coefficient`,
  `get_molecular_velocity_autocorrelation`, `get_atomic_velocity_autocorrelation`,
  `get_molecular_force_autocorrelation`, `get_atomic_force_autocorrelation`.

Molecule selectors are persistent `global_index` IDs, not array row numbers.
Frame/molecule functions accept ASE Atoms (`frame.atoms` for LammpsFrame).
Coordinates are Angstrom, time is ps, returned COM velocities are Angstrom/ps,
forces are eV/Angstrom, kinetic energy is eV, temperature is K, and density is
 g/cm³. ASE velocities are converted to Angstrom/ps; raw dump columns must not
be substituted for ASE momenta. Temperature requires explicit
`degrees_of_freedom`, accounting for constraints and removed motion.
`remove_com=True` subtracts uniform translation, not a shear-flow profile.

Periodic molecule COM/orientation/RMSD requires whole unwrapped molecules:
`positions_unwrapped=True` declares that the caller has done this. Displacements
and MSD additionally require *temporal* unwrapping. The module does not infer
multi-cell crossings, correct deformation, remove drift, or rigidly align MSD
inputs. These choices must be made before analysis. The molecule orientation
function returns a specified two-atom axis, not a unique three-dimensional frame.

### Pair lifetimes from a dump

```python
from dataclasses import asdict
from MartiniFuncs.io.lammps import iter_lammps_frames, read_lammps_input_settings
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_all_prereactive_pairs
from MartiniFuncs.statistics.md import get_pair_lifetime

settings = read_lammps_input_settings("tests/lammpsio_tests/infile.in")
# This fixture uses metal units: timestep is in ps.
pairs_by_frame, times_ps = [], []
for frame in iter_lammps_frames(
    "tests/lammpsio_tests/multiple_frames.lammpstrj", units=settings["units"]
):
    pairs, shifts = get_all_prereactive_pairs(frame.atoms, scale=1.3)
    pairs_by_frame.append(pairs)
    times_ps.append(frame.timestep * settings["timestep"])

episodes = get_pair_lifetime(pairs_by_frame, times_ps)
for episode in episodes:
    print(asdict(episode))
```

This retains candidate identities, not trajectory coordinates. Candidate pair
order and atom row order may change between frames. Each episode is identified
by two atom-ID sets. Keep the same selection criteria across the window;
filtered accepted-pair lists can be supplied instead of distance-screened lists.
Changing molecular membership defines a different identity.

An episode reports first/last detection, detection count, observed span in ps,
and left/right censoring. A one-frame detection has an observed span of zero;
its physical duration is unresolved. A boundary encounter is not a completed
lifetime. Actual entry/exit occurs between snapshots. Do not interpret the mean
of these spans as an unbiased mean lifetime. `max_gap_frames=0` gives continuous
detections; a positive value bridges that many absent samples and defines an
intermittent encounter. This is a sample-count rule even for irregular times.
For a trajectory slice, boundary censoring refers to that slice. Missing or
unprocessed frames must not be encoded as empty candidate lists.

### Motion and correlation arrays

Trajectory functions accept `(frames, selected_objects, 3)` arrays with a fixed
object identity/order and explicit `times_ps`. MSD/correlation requires uniform
sampling and averages over all available time origins and selected objects at
each lag. Results retain the required species/group `selection` label and sample
counts. These functions store arrays in memory and use direct O(T² N) lag sums;
use a suitable time window for large trajectories.

```python
from MartiniFuncs.statistics.md import (
    get_molecular_msd, get_molecular_diffusion_coefficient,
)
# com_positions_unwrapped: COM positions in Angstrom, one column per CPD.
# msd = get_molecular_msd(com_positions_unwrapped, times_ps,
#                         unwrapped=True, selection="CPD")
# fit = get_molecular_diffusion_coefficient(msd, fit_start_ps=10, fit_end_ps=50)
```

Choose a physically justified diffusive interval; the example interval is not a
recommendation for the short fixture. The estimator fits the three-dimensional
Einstein slope `6 D`, returning Angstrom²/ps and cm²/s plus fit diagnostics. It
does not establish diffusive behavior or uncertainty. A ballistic trajectory or
negative fitted slope is not a valid diffusion measurement. Report each species
separately. Lower-dimensional diffusion and diffusion tensors are not implemented.

Velocity correlations expect Angstrom/ps, force correlations eV/Angstrom. They
are dot products, not per-component correlations. Mean subtraction is optional
and subtracts each object's temporal mean. Normalization divides by lag zero;
zero-variance normalization raises an error. Molecular force correlations use
molecular resultant forces. Neither total MD forces nor their correlations are
automatically environment-only forces.

### Preliminary NPT molecular diffusion

Run `python -m examples.analyze_npt_diffusion` to calculate whole-molecule COM
trajectories and species-resolved MSD fits for the 1 ns NPT fixture. The workflow
supports orthorhombic boxes with a fixed box center and fixed initial molecular
topology (`analyze_run` itself supports any fully periodic cell). It reconstructs
whole molecules before wrapping their COMs into a centered cell and applying
`structure.periodic.unwrap_orthorhombic_com`. This uses the
TOR algorithm (Bullerjahn et al., DOI: 10.1021/acs.jctc.3c00308), not atomwise
image-flag trajectories. Sufficient temporal sampling remains an assumption.
All initial bonds are checked for excessive stretching during reconstruction.

Results and reusable COM trajectories are in `reports/npt_statistics/diffusion`.
Run `python -m examples.check_diffusion_windows` to reuse that cache for the
last half (500-1000 ps) and last quarter (750-1000 ps). This saves
`late_windows.json`, comparing identical 10-50, 25-75, and 50-100 ps lag fits.
The plotting notebook now uses these late windows; the original
`diffusion.json` retains the earlier exploratory analysis.

For the 50-100 ps lag fit, last-half/last-quarter estimates are 5.84e-9/2.70e-9
cm²/s for CPD and 1.81e-9/1.65e-9 cm²/s for the dienophile. No drift or finite-size
correction is applied. These nested windows are not independent replicates;
fit-window variation is not a confidence interval. Early equilibration is
excluded, but the remaining dependence on trajectory and lag windows should
be assessed before selecting an equilibrium coefficient. See the plots and
full numerical comparison in `examples/02_plot_md_statistics.ipynb`.

### Late-time stability diagnostics

Run `python -m examples.analyze_npt_stability` after regenerating the COM cache
with `python -m examples.analyze_npt_diffusion`. The cache now retains the actual
wrapped molecular centers as well as TOR coordinates; NPT structural analysis
must use the former. The report uses four disjoint 125 ps blocks from 500 ps
through, but excluding, 1000 ps (250 frames each).

`MartiniFuncs.statistics.stability.get_series_statistics` provides scalar summaries,
first-zero-truncated autocorrelation diagnostics, and batch-mean standard errors
at multiple averaging durations. `get_molecular_com_rdf` computes species COM
RDFs and nearest-neighbor distances in fully periodic (orthorhombic or triclinic) cells. RDFs
use each frame's volume and exclude self pairs; same-species normalization uses
the finite population of available pairs. Distances and box lengths are in
angstrom. These diagnostics do not return an equilibrium pass/fail verdict.

The generated `reports/npt_statistics/stability/README.md` contains scalar and
structural plots; `stability.json` stores numerical results. Energy, pressure,
reactive orientations, and independent replicas are not included in this report.

### Building initial configurations

Start with [examples/03_build_initial_configurations.ipynb](examples/03_build_initial_configurations.ipynb)
for a complete 40+40 example with editable composition/cell settings, Psience
views, periodic contact checks, and data/run-template export. Set `WRITE_FILES=True`
in the setup cell to enable export; existing files remain protected by default.

`MartiniFuncs.structure.packing` accepts whole nonperiodic ASE molecule
templates (coordinates in Angstrom), copy counts, and an orthorhombic periodic
cell. It randomly rotates and translates rigid copies and rejects intermolecular
atom contacts below `min_distance`, including contacts across box boundaries.
It supports any number of species, not just CPD/dienophile.

```python
from ase.io import read
from MartiniFuncs.structure.packing import (
    MoleculeSpec, pack_molecules, write_initial_data,
)
from MartiniFuncs.io.lammps_inputs import render_lammps_input
from pathlib import Path

a = read("molecule_A.xyz")  # Whole molecular geometry, Angstrom
b = read("molecule_B.xyz")
atoms = pack_molecules(
    [MoleculeSpec("A", a, 40), MoleculeSpec("B", b, 40)],
    cell=[50, 50, 50], seed=24673, min_distance=2.0,
)
metadata = write_initial_data(atoms, "initial.data")
text = render_lammps_input(
    data_file="initial.data", type_symbols=metadata["type_symbols"],
    ensemble="npt", steps=100000, force_source="external_uma",
    temperature_K=400, pressure_bar=25000, output_prefix="npt",
)
Path("npt.in").write_text(text)
```

Packing is not minimization or equilibration. The uniform contact cutoff is
user-selected, not an MLIP energy criterion. Random insertion is best suited to
dilute starting boxes; dense requests can raise `PackingError` after a bounded
number of trials. No partial mixture is returned. A conservative molecular
bounding-sphere check excludes possible self-image overlaps. Triclinic packing,
charged systems, dense-packing optimization/Packmol, and density-based cell
design are not implemented yet. Templates must already be chemically meaningful
and whole; the packer does not reconstruct wrapped input molecules.

`write_initial_data` writes `units metal`, `atom_style atomic` data with masses,
element type ordering, and image flags. Molecule identities are retained in
`initial.data.json` because atomic style has no molecule-ID column. This sidecar
is not automatically loaded by the trajectory reader. File atom IDs are newly
assigned consecutively from one. There are no force-field bond terms, charges,
or initial velocities. One mass per element is required. Existing files are
protected unless `overwrite=True`. Packed ASE positions are wrapped; whole
coordinates are `atoms.positions + atoms.arrays['image_flags'] @ atoms.cell.array`.

Generate a concrete 40 CPD + 40 dienophile example with
`python -m examples.build_initial_configuration`. This writes a data file,
identity sidecar, ASE Extended XYZ, and `npt.in`, `nvt.in`, `nve.in` under
`reports/initial_configuration` (use `--output` for a fresh directory).
The example's 50 Angstrom box is intentionally dilute, not a recommended target
density. Its 400 K / 25000 bar NPT parameters illustrate the existing experiment;
they do not validate rapid compression from that dilute state.

Run templates require the collaboration's **external fairchem driver**, the
local **uma-s-1p1** model, and task **omol**. No driver invocation, model path,
charge/spin settings, or inference settings are guessed or embedded in LAMMPS
syntax. These must be configured in the existing launcher. The templates omit
`pair_style` intentionally and cannot supply UMA forces in standalone LAMMPS.
Each template is an independent fresh start; do not interpret them as a staged
equilibration protocol. `initialize_velocities=False` preserves velocities only
if the input data file actually contains them. It is not an exact restart.
NVE has no thermostat/barostat; initial temperature only sets velocities. Time
and damping are ps, pressure is bar. `run` remains the final executable command,
with final data/restart writes scheduled inside it. Dump fields match the
trajectory/force-analysis pipeline. NVT is also suitable as a configurable
production template; NVE supports energy-drift tests once the UMA driver is
attached and the starting state is prepared.

Interfaces follow [ASE LAMMPS data I/O](https://docs.ase-lib.org/_modules/ase/io/lammpsdata.html)
and [LAMMPS thermostat/barostat commands](https://docs.lammps.org/fix_nh.html).
