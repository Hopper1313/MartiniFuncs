# Workflow API and data flow

Purpose: build a reproducible UMA MD → molecular encounter → reaction-input
pipeline, while separating validated software behavior from untested physical
models. The new [workflow notebooks](../examples/workflows/README.md) use these
functions directly; examples are not alternate implementations of the algorithms.

See the [consistency review](consistency_review.md) for corrected bugs and remaining
legacy/scientific assumptions. Mathematical function docstrings give the equations,
input-variable definitions, and units; use `help(function)` or Jupyter `function?`.

For caching, optional process workers, and a reproducible benchmark, see
[trajectory performance](trajectory_performance.md).

For noninteractive simulation-to-cache-to-report workflows and Slurm templates,
see the [postprocessing CLI](postprocessing_cli.md). `postprocessing.cache_analysis`
calls `analyze_run`; `write_report` reuses cached arrays through the statistics
modules; `export_pairs` streams source frames through the extraction modules.

| Module / main functions | Inputs → outputs | Important behavior |
|---|---|---|
| `trajectory_analysis.resolve_run_files` | Run directory + optional filename overrides → `input`, `dump`, `data` paths | Direct matches take priority over recursive search; ambiguity raises with candidate names; absent optional files return None; no files executed |
| `lammpsio.iter_lammps_frames`, `read_lammps_frame` | Dump + explicit units + optional element/mass mapping → LammpsFrame(s) | Streams atomic data; sorts persistent IDs; aligns ASE geometry, source vectors and raw columns |
| `graph.build_bond_graph`, `connected_components`, `place_component` | Atoms + radius scale → molecular ID groups and whole coordinates | Distance-inferred connectivity; image shifts are integer lattice translations |
| `initial_configurations.MoleculeSpec`, `pack_molecules` | Whole ASE templates/counts + orthorhombic cell + seed/cutoff → packed Atoms | Random rigid insertion; periodic rejection; bounded failure, not minimization |
| `initial_configurations.write_initial_data` | Packed Atoms + filename/type ordering → data file + JSON metadata | Atomic style cannot store molecule IDs, so sidecar retains them; no automatic sidecar reload by dump reader |
| `lammps_inputs.render_lammps_input` | Ensemble, physical settings, type order, external force source → input text | No UMA weights or launcher guessed; `run` is last; no simulation executes |
| `geom_utils.get_all_prereactive_pairs` | Frame + cutoff → pair ID lists and shifts | CPD/dienophile screen; not a reaction prediction |
| `pair_selection.assess_prereactive_pairs` | Frame + pairs/shifts + optional PairCriteria → PairAssessment list | Reports geometry and reasons; `accepted` means chosen checks passed |
| `geom_utils.get_pair_shell_clusters` | Frame + complete pair IDs/shifts + radius → cluster records | Whole touched molecules retained; finite coordinates assembled as first molecule, second, shell |
| `get_cluster_positions_and_forces`, `pair_data_to_extxyz` | Cluster records + original frame → Extended XYZ | Preserves persistent IDs, roles, row-aligned vectors and source metadata |
| `geometry.align_geometry` | Finite mobile/reference structures → aligned copy, RMSD, transform | ID correspondence; rotates force/velocity vectors, never translates them |
| `visualization.to_display_molecule`, `plot_cluster`, `plot_pair_geometry` | ASE structures → Psience object/figure | Explicit Å→Bohr conversion; plot overlays in Å; chemistry-specific plane helper |
| `trajectory_analysis.analyze_run` | Run directory + units/timestep + frame controls → arrays/histories | Streams atoms; retains molecular COM, velocity/net-force arrays; no precomputed reports |
| `trajectory_pairs.FixedPairTopology`, `PairBatchRunner` | Initial graph + whole per-frame coordinates → ordered pair results | Internal fixed-topology cache; optional bounded process batches; shared screening/geometry equations |
| `molecular_window_statistics` | Analysis result + source-frame window + maximum lag → per-species MSD/correlations | All molecules and time origins; no drift subtraction; not a diffusion-convergence test |
| `md_statistics.get_molecular_diffusion_coefficient` | MSD result + explicit lag interval → slope/6 and diagnostics | 3D free-intercept fit; units Å²/ps and cm²/s; no confidence interval implied |
| `md_statistics.get_pair_lifetime` | Persistent pair IDs at sampled times → lifetime records | Continuous detection spans, first/last boundary censoring; one-frame span is unresolved |
| `stability.get_series_statistics`, `get_molecular_com_rdf` | Scalar history or wrapped COM/boxes → summaries/RDFs | Batch errors require decorrelation; RDFs use instantaneous volume, not TOR geometry |
| `force_validation.validate_forces` | Full dump + local model + state/tolerance → JSON-compatible report | Reads/evaluates chosen frames, explicit source units, no bundled weights |

## Structures passed between modules

**LammpsFrame:** `atoms` is an ASE object; `columns` contains original source-unit
columns; `timestep` is an integer MD step, not a physical time. `atoms.arrays`
holds IDs and source forces. Do not obtain source forces by attaching a new
calculator and calling `get_forces()`; use the stored `lammps_forces` array.

**Extracted structures:** `global_index` links local rows back to the source.
`pair_role` identifies first/second/shell atoms. A local index belongs to that
particular ordering. `forming_bonds_local` uses the extracted pair order and
does not survive arbitrary reordering. Alignment keeps row order and rotates
Cartesian vectors consistently.

**Trajectory result:** `frames` is the scalar history; `species` and
`molecule_atom_ids` establish fixed molecular ordering. Arrays have shape
`(retained_frames, molecules, 3)`:

- `com_angstrom`: temporally TOR-unwrapped COM for molecular MSD.
- `wrapped_com_angstrom`: actual centered-box COM for spatial structure/RDFs.
- `com_velocity_angstrom_ps`: mass-weighted molecular velocity.
- `net_force_eV_angstrom`: molecular sum of atomic source forces.
- `box_lengths_angstrom`: `(retained_frames, 3)` instantaneous lengths.

COM arrays are unavailable when the supported orthorhombic/fixed-center
convention fails. Do not wrap a TOR trajectory back into a changing box to
recover physical RDF coordinates. Use the separately retained wrapped array.
Velocities and net forces may be absent if their dump columns are missing.

`FRAME_STRIDE` reduces retained frames; temporal unwrapping still visits each
source frame. `MAX_FRAMES` limits source reading. `startframe`/`endframe` select
zero-based source indices in a half-open range; defaults select all available frames. Molecular storage is still O(frames × molecules).
Lag calculations are bounded by the requested maximum lag. Pair detection is
optional and can dominate runtime.

## Scope boundaries

Current package support does not imply general automatic reactive-site detection,
reaction-aware molecular identities, triclinic diffusion, pressure/log parsing,
dense packing, a validated external-force adapter or transition-state generation.
Gaussian/ORCA and legacy evaluator helpers remain in the repository, but the
documented active path uses UMA/fairchem throughout. No alternative potential is
introduced by this workflow.

Next validation steps: full-frame force reproduction; matched-geometry force
decomposition experiments; shell-radius/constraint convergence; controlled TS
reference calculations and reaction-path checks. Those are scientific studies,
separate from round-trip/indexing/unit tests.
