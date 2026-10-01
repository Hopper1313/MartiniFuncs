# Package consistency review

Reviewed the active `MartiniFuncs` modules, their tests, and the current workflow
entry points. Equations in function docstrings now identify input variables,
averaging dimensions, units, and coordinate conventions. Historical example
scripts are labeled where they still require fixed datasets/cached reports.
The separate `mlip-demo` directory and `cmcc-reactions` were not modified.

## Corrected issues

| Location | Issue and correction |
|---|---|
| `coreio.check_if_internal_part_of_multiple_fragments` | An intersection containing only atom 0 evaluated as false. Test whether the intersection is nonempty. |
| `coreio.get_symbol_from_mass` | The nearest-mass search included ASE's dummy element X. Exclude it and reject nonpositive/nonfinite masses. |
| `orcaio._validate_constraint_in_optimization` | The discrepancy flag was reset immediately before returning, always reporting success. Preserve failures detected in the selected frames. |
| `shell.build_bond_graph` | `periodic=False` still searched periodic images, allowing duplicate neighbors and nonzero image shifts. Disable PBC during the search and return zero shifts. |
| Legacy shell centering/extraction | `repair=False` returned None; final shell repairs ignored the supplied bond scale/skin. Return the shifted Atoms and honor both parameters. |
| `md_statistics.get_molecule_displacement` | A generator of atom IDs was exhausted by the first COM calculation. Materialize it once for both frames. |
| `trajectory_analysis.analyze_run` | Striding could hide timestep resets in skipped frames. Check every source frame before downsampling. |
| `molecular_window_statistics` | Uniform-time validation was looser than direct MSD/correlation calls. Use the same timestamp validator. |
| `force_validation.compare_frame_forces` | Invalid source forces could produce NaN report metrics. Reject nonfinite, empty, or incorrectly shaped source arrays. |
| `geom_utils.pair_data_to_extxyz` | Removed a duplicate metadata-copy loop; output is unchanged. |

Public signatures remain unchanged. Invalid inputs now fail earlier; repaired
legacy functions can give different results precisely in the cases above.
No pair thresholds, bond display behavior, or scientific models were changed.

## Remaining inconsistencies and assumptions

1. **Legacy shell indexing remains incompatible with arbitrary IDs.**
   `shell.get_idxs_for_molid` computes ID minus one and downstream functions
   use those values as ASE rows. This requires consecutive IDs in sorted order.
   `shell.process_lammps_frame` also assumes a fixed column set and three values
   per box-bound line. Use `lammpsio` plus `graph`/`geom_utils` in new workflows;
   replacing these legacy interfaces needs a deliberate compatibility migration.

2. **The legacy evaluator helper is not UMA integration.**
   `coreio.ase_atoms_to_molecule` defaults to AIMNet2 and falls back to AIMNet2
   even for other evaluator names. Its coordinate conversion is correct, but
   its evaluator API needs redesign before reuse for calculations. Use
   `visualization.to_display_molecule` for display and `force_validation` for UMA.

3. **Pair screening and angle measurements use different centers.**
   `get_all_prereactive_pairs` screens whole-CPD COM to dienophile C=C COM.
   `measure_pair_geometry` measures approach from the CPD termini midpoint
   toward the C=C midpoint. Both definitions are now explicit. Changing the
   screen would change historical candidate counts and needs scientific testing.
   Automatic site identification remains CPD/specific-dienophile chemistry.

4. **A temperature DOF correction is not a velocity correction.**
   `analyze_run(degrees_of_freedom="3N-3")` uses that denominator but does not
   subtract COM velocity. The low-level function offers `remove_com=True`.
   Flow/shear requires an explicit streaming-velocity convention; neither DOF
   subtraction nor whole-frame COM subtraction removes a velocity profile.

5. **Trajectory statistics assume fixed molecular membership.**
   Molecules are inferred from the initial bond graph, and species are grouped
   by formula (isomers merge). Existing-bond checks do not detect every new bond.
   TOR MSD requires fully periodic cells and sufficiently frequent sampling; RDF
   requires actual wrapped geometry. Since the 2026-10-01 refactor a moving box
   center or a triclinic/sheared cell no longer disables either array (see
   docs/refactor_changelog.md, step 1).

6. **Raw legacy vectors and normalized frame vectors are different APIs.**
   `my_readlammps` retains source units; `LammpsFrame` stores eV/Angstrom forces
   and ASE internal velocities. Extended XYZ copies vectors and metadata without
   conversion. Its `velocities` array is not automatically ASE `momenta`.
   For molecular velocity statistics, use the original frame or explicitly
   restore ASE velocities with the recorded unit convention.

7. **Legacy ORCA validation is numerical, not a complete constraint audit.**
   It compares internals against the first geometry, not explicit input targets,
   and does not wrap dihedral differences across +/-pi. The constraint writer's
   boolean convention is also unusual: false entries select bond constraints;
   angle/dihedral flags are not honored. These paths need dedicated modernization
   if they become part of the reaction workflow.

8. **Historical scripts still depend on old data.**
   `examples/analyze_npt_*.py` and `check_diffusion_windows.py` use fixed paths,
   windows, or generated caches. Prefer `examples/workflows/` for portable runs.
   File discovery selects paths; it cannot establish that files with matching
   atom IDs actually describe the same simulation.

## Reading the equations

Use `help(function)` in Python or `function?` in Jupyter. The most useful starting
points are `get_molecular_msd`, `get_molecular_diffusion_coefficient`,
`get_series_statistics`, `get_molecular_com_rdf`, `measure_pair_geometry`, and
`align_geometry`. Vectors use row coordinates: lattice translation is `n @ cell`
and rigid alignment is `r @ rotation + translation`.

Regression coverage includes synthetic geometry/force checks, source-file parsing,
pair extraction, packing, statistics, and workflow notebook execution. These checks
do not establish UMA force reproduction, equilibrium, or reaction-barrier accuracy.
No licensed model inference or ORCA calculation was run during this review.
