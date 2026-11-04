# Hand-off: anthracene + MA pair builder page with optimizer (as of 2026-11-03)

Repository: `C:\Users\nicho\Documents\Martini collab\MartiniFuncs`, branch `claude_refactor`
(never open PRs to master). Conda env for tests: `C:\Users\nicho\anaconda3\envs\mlip_uma\python.exe`
(`python -m unittest discover -s tests -p "test_*.py"`: 224 tests, all pass, 1 skipped).

Committed through `9f7cbf7`. **Uncommitted (2026-11-03, second session):** saved results,
the product reference and forming bonds (see "Added in the second session" below) in
`build.py`, `viewer.py`, `optimize/server.py`, the two test files, `containers/mlip/` (mlip.def,
README) and `docs/workflow_guide.md`. `examples/04_old_backup.ipynb` stays untracked on purpose.

## What exists

### Builder (Python)
`MartiniFuncs/chemistry/anthracene_ma/`
- `sites.py`: graph-based sites. Anthracene: termini (C9/C10) + central ring (6 C).
  Maleic anhydride (MA): alkene, ring carbons (4 C), bridging O.
- `build.py`:
  - `PairFrames`: templates in builder frames (default: UMA-relaxed molecules shipped as package data).
  - `build_pair(distance, distance_kind="height"|"centers"|"max_forming", offset, twist, roll, fold)`
    → `(Atoms, DielsAlderGeometry)`. Rows: anthracene 0–23, then MA 24–32; `pair_role` "ANT"/"MA".
  - `measure_placement(atoms)`: the inverse; returns the simplest of the symmetry-equivalent descriptions.
  - `structure_metrics(atoms)`: the shared Diels-Alder metrics, screen distance and closest contact.
  - `reference_structure(name)`: UMA-s-1p2p1 hand-off structures (charge 0, spin 1), names
    `anthracene`, `maleic_anhydride`, `reactant_complex`, `transition_state`, `product`.
    The 33-atom ones carry `pair_role` from the hand-off mapping (`REFERENCE_MA_ROWS`), so the
    product works with `structure_metrics`/`measure_placement`. `reference_energy(name)`: UMA energy (eV).
  - `to_builder_frame(atoms, frames)`: any 33-atom anthracene + MA structure reordered to the builder rows
    (graph isomorphism per molecule, the symmetry-equivalent match closest to the template) and expressed
    in the anthracene frame fitted to it; `source_row` array gives the input row.
- Frame convention:
  - **Anthracene frame:** origin at the central-ring centroid; x = long axis; y = C9→C10; z = ring normal.
    When the anthracene is bent (TS, adduct), z points away from the outer rings, i.e. toward the MA.
  - **Placement:** R = Rz(twist) Rx(roll) Ry(−fold), then the MA C=C midpoint goes to (offset_x, offset_y, height).
  - **All angles 0:** stacked, reactive pose.
  - **MA body frame:** its normal is set by the vinyl-H bend when the MA is bent.
- Reference placements: reactant complex = stacked at 3.17 Å, fold 1.5°; TS = stacked at 2.33 Å, fold 15.2°.

### Page
`chemistry/anthracene_ma/viewer.py`
- **API:** `builder_page()` returns the HTML; `write_builder_html(path)`; CLI
  `python -m MartiniFuncs.chemistry.anthracene_ma.viewer OUT.html`. The static page is `reports/anthracene_ma_pair_builder.html`.
- **Rendering:** McUtils/Psience X3D (x3dom from x3dom.org, so it needs internet). The two molecules are
  rendered **separately** (`scene()`), so there are no inter-molecular bond cylinders.
- **Builder mode:** sliders drive one X3D `Transform` (`#anhydride-placement`) around the MA scene, a rigid placement.
- **Optimization frames:** every atom moves individually.
  - `scene()` ties each McUtils sphere to its atom.
  - Each bond cylinder is tied to its two atoms: `i, j` (atoms), `t` (centre fraction along the bond),
    `s` (height / bond length), `o2, o3` (sideways offset of double-bond cylinders, in a frame oriented by
    neighbour `k`).
  - JS `showPerAtom(P)` rebuilds translation, rotation and height from coordinates.
    `showRigid()` restores the stored attributes when returning to the builder.
- **Forming C···C pairs** (minimax matching) are drawn as an orange `LineSet` (`#forming-coords`), not as bonds.
- **Metrics:** JS `metrics(P)` reimplements `structure_metrics` for any 33-atom geometry. Plane normals come
  from a 3×3 Jacobi eigen-solve.
- **Atom selection:** clicking a sphere calls `pickAtom(i)`, injected as `onclick` on the sphere Shapes.
  Selected atoms are highlighted through the material `emissiveColor`. The default selection is the four
  forming carbons.
- **Optimize panel:**
  - Settings: model, constraint (fix forming distances [default] / restrain them, harmonic k / fix selected
    atoms / none), optimizer (BFGS, LBFGS, FIRE, Sella minimum, Sella TS), fmax, steps, charge, spin.
  - It optimizes the *shown* geometry, which can be a builder pose or an optimization frame.
  - It polls the job, streams frames, plots an energy trace, and has a frame slider and "Back to builder".
  - Exports: the shown XYZ, or the whole path as multi-frame XYZ.
- **Offline:** opened as a file, the optimizer panel shows "offline" and is disabled.
- **Reference table:** flexible UMA reactant complex, TS and product (`reference_entries`).
- **JS hooks for testing:** `window.builderState()` (now also `source`, `formingBonds`, `metricsTable`),
  `window.setBuilder(params)`, `window.pageResults()`, `window.showResult(id)`, `window.setReference(id)`,
  `window.importResults(jsonText)`.

### Optimizer
`MartiniFuncs/optimize/` (generic; nothing system-specific)
- `engine.py`:
  - `optimize(symbols, positions, calculator, constraint, optimizer, fmax, steps, charge, spin, callback)`,
    with constraint modes `none`, `fix_atoms`, `fix_distances`, `restrain_distances`.
  - ASE optimizers use `FixAtoms`/`FixBondLengths`. `FixBondLengths` needs `tolerance=1e-8`: ASE's default
    1e-13 fails with float32 ML forces.
  - Sella uses `sella.Constraints` (`fix_translation`, `fix_bond`).
  - `RestrainedCalculator` adds 0.5 k (d − d0)²; tested against finite differences.
  - `ModelCache` loads each model once: `uma`, `aimnet`, `mace`, `gfn2` (tblite), `emt` (test).
    Weights come from env vars `UMA_CHECKPOINT`, `AIMNET_MODEL`, `MACE_MODEL`.
  - Charge/spin: `atoms.info["charge"/"spin"]` for UMA and MACE; arguments for AIMNet2 and tblite.
- `worker.py`:
  - One process per Python environment, JSON lines over stdin/stdout (`hello`, `run`, `cancel`, `quit` →
    `status`, `frame`, `done`).
  - **It duplicates the stdin/stdout handles and swaps fds 0/1 for devnull/stderr.** Without that, on
    Windows the job thread stalls when a library queries stdin while the main thread reads it, and library
    prints corrupt the protocol.
- `server.py`:
  - Standard library only. `--page MODULE:FUNCTION`, `--worker NAME=PYTHON:model,model`.
  - Endpoints: `GET /`, `GET /models`, `POST /optimize`, `GET /jobs/ID?since=N`, `POST /jobs/ID/cancel`,
    `GET /results`, `GET /results/ID`, `POST /results`, `DELETE /results/ID`.
  - One job per worker. Jobs live in memory only; saved results live in `ResultStore` (memory, plus one
    `DIR/ID.json` each with `--results DIR`, loaded at start). The container's `server` app passes
    `--results "$PWD/pair_builder_results"`.
- Tests: `tests/test_optimize.py` (engine, worker protocol, server, using EMT).

### Container
`containers/mlip/` (README there)
- Apptainer `mlip.def`, built in WSL2 Ubuntu to `~/containers/mlip.sif` (1.0 GB):
  ```bash
  apptainer build --fakeroot --ignore-subuid mlip.sif containers/mlip/mlip.def
  ```
  run from the repo root. `--ignore-subuid` is needed on WSL2.
- Layout: base Python 3.12 (CPU torch 2.13, numpy 2.4.6, ASE, Sella, geomeTRIC, tblite, RDKit, McUtils,
  Psience, MartiniFuncs) plus venvs `/opt/envs/uma` (fairchem 2.22, aimnet 0.2.5, e3nn 0.6) and
  `/opt/envs/mace` (mace-torch 0.3.17, e3nn 0.4.4). Versions are pinned in `requirements-*.txt`.
- Apps: `uma`, `mace`, `base`, `selftest` (energies vs. `reference_energies.json`; passes to ~1e-7 eV), `server`.
- **MartiniFuncs is copied into the image at build time:** rebuild after code changes (~4 min), or run with
  `--env PYTHONPATH=/mnt/c/Users/nicho/Documents/Martini\ collab/MartiniFuncs` to use the live source.
- Run the page and optimizer (WSL terminal), then open http://127.0.0.1:8000/ in Windows:
  ```bash
  apptainer run --app server \
    --bind "/mnt/c/Users/nicho/Documents/Martini collab/uma_data:/models/uma" \
    --bind /mnt/c/Users/nicho/.cache/mace:/models/mace \
    --bind "/mnt/c/Users/nicho/Documents/Martini collab/aimnet2_2025_b973c_d3_0.jpt:/models/aimnet/aimnet2_2025_b973c_d3_0.jpt" \
    ~/containers/mlip.sif [--port 8001]
  ```
- Verified through the container: UMA, AIMNet2, MACE-OMOL, GFN2-xTB, Sella (minimum, TS) with fixed forming
  distances. First use of UMA loads for about 35 s.

### Gotchas learned
- **Windows conda envs (outside the container):** `torch.compile` fails (no C++ compiler); set
  `TORCHDYNAMO_DISABLE=1`. Windows `aimnetcentral` is aimnet 0.1.1, which gives wrong energies when one model
  object is reused across molecules of different size (fixed in 0.2.5, the version in the container).
  The Windows `mace` env has no Sella.
- **fairchem imports ray and wandb at load time:** both must stay installed.
- **Don't stop servers with `pkill -f` patterns:** they also match the user's own server and workers. Stop by PID.

## Added in the second session (2026-11-03)

1. **Saved results** (page "Saved results" panel + server `ResultStore`).
   - Every finished job (converged, step limit, or cancelled with frames) is saved automatically:
     id, name, created, job, model, charge, spin, optimizer, fmax_target, max_steps, constraint (mode, pairs,
     targets — start distances, also recorded for fix_distances —, k, atoms), status, converged, steps,
     energy (model energy: restraint removed), restraint_energy, fmax, fmax_kind (`raw` for Sella,
     `constraint-projected` for ASE), positions, metrics (recomputed by the page on load), start (text),
     parent (id of the result it continued from), path (all frames, if "keep paths" is ticked).
   - Built-in entries: UMA reactant complex, TS, product (Show-able) and the separated reactants (energy only:
     anthracene + MA). Default reference: the UMA reactant complex.
   - Table: ref radio, name (click to rename) with model/q/spin and details, ΔE kcal/mol and eV vs the
     reference ("n/a" when model, charge or spin differ), max forming distance, Show / Settings / delete
     (click twice). The metrics table gains a "Δ vs reference" column for every metric and the energy.
   - Show loads the result (its path into the frame slider); optimizing then continues from it and records
     `parent`. Settings copies model/optimizer/fmax/steps/charge/spin/constraint (and fixed atoms).
   - Export/Import JSON (`{"format": "MartiniFuncs anthracene_ma results", "version": 1, "results": [...]}`);
     import skips ids already present and structures that are not 33 atoms.
   - Opened as a file: results kept in the page only (import works, nothing to optimize).
2. **Product reference:** `reference_structure("product")` with `pair_role`; "UMA product" in the reference
   table, as a Show-able result and as a rigid preset (height 1.90, fold 30.1). UMA energies vs the complex:
   TS +0.913 eV (+21.05 kcal/mol), product −1.173 eV (−27.06), separated reactants +0.426 eV (+9.83).
3. **Forming C–C bonds:** two cylinders (`forming-bond-K` in `forming-bond-switch-K`), radius and colour
   copied from a McUtils C–C bond (`forming_bond_markup`), placed per shown geometry from the minimax pairs
   of `metrics(P)`. View bar: forming C···C `auto` (default: bond below the cutoff, else line) / `line` /
   `bond` / `off`, cutoff default 1.8 Å. Works in builder poses, frames and references (product → bonds).
   Bond orders inside each molecule stay as McUtils drew them (the MA C=C stays a double bond).

Checked: unit tests (new: product reference, `to_builder_frame`, page data, `ResultStore`/endpoints);
the page in headless Chromium against the server with an EMT worker (run, auto-save, continue, rename,
reference switch, reload from the results folder, import, offline file mode; no JS errors).

## Portability (third session, 2026-11-03)

- `containers/mlip/fetch_models.sh`: downloads the weights (inside the image, so only apptainer is
  needed) to `~/mlip_models/{uma,mace,aimnet}`: UMA via `huggingface_hub` (`facebook/UMA`,
  `checkpoints/uma-s-1p2p1.pt`, gated, needs `HF_TOKEN`), MACE-OMOL from the mace-foundations release
  `mace_omol_0` (only with `--accept-asl`), AIMNet2 via aimnet's registry (`aimnet2-b973c-2025-d3_0`,
  SHA-256 checked; the registry file is `.pt`, the maintainer's old file was `.jpt`). Then the self-test.
- `containers/mlip/run_server.sh`: binds the models present, sets AIMNET_MODEL, makes and binds the
  results folder, `--source DIR` for live code, `--selftest`. Other arguments go to the server.
- x3dom 1.8.3 ships in the image (`/opt/mlip/assets/x3dom`, from the npm tarball, SHA-256 checked);
  server `--assets DIR` serves `/assets/...`; the page loads `assets/x3dom/x3dom.js` and falls back to
  x3dom.org (file:// or a server without assets). `.gitattributes`: `*.sh`/`*.def` always LF.
- Not verified here: the scripts against a real apptainer (only with a stub), the UMA/AIMNet2
  downloads (blocked from the sandbox), and whether the AIMNet2 `.pt` reproduces the `.jpt`
  reference energy (the self-test will tell).
- Release: `gh release create` with `mlip.sif` + `mlip.sif.sha256` (README, "Publishing a release").

## Forces and vibrations (2026-11-04)

- Engine: `single_point`, `hessian(method="auto"|"analytic"|"finite_diff", analytic=..., step=0.005)`,
  `vibrations(symbols, positions, H)` (mass-weighted, translations/rotations projected out by SVD of the
  6 external vectors, cm⁻¹ with imaginary as negative, Gaussian-convention reduced mass and force constant,
  modes scaled so the largest atom moves 1). Analytic Hessians via `ModelCache.analytic_hessian`: UMA = a second
  predict unit with `InferenceSettings(predict_untrained_hessian={"omol"}, activation_checkpointing=False)`
  (loads the checkpoint again on first use), AIMNet2 `AIMNet2ASE.get_hessian`, MACE `get_hessian` ((3N, N, 3)
  reshaped). Every analytic Hessian is checked against one finite-difference column (largest diagonal): a negated
  convention is corrected, a mismatch > 5 % is reported in `note`; `auto` falls back to finite differences if
  the analytic call fails. Optimization frames now carry `forces` (model + restraint, before constraint projection).
- Worker/server: request `task` = optimize | forces | hessian; `POST /forces`, `POST /hessian`; status `done`
  and `result` in the job state; Hessian progress/cancel every 6 coordinates.
- Page: "Forces & vibrations" panel (Calculate forces / Calculate Hessian + method). Results attach to the shown
  saved result as `analysis.forces` / `analysis.vib` (final frame, same model/charge/spin), otherwise a new
  "single point" result is saved and shown. Force arrows (auto scale: longest 1.5 Å), max |F| / RMS row; modes
  list (first imaginary selected), amplitude, play/pause animation (per-atom, ~30 fps), mode arrows (3 × amplitude),
  warning when max |F| > 0.05 eV/Å. Arrows: `force-arrow-*`, `mode-arrow-*` nodes (`arrow_markup`).
- Not yet checked with the real models: the three analytic Hessian paths (UMA settings in particular) — run
  Calculate Hessian (auto) on the UMA TS and compare with finite differences; expect one imaginary mode.

- Sella Hessian (Optimize panel, Sella only): "Sella's own (iterative)" (default, as before), "model Hessian
  when Sella needs one" (Sella `hessian_function`, `eig=True`: at the start and whenever Sella rediagonalizes),
  or "every N steps" (`diag_every_n=N`). The model Hessian uses the Forces & vibrations method (auto / analytic /
  finite differences); for restrained runs `restraint_hessian` adds the restraint term. Recorded in saved results
  (`sella_hessian`, `sella_hessian_every`, `hessian_method`) and copied by Settings.

## Ideas not done

- Delete/rename of built-in references is blocked; no "duplicate" or "compare two results side by side"
  view beyond the reference column.
- The page's ΔE uses the final model energy; for restrained runs the restraint energy is shown separately.
- Topology on the fly for bonds inside each molecule (bond orders) is not attempted.
