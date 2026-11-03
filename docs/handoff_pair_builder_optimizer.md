# Hand-off: anthracene + MA pair builder page with optimizer (as of 2026-11-03)

Repository: `C:\Users\nicho\Documents\Martini collab\MartiniFuncs`, branch `claude_refactor`
(never open PRs to master). Conda env for tests: `C:\Users\nicho\anaconda3\envs\mlip_uma\python.exe`
(`python -m unittest discover -s tests -p "test_*.py"`: 224 tests, all pass, 1 skipped).

**Nothing below is committed yet.** Last commit is `abc13d6`. Uncommitted: the chemistry
refactor (shared Diels-Alder module), `chemistry/anthracene_ma/`, `optimize/`, `containers/mlip/`,
`examples/build_from_smiles.py`, tests. `examples/04_old_backup.ipynb` stays untracked on purpose.

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
- **Reference table:** flexible UMA reactant complex and TS only.
- **JS hooks for testing:** `window.builderState()`, `window.setBuilder(params)`.

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
  - Endpoints: `GET /`, `GET /models`, `POST /optimize`, `GET /jobs/ID?since=N`, `POST /jobs/ID/cancel`.
  - One job per worker. Jobs live **in memory only**.
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

## Future work requested

1. **Keep optimized geometries and compare them.** At present one geometry is optimized at a time; changing
   the structure or rerunning loses the previous result.
   - Store each finished optimization as a saved result: name, model, charge/spin, optimizer and settings,
     constraint (mode, pairs/atoms, targets, k), converged flag, steps, final energy, final max force
     (record whether it is constraint-projected or raw for Sella), final positions, metrics. Optionally the
     whole path.
   - The page needs a results list: show or load any saved result into the view, choose a reference, and
     show ΔE (eV, kcal/mol) and metric differences against it.
   - Energies are only comparable for the same model, charge and spin; flag or block comparisons across models.
   - Persistence options: in-page with export/import of a JSON file (simplest, survives server restarts if
     exported), and/or a server-side store, e.g. a JSON results folder bound into the container. Today jobs
     exist only in the server's memory.
   - Also consider "continue from saved result", e.g. re-optimize with a different model or constraint.
2. **Add the product to the reference metrics.** The table has only the reactant complex and TS because
   `structure_metrics` and `measure_placement` split the structure into two molecules by bonding, and the
   adduct is one 33-atom molecule (`_pair_sites` raises). Both already accept a `pair_role` array instead.
   For the reference product, assign roles from the hand-off atom mapping: MA rows `[0,1,2,3,4,5,20,21,32]`,
   the rest anthracene; `tests/anthracene_ma_tests/handoff_manifest.json` has the mapping. Then add
   "UMA product" to `references` in `builder_page()` and possibly a product preset. The product's metrics
   (forming C–C about 1.56 Å, screen distance about 1.90 Å) are the useful end point for criteria.
3. **Show forming C–C bonds as bonds when they exist.** Today the two forming pairs are only an orange line;
   inter-molecular bonds can't appear, because the molecules are rendered separately.
   - Idea: add two half-cylinder pairs per forming bond to the scene (styled like McUtils bonds, inside a
     `Switch`). Place them with the same per-atom rule as other cylinders (`t`, `s`, from the matched pair).
   - Switch line ↔ bond by a user option (always line / always bond / automatic when d < cutoff, e.g. about
     1.7–1.8 Å, or the package's bond rule 1.3 × sum of covalent radii).
   - Use the pairs chosen by the minimax matching in `metrics(P)`.
   - Applies to builder poses, optimization frames and the product reference, so loading the product shows
     real C–C bonds.
   - The same "topology on the fly" could later handle bonds that break, but none do in this reaction.
