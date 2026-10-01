# Refactor change log (codex_refactor branch, 2026-10-01)

Each step lists what changed, why, and how it was verified. Verification used:

- **Unit tests**: `python -m unittest discover -s tests -p "test_*.py"`. Before the refactor there were
  117 tests, all passing (1 skipped).
- **Baseline comparison**: before any edit, a capture script recorded the outputs of the unmodified code:
  reader arrays, bond graphs, the reaction screen, `analyze_run` variants (serial, 2 workers, cached and
  reference backends), every statistic, and complete CLI `run`/`report`/`extract` output trees. The same
  script was re-run after each step, and every output was compared by value (relative tolerance 1e-9).
  Each difference was then classified as intended, or traced and fixed.
- **Test data**:
  - the first 400 frames (5 fs apart) of `tests/NVT_shear/nve_2p5.lammpstrj`, which is triclinic and sheared;
  - an orthorhombic copy of those frames, with the tilt removed from the box bounds;
  - `tests/lammpsio_tests/multiple_frames.lammpstrj` (NPT, 9 frames);
  - `reports/com_tracking_validation/com_comparison.npz` (2001 NPT frames, independently audited COMs).

---

## Step 1: statistics core

### 1.1 FFT lag statistics
**Changed:** added `md_statistics.lag_msd` and `lag_correlation`. These are now the single implementation
used by:
- `trajectory_analysis.molecular_window_statistics`, which feeds `write_report`, the notebooks and the CLI;
- `md.get_molecular_msd` and the molecular/atomic VACF and force-correlation functions;
- `diffusion_diagnostics._msd`.

**Why:** the four previous copies looped over lags, costing O(T × lags). At 300k frames with 100 ps of
lags that is hours per statistic, which is why `--no-lag-statistics` was needed. The FFT form computes
the same all-origin estimator in O(T log T). Molecules are processed in blocks so memory stays bounded.

**Verified:**
- New unit tests compare against explicit loops (relative error < 1e-10), including large coordinate offsets.
- On the fixtures, every shared lag matches the baseline to ≤ 7e-13 relative.
- Benchmark: 20k frames × 2k lags took 51 s with the old loop vs 0.4 s for the FFT core.

### 1.2 Lag count kept the requested endpoint
**Changed:** added `md.lag_count`. Previously `floor(max_lag/dt)` dropped the last lag through float
rounding; for example `--max-lag-ps 1.5` at dt = 0.005 ps returned 300 lags instead of 301.

**Verified:** report CSVs gain exactly one row. The last lag now equals `--max-lag-ps`, and all shared
rows are unchanged.

### 1.3 COM unwrapping for any periodic cell (triclinic, sheared, moving center)
**Changed:** added `md.minimum_image`, `md.wrap_about_center`, `md.unwrap_com_trajectory` and
`md.cell_plane_spacings`. `analyze_run` now:
- tracks COMs in every fully periodic cell, not only orthorhombic cells with a fixed center;
- stores `cell_angstrom` (frames × 3 × 3) and `max_com_step_cell_fraction`;
- adds a notice when a COM moves more than 0.25 of the cell per frame, or when the cell is
  non-orthorhombic (lab-frame displacements include any imposed shear flow).

`md.unwrap_orthorhombic_com` now delegates to the general function. The duplicated inline TOR code is gone.

**Why:** sheared runs, such as `tests/NVT_shear`, previously produced no COM, MSD or reference MSD.
Each COM is first wrapped into the current cell about its center, exactly as before. This canonical
wrap matters: under NPT dilation the image chosen for a COM changes its affine displacement by
(image) × ΔL. A first attempt without the centered wrap changed the NPT fixture COMs by up to 0.22 Å
(ΔL = 0.176 Å between its first two frames). The comparison caught this and it was corrected.

**Convention, documented:** in a deforming cell the TOR steps differ from image-flag unwrapping
(r + n·H_t) by n·ΔH per frame. Image-flag unwrapping moves each periodic image row with the deforming
cell; TOR follows each particle's in-cell motion. This is the same n·ΔL term that TOR deliberately
removes under NPT (Bullerjahn et al. 2023).

**Verified:**
- Orthorhombic and NPT outputs are identical to the baseline.
- A new synthetic test runs a growing-tilt triclinic trajectory crossing every face; TOR recovers each
  in-cell step to 1e-9.
- On the real sheared fixture, the TOR steps equal the image-flag COM steps minus n_root·ΔH. The
  remainder is below one per-step cell change, because LAMMPS image flags record crossings made at
  earlier tilts.
- The 400-frame shear COMs agree with an independent earlier reconstruction to 4.4e-3 Å over 1.2 Å
  of motion.
- Serial and 2-worker runs are bit-identical.

### 1.4 RDF for triclinic cells
**Changed:** `stability.get_molecular_com_rdf` accepts `(frames, 3)` box lengths or `(frames, 3, 3)`
cells. Non-orthorhombic cells use the fractional nearest image, then the shortest of its 27
neighbours, which is exact below half the smallest plane spacing. Volumes use |det H|.
`write_report` passes cells when available. The `triu_indices` call is hoisted out of the frame loop.

**Verified:** a brute-force image search confirms the minimum-image distances; passing orthorhombic
cells gives the same g(r) as passing lengths; the baseline RDF CSVs are unchanged.

### 1.5 Shared estimators and validators
**Changed:**
- `md.validate_times` is the single timestamp check, used by `md` and `diffusion_diagnostics`.
- `stability.scalar_autocorrelation` (FFT) and `stability.statistical_inefficiency` are shared by
  `get_series_statistics` and `diffusion_diagnostics.scalar_stationarity`.

**Why:** the previous copies used different tolerances and minimum sample counts.
`get_series_statistics` used an O(N²) `np.correlate`, which takes minutes per scalar at 300k frames.

**Verified:**
- A test compares the FFT ACF with the direct estimator (relative error < 1e-9).
- The baseline stability JSON and stationarity results are unchanged.

### 1.6 Diffusion units unified to cm²/s
**Changed:** `diffusion_diagnostics` now reports `D_cm2_s` and `ci95_cm2_s` (previously `D_m2_s` and
`ci95_m2_s`) and adds `D_pooled_cm2_s`. Notebook 08 and its test were updated.

**Why:** `md_statistics` reports cm²/s; two unit conventions for one quantity invite factor-10⁴ errors.

**Verified:** on the NPT fixture, new values equal old × 10⁴ exactly; status, reasons and alpha are unchanged.

### 1.7 Diffusion scaling exponent: kept, with extra diagnostics
**Changed:** fits now also report `alpha_plateau_corrected` (log-log fit of MSD − max(intercept, 0))
and `residual_rms_angstrom2`. The screen still uses the raw exponent.

**Why:** in the first review I suggested screening on a plateau-corrected exponent. Testing showed this
is wrong: a purely confined, noisy MSD produced a corrected exponent of 0.94, because overlapping time
origins make the curve smooth across lags. The raw exponent is the standard test for the diffusive
(Fickian) regime and correctly rejects curves still dominated by the cage plateau. That recommendation
is withdrawn.

**Verified:** the existing confined-motion and Brownian tests pass. The Brownian case gives alpha ≈ 1.

### 1.8 Smaller correctness fixes
- **Pair lifetimes, `md.get_pair_lifetime`:**
  - left-censoring now uses the same gap allowance as right-censoring (`start <= max_gap_frames`);
  - `max_gap_frames=True` is rejected.
  - Verified with a new test; in the baseline report with `--max-gap-frames 1` exactly one episode
    changed, as designed.
- **`pair_lifetimes.csv`:** new `first_source_frame` and `last_source_frame` columns.
  `first_frame` and `last_frame` remain positions within the report window.
- **`write_report --stability`:** requires 3 frames, matching `get_series_statistics`; previously it
  checked for 2, then failed with a different message.
- **`thermo.attach_thermo`:** a thermo alias that collides with a frame column (e.g. `temperature_K`
  when temperature was computed) is now joined as `thermo_temperature_K`. Previously this always raised.
  `on_conflict="error"` keeps the old behaviour, and `coverage["renamed"]` lists any renames.
- **`frame_parallel.prepare_frame`:**
  - returns a named `FrameSummary` instead of a 9-element tuple;
  - identity checks raise `ValueError` rather than `np.testing` `AssertionError`, which the CLI did not
    catch;
  - frame 0 is no longer screened for reactions twice.

### New test files and fixtures
- `tests/test_statistics_core.py`: 12 tests covering the items above.
- `tests/lammpsio_tests/sheared_frames.lammpstrj`: 10 triclinic frames (200 fs apart) from the NVT shear run.

---

## Step 2: fast bond graph and neighbour search

### 2.1 KD-tree periodic neighbour search
**Changed:** new `graph.periodic_pairs(positions, cell, pbc, cutoff)` returns every
(i, j, S, d) with |r_j + S·H − r_i| < cutoff, using a SciPy KD-tree. Positions are wrapped into
the cell; ghost images are added only within the cutoff of a face, with several layers when the
cutoff exceeds a face spacing. Shifts are mapped back to the input coordinates. Mixed periodicity
falls back to ASE `neighbor_list`. It is used by:
- `graph.build_bond_graph`: same bond rule, d < scale·(r_i + r_j);
- `graph.build_shell_neighbor_index`;
- `reaction_detection.find_possible_products`: SciPy connected components on the pair list, with
  no adjacency dictionary.

**Why:** ASE's `NeighborList` plus a per-bond Python loop took about 36 ms per frame. Every source
frame of the optional reaction screen paid that, 38–43 ms in total.

**Verified:**
- On 300 random cells (orthorhombic, triclinic, mixed PBC, cutoffs up to several cell widths), the
  pair set is identical to ASE's and distances agree to 5e-15 Å.
- Bond graphs (scale 1.15 and 1.3, periodic and not) and shell-index contacts are identical to the
  baseline on 5 fixture frames.
- New tests in `tests/test_neighbor_search.py`.

| per frame (1360 atoms) | before | after |
|---|---|---|
| `build_bond_graph` | 36 ms | 7 ms |
| reaction screen | 38–43 ms | 3 ms |
| shell neighbour index | 34 ms | 5 ms |

### 2.2 Deterministic adjacency order
**Changed:** each adjacency list is sorted by (neighbour ID, shift).

**Why:** ASE returned neighbours in spatial-bin order, which depends on geometry; atom 158 had
different neighbour orders in two dumps of the same system. Atom order within molecules (BFS from
the same root) is now reproducible across trajectories and frames.

**Effect on outputs:** the order of atoms inside each molecule list (`molecule_atom_ids`,
`molecules.json`, cached pair atom lists, `index.jsonl` `pair_atom_ids`) and inside exported
extxyz structures changes. The atoms, coordinates, forces, velocities, shifts and geometry are
unchanged.

**Verified:**
- With outputs canonicalized by atom ID, every CLI product (cache, reports, `pairs.extxyz`,
  `clusters.extxyz`, `index.jsonl`) is identical to step 1. The only difference is one distance in
  its 10th decimal.
- `forming_bonds_local` still points at the correct atoms in all 2210 exported records.

### 2.3 ID-map cache
**Changed:** `graph.get_index_maps` caches the ID→row dictionary per ID array, validated by content
on every call, and returns the cached dictionary (treat as read-only).

**Why:** it rebuilt a 1360-entry dictionary plus an `np.unique` check on every call: hundreds of
calls per frame, 31–53% of pair-search and shell time.

**Verified:** a test shows an in-place change of IDs invalidates the cache.

### 2.4 One induced-subgraph helper
**Changed:** new `graph.induced_subgraph`. `pair_selection._subgraph` uses it, and a new
`geom_utils._mal_site_centers` uses it in `get_all_prereactive_pairs` and
`get_pairs_distance_distribution`. These no longer slice `Atoms` and run a new periodic neighbour
search for each of the 40 dienophiles per frame. `get_pairs_distance_distribution` now also checks
that exactly two C=C atoms were found, as `get_all_prereactive_pairs` already did.

**Verified:** reference-backend pairs are identical to the baseline.

| per frame | before | after |
|---|---|---|
| `get_all_prereactive_pairs` (reference backend) | 154 ms | 29 ms |
| `get_pair_shell_clusters` | 91 ms | 23 ms |

### Measured end-to-end effect (400 shear frames)

| run | before | after |
|---|---|---|
| `analyze_run` with reaction screen and pairs | 19.1 s | 5.2 s |
| CLI `run` (orthorhombic, cache + report + export with shells) | 59 s | 25 s |
| CLI `extract` with 2 workers | 34 s | 12 s |

---

## Step 3: cache format and I/O

### 3.1 Compact cache (schema 2)
**Changed:**
- `cache_analysis` writes frames as `frames.npz`, one array per column.
- Pairs are stored as molecule indices: `pair_frame_offsets.npy`, `pair_molecules.npy` (CPD index,
  dienophile index into `molecule_atom_ids`) and `pair_shifts.npy`.
- `analysis.json` now holds only metadata, with `pair_storage` recording the layout.
- `load_analysis_cache` returns `pairs_by_frame` and `pair_shifts_by_frame` as read-only sequences
  that build each frame's lists on access, with the same contents as before.
- If a pair molecule is not one of the initial molecules (possible with the reference backend after a
  reaction), pairs fall back to a compact `pairs.json`.
- **Schema 1 caches, including your existing `tests/NVT_shear/cache`, still load and report unchanged.**

**Why:** `analysis.json` held 34 atom IDs per pair as indented JSON: 490 MB for the NVT shear run. It
was parsed in full up to three times per `run`, each time taking 16 s and 1.5 GB.

**Verified on `tests/NVT_shear/cache` (300,001 frames, 684,394 pairs):**
- Converting it to the new layout gives identical frames, pairs (including atom order) and shifts in
  every frame.
- Full reports from the old and new caches are byte-identical. The report covered VACF and force ACF
  for both species over all 300,001 frames × 20,001 lags (100 ps), plus pair lifetimes.

| | before | after |
|---|---|---|
| storage (excluding the molecular arrays) | 490 MB | 30 MB |
| load time | 16.0 s | 2.7 s |
| peak memory during load | 1.54 GB | 0.16 GB |
| that full report | ~5 h of lag loops | 22 s |

The old-loop cost was measured, not guessed: 29 s for 200 lags of one statistic for one species, i.e.
49 min for 20,001 lags, against 3.7 s by FFT (values agree to 4e-14).

### 3.2 Reuse checks read only the manifest; CLI stages share the loaded result
**Changed:** `cache_analysis` decides on reuse from `manifest.json` alone. `write_report` and
`export_pairs` accept `preloaded=(result, manifest)`, and CLI `run` passes the result from the cache
stage to both instead of reading the cache twice more.

### 3.3 Frame offsets: faster scan, saved once
**Changed:**
- `frame_parallel.frame_offsets` scans 64 MiB blocks with `bytes.find` instead of reading line by line
  in Python.
- `export_pairs` uses `postprocessing.cached_frame_offsets`, which saves `frame_offsets.npy` in the
  cache (written atomically) on first use.

**Verified:**
- Identical offsets to the line-by-line scan on every fixture, CRLF files, leading blank lines and a
  tiny file, at block sizes down to 15 bytes (covering every block-boundary case).
- The 43 GB dump is indexed in 50 s (0.87 GB/s, disk-bound) against about 104 s before, and only once.

### 3.4 Export parses only the selected frames
**Changed:** each export task now reads every selected frame from its own byte range. Previously a
task parsed the whole contiguous range between its first and last selected frame and discarded the
rest, so `--frame-stride k` parsed about k× more frames than needed.

**Verified:** a test checks that an export window writes exactly the selected frames. All export
products are byte-identical to step 2.

### New tests
`tests/test_cache_format.py`:
- schema 2 round trip and compactness;
- schema 1 loading, with reports identical to schema 2;
- saved offsets and selected-frame export;
- the offset scanner at every block size.

---

## Step 4: single-pass dump reader

### 4.1 Each frame is parsed once
**Changed in `lammpsio`:**
- **`_iter_blocks`** splits frames with `bytes.find` over 16 MiB reads instead of decoding and handling
  each line in Python. Errors (e.g. junk before the first frame) are unchanged.
- **New `_atom_columns_fast`** parses all numeric ATOMS columns in one C-level `np.loadtxt` pass with
  per-column dtypes. Integers are as strict as `int()`; floats are bit-identical to `float()`. It is
  used only when every row has exactly `ncols − 1` single spaces and no tabs, because `loadtxt` silently
  ignores surplus columns. Any anomaly (irregular spacing, missing or surplus columns, non-integer IDs,
  non-finite values) falls back to the previous row-by-row parser, now `_atom_columns_strict`, which
  raises the same detailed error as before.
- **New `_atoms_from_columns` and `_cell_from_bounds`** build the ASE `Atoms` directly from the parsed
  columns, using ASE's dump conventions:
  - the cell comes from `construct_cell` (bounding box minus tilts, labelled tilt order, `abc origin`
    boxes);
  - positions from x / xs / xu / xsu, in ASE's priority order;
  - unit conversion of positions, velocities, forces and charges;
  - `f_`/`v_`/`c_`/`d_` columns become float arrays; `i_` and `type` become int arrays; quaternions are
    supported.
- Previously every frame was parsed by our parser, re-joined as text (with an element column spliced in
  when missing), and then parsed again by ASE's `read_lammps_dump_text`.

**Verified:**
- Reader outputs on 5 fixtures, compared with the original baseline at zero tolerance: positions,
  cells, origins, bounds, forces, masses, all raw columns, arrays and metadata are bit-identical.
  Velocities differ by at most 1.1e-16 Å/fs, because the old path round-tripped them through momenta.
- The original reader (from git HEAD) and the new one gave identical results on 23 synthetic layouts:
  real units, scaled triclinic with mixed PBC, general `abc origin`, permuted tilt labels, xu and xsu,
  type mapping with masses and charges, mass-only, compute/fix/int columns, UNITS/TIME headers, CRLF,
  irregular spacing, tabs, several frames, leading blank lines, no final newline. Each malformed case
  (float ID, surplus/missing column, NaN, junk before the first frame, truncation) gives the identical
  error message.
- End-to-end, every capture output equals step 3 (0 differences).
- New tests in `tests/test_reader_fast_path.py`:
  - the fast path equals the strict parser on real frames;
  - every anomaly falls back to the strict parser;
  - block splitting is identical at read sizes down to 1 byte.

| | before | after |
|---|---|---|
| reader, per frame of the 43 GB dump | 6.2–7 ms | 2.9 ms |
| `analyze_run`, 400 orthorhombic frames | 3.7 s | 2.4 s |
| statistics capture | 5.6 s | 2.9 s |

---

## Step 5: duplication, defaults and documentation

### 5.1 Commented-out legacy code removed
**Changed:** removed the pre-refactor copies kept as comments at the end of `graph.py` (710 lines) and
`geom_utils.py` (422 lines). Both blocks were checked to contain only comments and blank lines, and
both remain in git history.

### 5.2 `shell.py` retired
**Changed:** `shell.py` is marked deprecated. Its docstring lists its known defects, and importing it
emits a `DeprecationWarning` pointing to `graph` and `geom_utils.get_pair_shell_clusters`. Its code is
otherwise frozen; nothing in the package imports it. It can be deleted once your own scripts no
longer use it.

### 5.3 One bond-scale default: 1.3
**Changed:** `graph.build_bond_graph` and `graph.repair_fragmented_molecules` defaulted to 1.15, while
every other function used 1.3. `geom_utils.get_diels_alder_reaction_indices_from_pair` relied on the
1.15 default; it now takes `scale=1.3` and passes it to all three of its graph builds.

**Verified:** a test asserts that the default is 1.3 for all 13 public functions with a scale
argument. This changes results only for calls that relied on the old 1.15 default; capture outputs,
which pass scales explicitly, are unchanged.

### 5.4 One `type_symbols` convention
**Changed:** new `lammpsio.normalize_type_symbols` and `type_symbol_list`. Every reader and writer now
accepts either `{type: symbol}` (string keys from JSON are fine) or a list in type order (type k is
element k−1).
- Previously the reader indexed `type_symbols[type]`, so passing the list the writers expect shifted
  every element by one type, silently.
- `write_initial_data` and `render_lammps_input` now also accept the mapping.

**Verified:** tests confirm the list and mapping forms give identical reader results, an identical
LAMMPS input, and identical data files. Invalid forms are rejected.

### 5.5 Duplicate process pool removed
**Changed:** removed `trajectory_pairs.PairBatchRunner`, `_evaluate_batch` and `_initialize_worker`,
which production code no longer used; `frame_parallel.ordered_tasks` is the one pool. Its test now
checks the real pool: results arrive in task order and a worker's exception surfaces at its position.

### 5.6 CLI: the `collect` reaction screen can be turned off
**Changed:** new `--no-reaction-screen`, mutually exclusive with `--reaction-atom-counts`. `collect`
still screens 11+23 by default.

**Verified:** parse checks for the `collect` and `run` defaults, the explicit counts, and the new flag.

### 5.7 Documentation brought in line with the code
Updated:
- README;
- `docs/consistency_review.md`, `diffusion_assessment.md`, `parallel_postprocessing.md`,
  `postprocessing_cli.md` (new cache layout), `reference_msd.md` and `workflow_api.md`;
- the markdown cells in notebooks 02, 04 (both versions) and 08;
- the `--stability` help text.

These had said that sheared/triclinic COM tracking, thermo parsing and the FFT ACF were unsupported or
quadratic. `reference_msd.md` now also explains why single-reference squared displacements do not
subtract when the reference frame moves.

---

## Overall verification summary

- **Unit tests:** 144 pass (1 skipped, as before). There were 117 before the refactor; 27 are new, in 6
  new test files.
- **End-to-end comparison with the pre-refactor baseline** (outputs ordered by atom ID): 118
  differences, every one an intended change listed above:
  - the endpoint lag;
  - diffusion keys in cm²/s;
  - COM, MSD and RDF data now available for the sheared run;
  - schema-2 cache files;
  - new lifetime columns, and one censoring change under `--max-gap-frames 1`;
  - the cell array;
  - one shell distance differing in its 10th decimal.

  No pair, exported structure, force, shell, RDF, MSD or orthorhombic/NPT COM value changed.
- **Real data:**
  - your 300,001-frame NVT shear cache loads under the new code;
  - converted to the new layout, it reproduces every frame, pair and shift;
  - the full-length reports from the old and new caches are byte-identical.

| benchmark (400 frames unless noted) | before | after |
|---|---|---|
| `analyze_run` with reaction screen and pairs | 19.2 s | 3.9 s |
| `analyze_run`, reference pair backend (30 frames) | 4.7 s | 1.0 s |
| CLI `run` (cache, report, export with shells) | 59 s | 22 s |
| CLI `extract`, 2 workers | 34 s | 12 s |
| full 300k-frame lag report (two statistics, two species) | ~5 h of lag loops | 22 s |
| cache load (300k frames) | 16 s / 1.5 GB | 2.7 s / 0.16 GB |

## Known issues not addressed in this pass

From the first review, still open:
- **Reader:** `iter_lammps_frames(wrap=True)` leaves image flags unchanged, so whole = r + n·H no longer
  holds; `xu` with image flags double-counts the image.
- **Triclinic box keys:** `box_bounds['xlo'…]` hold LAMMPS bounding-box values for triclinic dumps,
  while `read_input_datafile` returns real origins under the same names.
- **`geom_utils.pair_data_to_xyz`:** it requires velocities and truncates small forces to 8 decimals.
- **`orcaio`:**
  - constraint validation fails on files it writes with values;
  - charge and multiplicity are hard-coded;
  - `header='Auto'` passes `None` to `open`.
- **`coreio`:** it always attaches an `aimnet2` evaluator, and Psience is imported at module level.
- **Export:** each pair is still placed three times per frame (geometry, record, shell).
- **Repository:** `__pycache__/` is tracked in git although `.gitignore` excludes it.

---

# Round 2 (2026-10-01)

Verification works as in round 1. A fresh baseline was captured at commit `1fbef82`, and each phase's
outputs were compared with it. Each phase's commit was tested on its own checked-out tree.

## Phase A: input/output bug fixes

### A.1 Image flags always match the stored positions
**Changed:** `atoms.arrays["image_flags"]` now satisfies r + n·H = the LAMMPS-unwrapped position in
every case:
- `x`/`xs` columns: the dumped `ix`/`iy`/`iz`;
- already-unwrapped `xu`/`xsu` columns: zero;
- `wrap=True`: the dumped flags plus the lattice shift that wrapping applied.

The raw `ix`/`iy`/`iz` values remain in `frame.columns`.

**Why:**
- With `wrap=True` the flags were kept unchanged, so whole-molecule reconstruction from them was
  wrong by whole box vectors.
- With `xu` columns the image was counted twice.

**Verified:**
- On the NPT fixture, positions reconstructed from image flags after `wrap=True` used to differ from
  those without wrapping by up to 22.8 Å (a full box); they now agree to 7e-15 Å.
- New tests cover triclinic wrapping, `xu` against `x`, and `xu` with `wrap=True`.

### A.2 Unambiguous box origin and cell
**Changed:** dump `box_bounds` and the bounds from `read_input_datafile` both gain `origin` and `cell`
(rows a, b, c). In dumps, `xlo`…`zhi` keep their LAMMPS meaning, which is the bounding box for
triclinic cells. The data-file bounds keep the real box limits.

**Verified:**
- On the sheared fixture, `origin` and `cell` equal ASE's `celldisp` and cell, while `xhi − xlo`
  differs from a, the tilted case.
- The data-file cell diagonal equals hi − lo.

### A.3 `geom_utils.pair_data_to_xyz`
**Changed:** velocities and forces are optional, the comment line's `columns=` lists what was
written, and values use 17 significant digits, as in the extxyz writer. Previously it raised
`KeyError` without velocities, and `%12.8f` turned forces below 5e-9 eV/Å into 0.

### A.4 `coreio`
**Changed:**
- Psience and McUtils are imported only inside the functions that use them, and `ANG_TO_BOHR` /
  `BOHR_TO_ANG` are computed on first access. The mass/XYZ helpers now import without Psience.
- `ase_atoms_to_molecule(..., energy_evaluator=None)` gives a molecule without an evaluator; this used
  to crash. The default still attaches Psience's `aimnet2` evaluator, because `orcaio` (deliberately
  unchanged) relies on it.

**Verified:** tests import `coreio` with Psience and McUtils blocked, and exercise the
no-evaluator option.

**Baseline comparison:** the only differences are the added `box_bounds` keys and the corrected flags
in the `wrap=True` fixture.

## Phase B: performance

### B.1 Export
**Changed:**
- **`pair_data_to_extxyz`** formats each row with one `%` operation (`"%.17g"` for floats, `"%s"`
  otherwise) instead of one `format` call per value.
- **`graph.frame_cache()`** memoizes `unwrap_component` within one frame, so each molecule is
  unwrapped once per frame instead of up to three times (pair record, geometry, shell). The memo is
  dropped after each frame, because object ids can be reused.
- **`pair_selection.site_cache()`** reuses the reactive-site search across frames. It is keyed by the
  molecule's atom IDs, its elements and its exact intramolecular bond list, so any bond change misses
  the cache.
- `_export_frame_chunk` uses both.

**Verified:**
- The export of 200 frames (`pairs.extxyz`, `clusters.extxyz`, `index.jsonl`) is byte-identical
  before and after.
- Capture comparison against phase A: 0 differences.
- New tests in `tests/test_export_caches.py`: the caches don't change geometry or shells, both are
  released on exit, and the site cache misses when a bond is removed.

| | before | after |
|---|---|---|
| export, per frame | 53.5 ms | 38.4 ms (1.4×) |
| CLI `run` (400 frames, with export) | 22.3 s | 17.3 s |
| CLI `extract` | 11.8 s | 9.1 s |

**Honest note:** the 3–5× suggested earlier was not reached. Most of the remaining export time is
per-atom shell bookkeeping in Python (`get_shell_contacts`, `place_shell_components`), plus a bond-graph
rebuild per frame. That rebuild is kept on purpose, as an independent check that molecule membership
hasn't changed.

### B.2 Reaction screen
**Changed:** for orthorhombic periodic cells, the screen finds bonded candidates with a single periodic
KD-tree (SciPy `boxsize`): no ghost images, shifts or sorting, which connectivity doesn't need. Other
cells keep the general `periodic_pairs` path. The screen stays generic: any A+B-sized component
anywhere, as before.

**Verified:** connected components are identical to the bond-graph rule on 21 frames, including a copy
shifted so molecules straddle every face.

| per frame | before | after |
|---|---|---|
| orthorhombic | 3.0 ms | 1.9 ms |
| triclinic | 3.0 ms | 3.0 ms (unchanged path) |

The sub-1 ms figure suggested earlier needs a chemistry-specific screen (reactive sites only), which
would miss other reactions. It was not done.

## Phase C: dynamics science

Equations, assumptions, caveats and results on your NVT shear run are in `docs/dynamics_diagnostics.md`.

**New module `MartiniFuncs/dynamics.py`:**
- non-affine unwrapping;
- non-Gaussian parameter, self van Hove function, successive-step correlation, sampling diagnostics;
- Green–Kubo diffusion, Yeh–Hummer correction, NEMD shear viscosity;
- P₁/P₂ rotational correlations and correlation times;
- PMFs from g(r), with 1D/2D Jacobians, and an `index.jsonl` geometry loader.

**`analyze_run` additions:**
- always stored: `cell_origin_angstrom` and `com_nonaffine_angstrom` (affine cell motion removed at
  every source frame);
- opt-in: `orientations=True` → `orientation_axes`, sign-continuous principal axes per molecule (CLI
  `--orientations`);
- `source_range=(start, stop)`, for both the serial and parallel readers;
- `on_topology_change="record"/"truncate"` with `topology_persistence`.

**Other additions:**
- `trajectory_analysis.analyze_segments`, which chains fixed-topology segments;
- `reaction_detection.topology_change`;
- `frame_parallel.principal_axes`;
- `md_statistics.affine_step`.
- The cache stores the new arrays (older caches simply lack them).

**Verified:** `tests/test_dynamics.py` has 16 tests, each against a case with a known answer:

| feature | reference case | result |
|---|---|---|
| shear and dilation removal | pure affine motion | 0 to 1e-9 Å |
| shear and dilation removal | shear + Brownian | recovers 6Dt (lab-frame MSD is more than 2× larger) |
| non-Gaussian parameter | Gaussian; mixture with f = ½ | 0; 1 |
| van Hove | Brownian | matches the Gaussian |
| step correlation | i.i.d. rattling; random walk | −½; 0 |
| sampling diagnostics | free flight; caged | ratio 1; ratio < 0.2 |
| Green–Kubo | exponential VACF | D = σ²/γ |
| Yeh–Hummer, NEMD | closed-form values | exact; units and sign checked |
| rotational correlations | rotational diffusion | e^(−2D_rτ), e^(−6D_rτ) |
| principal axes | ring and rod | correct axes |
| PMF | inverting a Boltzmann factor; isotropic angles | exact; flat |
| segments | synthetic reaction in a real dump | split at the persistent event; earlier one-frame contact ignored; product becomes C₁₆H₁₅NO₂ |
| `source_range` | full run, serial and parallel | identical |

The shear fixture's stored non-affine trajectory equals the array function, with the flow only along x.

**Baseline comparison against phase B:** only additions (new arrays, cache files and keys), plus the
corrected notice: thermo logs *are* parsed, by `MartiniFuncs.thermo`.

**Correction to the first review:** I called the 1.43–1.46 g/cm³ NPT density suspicious. The input
(`tests/lammpsio_tests/infile.in`) compresses the system to 2.5 GPa on purpose, and the new diagnostics
(caging, an elastic shear response of about 1 GPa) are consistent with that state. The slow dynamics are
physical for this state point, not evidence of a pressure bug.

## Phase D: repository hygiene

- **Removed:**
  - `MartiniFuncs/shell.py`, deprecated in round 1, together with its three legacy tests in
    `tests/test_consistency.py`;
  - the empty `MartiniFuncs/gaussianio.py`.
- **Untracked:** the 14 `__pycache__/*.pyc` files. They were already in `.gitignore`; your local copies
  are kept.
- **Added `.gitattributes`:**
  - text is stored as LF and checked out with the platform ending;
  - `.lammpstrj` and `.data` fixtures are kept byte-exact;
  - binary outputs are marked binary.

  One renormalization commit changed only line endings (no content change ignoring CR). Before it the
  index held 56 CRLF, 55 LF and 15 mixed-ending text files; now every text file is LF.

**Verified:** 165 tests pass after each commit (168 minus the 3 removed with `shell.py`).

## Reaction events never terminate the CLI (2026-10-01)

**Why:** reaction events are rare but wanted, and they must not end a long job. Previously:
- `collect` enabled the abort-on-34-atom screen by default;
- the export raised "cached pair membership differs" as soon as two molecules had bonded;
- `run` without the screen raised when a molecule fell apart.

**Changed:**
- **New CLI options** for `run` and `collect`: `--reaction-events {record,stop,off}` (default `record`)
  and `--reaction-persistence N` (default 5 source frames). These use `analyze_run`'s
  topology-change detection.
- **Report:** writes `topology_events.csv`; the manifest has a `topology_events` count.
- **Export:** flags pairs whose molecules are no longer separate components (`membership_changed` in
  `index.jsonl`, `structures_with_changed_membership` in the manifest) instead of aborting.
- **`collect`** no longer forces the legacy screen. `--reaction-atom-counts` remains available as an
  explicit opt-in abort.

**Verified:** `tests/test_reaction_cli.py` runs the CLI on a synthetic reacting trajectory (a CPD bonded
to a dienophile from frame 4, with a one-frame contact at frame 2):

| command | result |
|---|---|
| `run` with 2 workers | exit 0, one event at frame 4 (C16H15NO2), the frame-2 contact ignored, all 9 frames analyzed and exported, at least one structure flagged |
| `run --reaction-events stop` | exit 0, frames 0–3 only |
| `collect` (default) | exit 0, the event recorded |
| legacy screen | still aborts (exit 2) |

All 168 tests pass.
