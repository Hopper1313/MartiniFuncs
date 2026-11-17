# Organic crystal workflows (`MartiniFuncs.crystals`)

From a CIF to whole-molecule slabs, two-crystal interfaces for collision MD, and MLIP
bulk and surface properties. Install the extra once: `pip install -e .[crystals]`
(pymatgen and spglib; UMA needs `.[uma]`, MACE needs `mace-torch` in its own environment).

## What uses which library

| Step | Implementation |
|---|---|
| CIF parsing | ASE's CIF tokenizer (`ase.io.cif.parse_cif`) + the CIF's own symmetry operators (`ase.spacegroup.parse_sitesym`). ASE's `read` failed on 3 of 5 CSD sample files (space-group setting lookups, text before `data_`), and pymatgen's parser drops `_atom_site_disorder_group`, which the major/minor disorder choice needs. |
| Whole molecules | package bond graph (`structure.graph`, covalent radii x 1.3) |
| Miller basis, terminations | `slabs.miller_basis` (det 1, stacking vector on the (h k l) side; pymatgen's oriented cell can be doubled or point to the bottom face) and a COM-layer cut so no molecule is ever split |
| Symmetry-distinct faces | pymatgen `get_symmetrically_distinct_miller_indices` (spglib) |
| Lattice matching | pymatgen `ZSLGenerator` (Zur-McGill), with mirror-image matches removed and the strain from a polar decomposition |
| Wulff shape | pymatgen `WulffShape` |
| Relaxations, EOS | ASE `FrechetCellFilter`, `FixSymmetry` (spglib), `EquationOfState`, BFGS/LBFGS/FIRE |

## Quick start

```python
from MartiniFuncs.crystals import (read_cif, build_slab, terminations, flip_slab, match_lattices,
                                   build_interface, set_collision_velocities, crystal_calculator,
                                   relax_bulk, bfdh)

ant = read_cif("ANTCEN.cif")                 # P2_1/a, Z = 2, Z' = 0.5
mal = read_cif("MLEICA.cif")
print(ant.summary())

bfdh(ant, max_index=2)[:5]                   # likely faces, no calculator needed
terminations(ant, (0, 0, 1))                 # cut planes, fewest broken contacts first

# no supercell/min_lengths here: matching starts from the primitive surface cells
a = build_slab(ant, (0, 0, 1), min_thickness=20)
b = build_slab(mal, (1, 1, 0), min_thickness=20)
matches = match_lattices(a, b, max_area=1200, max_length_tol=0.04, max_angle_tol=0.04, sort="strain")
for m in matches[:5]:
    print(m.area, m.max_strain, m.n_atoms)    # ANTCEN (001) / MLEICA (110): 724 A^2 at 1.0 % strain
interface = build_interface(a, b, matches[0], gap=3.0, vacuum=20, strain_fraction=0.5)
set_collision_velocities(interface, 500.0, temperature_K=300)   # 500 m/s approach speed
interface.write("interface.extxyz")
```

### Slab controls

| Option | Meaning |
|---|---|
| `layers` / `min_thickness` | whole d_hkl layers, or the fewest layers reaching a thickness (Angstrom) |
| `termination` | index into `terminations()` (0 = fewest intermolecular contacts cut per nm^2) |
| `supercell` | `(m, n)` or a 2x2 integer matrix on the reduced in-plane vectors |
| `min_lengths`, `orthogonal`, `max_det` | choose the smallest in-plane supercell with both vectors at least this long (angle 60 to 120 degrees; `orthogonal=True` prefers 90) |
| `max_atoms` | raise instead of building something larger |
| `vacuum` | Angstrom along z; c is perpendicular to the surface |

Molecules belong to layers by centre of mass, so slabs are always stoichiometric and never cut
a molecule. The top face is (h k l), the bottom (-h -k -l); `flip_slab` turns a slab over.

### Interfaces for collisions

`match_lattices(a, b)` returns coincident supercells sorted by area (`sort="strain"` for the
least strained). `build_interface` puts B above A. By default B's bottom face (-h -k -l) meets
A's top face (h k l). To bring B's (h k l) face down instead, match and build with
`flip_slab(b)`: flipping mirrors the in-plane lattice, so it must be done before matching.
`strain_fraction` puts the strain all in B (1), all in A (0) or splits it (0.5). With
`rigid_molecules=True` only molecular centres move with the strain, so no bond is stretched,
but the in-plane spacing of the molecules changes by up to the lattice mismatch.

`strain_fraction=None` strains neither slab: the cell is A's, B is only rotated and keeps its own
spacing, and the mismatch collects at the cell boundary as a seam. `atoms.info["seam_misfit_A"]`
gives how far molecules across the boundary sit from their crystal positions, one value per cell
vector (0.26 and 0.30 Angstrom for anthracene (1 1 0) / maleic anhydride (1 1 0)). Two
incommensurate lattices cannot share one periodic cell without either strain or a seam; pick the
seam when the slabs' own lattices matter more than a defect line at the cell edge.

### Interface size

The atom count is the in-plane area times each slab's thickness. Three controls set it:

| Control | What it changes |
|---|---|
| which `LatticeMatch` you pass | the coincident cell: smaller cells need more strain |
| `build_interface(..., repeat=(n, m))` | the matched cell repeated n x m times; strain unchanged, a seam misfit grows by n and m |
| `build_slab(layers=... / min_thickness=...)` | each slab's thickness, separately |

`interface_options` lists every combination within your limits, with exact atom counts:

```python
from MartiniFuncs.crystals import interface_options

opts = interface_options(ant, (1, 1, 0), mal, (1, 1, 0), thicknesses=(10, 15, 20, 30),
                         repeats=((1, 1), (2, 1), (1, 2), (2, 2)), max_strain=0.015,
                         min_atoms=2000, max_atoms=12000)
for o in opts:
    print(o.n_atoms, round(o.area), o.cell_lengths, o.max_strain, o.repeat, o.layers_a, o.layers_b, o.seam_misfit)
interface = opts[1].build(strain_fraction=0.5, gap=3.0, vacuum=20)      # or strain_fraction=None
```

`thicknesses` also takes `(thickness_a, thickness_b)` pairs to size the slabs separately. By default
only cells on the size/strain front are kept (no smaller cell has less or equal strain);
`all_cells=True` keeps every match. For anthracene (1 1 0) / maleic anhydride (1 1 0) on the
UMA-relaxed cells, the smallest interfaces under 1.5 % strain are 2,736 atoms (1,141 Å², 1.33 %)
and 3,240 atoms (1,348 Å², 0.81 %), both with 12–14 Å slabs.

### Rotation, registry and Diels–Alder alignment

Two faces can be joined through several coincident cells. Each cell fixes the in-plane rotation
of B (`LatticeMatch.rotation_deg`, the R of F = R·U): any other angle breaks the periodic match,
so choosing the rotation means choosing the cell. The lateral registry of B over A is free:
`build_interface(..., shift=(s1, s2))`, in fractions of the cell.

Neither is chosen for chemistry by default. `MartiniFuncs.chemistry.anthracene_ma.contact` scores
an anthracene / maleic anhydride contact: an anthracene site is *lined up* when a maleic
anhydride C=C midpoint lies within 1.5 Å of its C9/C10 midpoint in plane and the two axes are
within 30° of parallel; *orientation-compatible* sites have such a C=C anywhere in the cell, a
number fixed by the rotation. Use it to rank cells and to build at the best registry:

```python
from MartiniFuncs.chemistry.anthracene_ma.contact import best_shift, contact_alignment

opts = interface_options(ant, (1, 1, 0), mal, (1, 1, 0), thicknesses=(18,), max_strain=0.02,
                         all_cells=True, scorer=contact_alignment, sort="score")
for o in opts:
    print(round(o.area), o.max_strain, o.match.rotation_deg, o.scores["orientation_compatible"],
          o.scores["lined_up_best"], o.scores["n_sites"])
interface = opts[0].build(strain_fraction=0.5, shift=best_shift)   # rebuilt at its best registry
```

For anthracene (1 1 0) / maleic anhydride (1 1 0) T0 the least-strained cell (1,348 Å², 0.81 %,
B rotated 88.3°) crosses every C=C with the C9···C10 axes (0 of 26 compatible). The same-size
cell at 0.99 % (−32.6°) makes all 26 compatible and lines up 6 at its best registry; the
1,141 Å² cell at 1.51 % (6.1°) lines up 11 of 22. The showcase page uses the least-strained fully
compatible cell (−32.6°) as its default. This is a geometric screen: the groups are still about
8 Å apart vertically at a 3 Å atom-to-atom gap, and no energies are involved.

Maleic anhydride (1 0 0) T2 is a strong alternative to (1 1 0). BFDH ranks it only 8th (the 2₁ axis
halves its layer spacing), but its relaxed γ is the lowest computed (111.6 mJ/m²). On anthracene
(1 1 0) its fully compatible cell (1,452 Å², 1.17 %) lines up 13 of 28 sites. Its least-strain cell
(0.62 %) again crosses every C=C. For orthorhombic (h 0 0) and (0 0 l) faces the
terminations related by the 2₁ axis along the normal score identically: their contact layers are
180° in-plane copies, and a C=C axis is unchanged by a 180° turn.

## Properties with an MLIP

```python
calc = crystal_calculator("uma", model="uma_data/uma-s-1p2p1.pt", task="omc", ncpus=8)
# or crystal_calculator("mace-off", model="medium")
```

Use UMA's **`omc`** task for molecular crystals (PBE+D3 training data). `omat` is for inorganic
materials and has no dispersion; `omol` is for isolated molecules. Use the same calculator for the
gas-phase references. On CPU/Windows the calculator skips `torch.compile`; pass
`inference_settings="turbo"` on a GPU node for long MD.

```bash
python -m MartiniFuncs.crystals.workflow ANTCEN.cif --calculator uma \
    --model uma_data/uma-s-1p2p1.pt --task omc --out antcen_uma --n-faces 6 --elastic
```

The workflow writes `results.json` after every step, plus the relaxed bulk, the gas molecules and every slab:

| Key | Content |
|---|---|
| `structure` | formula, Z, Z', space group, centrosymmetric, density, disorder sites dropped |
| `bulk` | relaxed cell, volume and cell changes vs. the CIF (thermal expansion: the CIF is at `source.temperature_K`, the MLIP at 0 K) |
| `lattice_energy` | E<sub>latt</sub> per molecule (eV, kJ/mol) and ΔH<sub>sub</sub> at 298.15 K |
| `eos`, `elastic` | Birch–Murnaghan E<sub>0</sub>, V<sub>0</sub>, B<sub>0</sub>, B′<sub>0</sub>; C<sub>ij</sub>, Voigt/Reuss/Hill moduli, Born stability |
| `bfdh` | d<sub>eff</sub> with systematic absences, BFDH area fractions |
| `surfaces` | γ (cleaved and relaxed) per face and termination |
| `wulff`, `face_ranking` | area fraction of each face family; faces by surface energy |

### Equations

Energies are per molecule and are 0 K potential energies unless stated (1 eV = 96.485 kJ/mol;
1 eV/Å² = 16.02 J/m²).

- **Lattice energy**: E<sub>latt</sub> = E<sub>crystal</sub>/N<sub>mol</sub> − E<sub>gas</sub>.
  - E<sub>crystal</sub> is the energy of the relaxed bulk cell holding N<sub>mol</sub> molecules.
  - E<sub>gas</sub> is one isolated molecule relaxed with the same calculator, with no charge or spin set.
  - It is negative for a bound crystal.
- **Sublimation enthalpy**: ΔH<sub>sub</sub>(T) ≈ −E<sub>latt</sub> − 2RT.
  - This is the enthalpy to take one mole of molecules from the crystal to the ideal gas at T, the quantity measured experimentally.
  - The −2RT comes from the crystal's 6 lattice modes per molecule (6RT, classical) against the gas's 3/2 RT translation, 3/2 RT rotation and RT for pV (4RT in total).
  - Zero-point energy and changes in intramolecular vibrations are neglected.
- **Surface energy**: γ = (E<sub>slab</sub> − N e<sub>bulk</sub>) / 2A.
  - E<sub>slab</sub> is the periodic slab with vacuum, holding N molecules; e<sub>bulk</sub> = E<sub>crystal</sub>/N<sub>mol</sub>.
  - A is the in-plane cell area, and 2A because the slab has two surfaces.
  - When the top (h k l) and bottom (−h −k −l) faces differ (non-centrosymmetric crystals), γ is their average.
- **Linear fit**: E<sub>slab</sub>(n) = n e<sub>layer</sub> + 2γA over slabs of n layers.
  - The intercept gives γ without e<sub>bulk</sub>.
  - The slope gives Δe = e<sub>layer</sub>/m − e<sub>bulk</sub> (m molecules per layer). A non-zero Δe is a mismatch between the slab interior and the bulk reference, and makes the direct γ drift with thickness.
- **Removal energy**:
  - Vertical: E<sub>rem</sub> = E(slab − m) + E(m) − E(slab), with geometries frozen.
  - Adiabatic: the vacancy slab and the molecule (E<sub>gas</sub>) both relaxed.
  - Bounds: |E<sub>latt</sub>| at a kink ≤ E<sub>rem</sub> at a terrace ≤ 2|E<sub>latt</sub>| in the bulk.
- **Interface strain**: F = R U, mapping B's coincident supercell onto A's.
  - R is B's in-plane rotation and U the symmetric stretch.
  - The principal strains are ε<sub>k</sub> = λ<sub>k</sub>(U) − 1; max strain = max |ε<sub>k</sub>|.

`thickness_convergence` checks the number of layers and also gives the linear-fit γ.

The workflow's `--n-faces` takes the top BFDH faces only, so a low-energy face with a small d_eff can
be missed. Compute a few more with `examples/relax_surface_energies.py --faces ...`. Maleic anhydride
(1 0 0), 8th by BFDH, turned out to have the lowest γ.

### How big a slab

`examples/surface_size_convergence.py` varies each size separately: layers (relaxed and cleaved γ,
plus the linear fit), the in-plane supercell (relaxed), and the vacuum (cleaved).

```bash
python examples/surface_size_convergence.py mleica_uma --faces 110 101 --terminations 0 \
    --layers 3 4 5 6 8 --supercells 1x1 2x1 1x2 2x2 --vacuums 10 15 20 30 \
    --model uma_data/uma-s-1p2p1.pt --ncpus 24
```

Pass `--rattle 0.1` when testing supercells. Without it a supercell keeps the primitive cell's
symmetry through the relaxation, so it reproduces the 1×1 result by construction.

To compare faces at matched thickness, give thicknesses in Å and per-face terminations:

```bash
python examples/surface_size_convergence.py mleica_uma --faces 110:0,1,3 010:0,2 101:0,1 \
    --thicknesses 20 30 40 50 --skip lateral vacuum --model uma_data/uma-s-1p2p1.pt --ncpus 24
```

Each computed slab is written to `<out>_structures/` (relaxed when relaxed); pass `--no-structures` to skip.

UMA `omc` results. The supercell and vacuum tests cover anthracene (1 1 0), (0 0 1) and maleic
anhydride (1 1 0), (1 0 1), (1 0 0), (0 0 1), (1 1 1). The 20–50 Å series covers every distinct
termination of the top-6 BFDH anthracene faces and all 8 maleic anhydride faces:

| Size | Effect on γ |
|---|---|
| vacuum 10–30 Å | none (6 Å cutoff, no long-range electrostatics) |
| in-plane 1×1 → 2×2, rattled | < 1 mJ/m²; no reconstruction found by local relaxation |
| below ~20 Å (3–4 thin layers) | 2–4 mJ/m² high (anthracene (1 1 0): 95.2 at 14 Å) |
| 20 → 50 Å | ≤ 2.5 mJ/m², and < 1.3 mJ/m² for all but three terminations |
| linear fit (20–50 Å) vs 20 Å value | ≤ 1.3 mJ/m² |
| relaxation scatter (rattled starts) | 1–4 mJ/m² |

The fit's slope puts molecules inside anthracene slabs 1–2 meV below the bulk cell, and inside
maleic anhydride slabs within 0.7 meV of it. That is why anthracene's direct γ drifts down slightly
with thickness. Fits that include slabs thinner than about 20 Å overstate the offset (3–4 meV for
anthracene (1 1 0) over 3–8 layers), because those slabs are not yet in the linear regime. Read γ
from slabs of 20 Å or more as ±1–2 mJ/m². Face orderings closer than about 3 mJ/m², such as maleic
anhydride (1 0 0) at 111.6 against (1 0 1) at 113.5, are not resolved.

Relaxed γ (mJ/m², lowest termination) and Wulff area fractions:

| Face | γ at 20 Å | γ at 50 Å | γ from fit | Wulff area (fit) |
|---|---|---|---|---|
| anthracene (0 0 1) | 76.5 | 76.2 | 76.7 | 32.7 % |
| anthracene (1 1 0) | 92.6 | 91.6 | 93.3 | 30.1 % |
| anthracene (2 0 −1) | 88.7 | 87.0 | 89.9 | 18.9 % |
| anthracene (1 1 −1) | 106.5 | 105.5 | 107.2 | 16.9 % |
| anthracene (1 0 −1) | 118.4 | 117.0 | 119.4 | 1.4 % |
| anthracene (0 1 1) | 125.4 | 123.4 | 126.4 | 0 % |
| maleic anhydride (1 0 1) T0 | 113.5 | 114.1 | 113.2 | 33.3 % |
| maleic anhydride (1 2 0) T1 | 125.6 | 126.6 | 125.2 | 21.2 % |
| maleic anhydride (1 1 0) T0 | 118.1 | 118.4 | 117.8 | 15.1 % |
| maleic anhydride (1 0 0) T2 | 111.6 | 112.0 | 111.4 | 8.4 % |
| maleic anhydride (1 1 1) T0 | 137.8 | 137.9 | 137.6 | 7.3 % |
| maleic anhydride (0 1 0) T2 | 144.4 | 144.7 | 143.8 | 6.1 % |
| maleic anhydride (0 1 1) T0 | 151.5 | 152.7 | 150.2 | 5.9 % |
| maleic anhydride (0 0 1) T0 | 133.1 | 133.4 | 132.9 | 2.7 % |

The three Wulff shapes (20 Å, 50 Å, fit) agree within about 1 % of area for every face.

### Fracture and growth: cleavage and attachment energies

The Wulff shape is the equilibrium shape. Two other questions need other quantities.

**Which planes a crystal breaks along** (milling, impact). Use the cleavage work W<sub>cl</sub> = 2γ
of the surface as cut, before it relaxes: the Griffith energy per area of a new crack. Take it from
`surface_energy(..., relax=False)`, or better from the unrelaxed linear fit over 20–50 Å slabs.

**Which faces a grown crystal shows**. Use the attachment-energy model:

```python
from MartiniFuncs.crystals import attachment_energies, attachment_energy_shape
rows = attachment_energies(relaxed, calc, e_bulk, faces, lattice_energy=E_latt)
shape = attachment_energy_shape(relaxed, rows)   # Wulff construction with |E_att|
```

- **Attachment energy.** Each termination defines a d<sub>hkl</sub>-thick slice. E<sub>att</sub> = e<sub>bulk</sub> − e<sub>slice</sub> is the energy released per molecule when that slice attaches, and E<sub>latt</sub> = E<sub>slice</sub> + E<sub>att</sub>.
- **Growth rate.** A face grows at a rate proportional to |E<sub>att</sub>|, using its most strongly bound slice.
- **Command line:** `examples/attachment_energies.py` runs every termination of the given faces. Each slice is a single unrelaxed point, so the whole set takes about a minute on the cluster.

UMA `omc` results (W<sub>cl</sub> from the unrelaxed 20–50 Å fit, mJ/m²; E<sub>att</sub> in kJ/mol):

| Crystal | Easiest cleavage planes (W<sub>cl</sub>) | Weakest attachment (E<sub>att</sub>/E<sub>latt</sub>) | Growth shape |
|---|---|---|---|
| anthracene | (0 0 1) 154; (2 0 −1) 179; (1 1 0) 188 | (0 0 1) −25.9 (0.24) | (0 0 1) 48 %, (1 0 −1) 23 %, (1 1 0) 15 %, (1 1 −1) 11 % |
| maleic anhydride | (1 0 1) T0 234; (1 0 0) T2 236; (1 1 0) T0 254 | (0 1 0) T2 −18.0 (0.24) | (0 1 0) 43 %, (1 0 0) 30 %, (0 0 1) 15 %, (1 0 1) 12 % |

- **Anthracene:** (0 0 1) is both the easiest cleavage plane and the slowest-growing face, which matches its known (0 0 1) plate habit and flaky cleavage. The equilibrium shape does not show this.
- **Maleic anhydride:** the two measures differ. Per area, (1 0 1) and (1 0 0) cleave most easily. Per molecule, (0 1 0) attaches most weakly, because its 11 Å slices hold more molecules per area. For fracture experiments the per-area W<sub>cl</sub> is the relevant measure.
- **After fracture:** fragment surfaces relax locally, but at room temperature molecules don't move far enough to reshape a fragment towards the Wulff shape.
Anthracene (2 0 −1), 5th by BFDH and missing from the first runs, has the second-lowest γ. Maleic
anhydride (0 1 0) only appears on the shape through its T2 termination, which the first runs
skipped.

## Caveats

- **Disorder:** `disorder="major"` keeps groups `.`/`1`/`-1` (or occupancy >= 0.5 when there are no
  groups); check `source["dropped_disorder_sites"]`.
- **Hydrogens:** a CIF without H raises; `normalize_hydrogens=True` sets X-ray X-H bonds to neutron lengths.
- **Polar faces:** for non-centrosymmetric crystals the slab's two faces differ, so gamma is their
  average, and a net dipole across the slab can make energies converge slowly with vacuum.
- **Wulff:** `WulffShape` uses the lattice's point symmetry (the Laue class, except for
  pseudo-symmetric cells), and needs face families from enough directions to close the shape.

## Not yet included (natural next steps)

Surface relaxation with the in-plane cell free, fracture beyond the Griffith energy (slip planes,
plasticity, crack paths under realistic stress),
temperature (NPT MD lattice parameters, quasi-harmonic free energies), polymorph ranking,
and adhesion / work of separation of matched interfaces (`interfaces.separation_curve`
gives a rigid scan).
