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
| `lattice_energy` | E_latt per molecule (eV, kJ/mol) and dH_sub = -E_latt - 2RT |
| `eos`, `elastic` | Birch-Murnaghan B0; C_ij, Voigt/Reuss/Hill moduli, Born stability |
| `bfdh` | d_eff with systematic absences, BFDH area fractions |
| `surfaces` | gamma (unrelaxed and relaxed) per face and termination |
| `wulff`, `face_ranking` | area fraction of each face family; faces by surface energy |

Surface energy is gamma = (E_slab - N e_bulk) / 2A, with slabs cut from the relaxed bulk.
`thickness_convergence` checks the number of layers and also gives the linear-fit gamma, which
does not depend on the bulk reference.

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

UMA `omc` results for anthracene (1 1 0), (0 0 1) and maleic anhydride (1 1 0), (1 0 1), (1 0 0),
(0 0 1), (1 1 1):

| Size | Effect on γ |
|---|---|
| vacuum 10–30 Å | none (6 Å cutoff, no long-range electrostatics) |
| in-plane 1×1 → 2×2, rattled | < 1 mJ/m²; no reconstruction found by local relaxation |
| 3 → 8 layers | 2–4 mJ/m² on the thinnest slabs (anthracene (1 1 0): 95.2 → 91.9), then within ~1 mJ/m² from 4–5 layers |
| relaxation scatter (rattled starts) | 1–4 mJ/m² |

The linear fit can differ from the direct value. For anthracene (1 1 0) it gives 96.1 against 92.6,
because molecules inside the slab sit 3–4 meV below the bulk reference. The likely cause is a small
dependence of UMA's energy on the cell, not yet tested. Read γ as ±2–4 mJ/m²: face orderings closer
than that are not resolved.

## Caveats

- **Disorder:** `disorder="major"` keeps groups `.`/`1`/`-1` (or occupancy >= 0.5 when there are no
  groups); check `source["dropped_disorder_sites"]`.
- **Hydrogens:** a CIF without H raises; `normalize_hydrogens=True` sets X-ray X-H bonds to neutron lengths.
- **Polar faces:** for non-centrosymmetric crystals the slab's two faces differ, so gamma is their
  average, and a net dipole across the slab can make energies converge slowly with vacuum.
- **Wulff:** `WulffShape` uses the lattice's point symmetry (the Laue class, except for
  pseudo-symmetric cells), and needs face families from enough directions to close the shape.

## Not yet included (natural next steps)

Attachment energies (growth morphology), surface relaxation with the in-plane cell free,
temperature (NPT MD lattice parameters, quasi-harmonic free energies), polymorph ranking,
and adhesion / work of separation of matched interfaces (`interfaces.separation_curve`
gives a rigid scan).
