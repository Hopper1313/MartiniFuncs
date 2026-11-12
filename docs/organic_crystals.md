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
`rigid_molecules=True` only molecular centres move with the strain, so no bond is stretched.

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
