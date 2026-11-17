"""Which faces: symmetry-distinct Miller indices, BFDH ranking and Wulff shapes.

distinct_miller_indices   pymatgen get_symmetrically_distinct_miller_indices (spglib symmetry
                          of the structure) up to max_index
systematically_absent     h is absent when some operator (R, t) has h R = h and h.t not an
                          integer (screw axes, glide planes, centring)
bfdh                      Bravais-Friedel-Donnay-Harker ranking: for each face the effective
                          spacing d_eff = d_hkl / n with n the smallest multiple of (h k l)
                          that is not absent; larger d_eff = slower growth = more
                          important face. The BFDH area fractions are those of a Wulff
                          shape with "energies" 1 / d_eff.
wulff                     pymatgen WulffShape from {hkl: surface energy}: area fraction of each
                          face family, shape factor, anisotropy, weighted surface energy
attachment_energy_shape   growth morphology: the same construction with |E_att| of each face's
                          most strongly bound slice (surfaces.attachment_energies)

BFDH needs no calculator and is the quick screen for which faces to compute;
surface energies (surfaces.py) then give the Wulff (equilibrium) shape. Note
that WulffShape applies the point symmetry of the lattice, which equals the
Laue class of the crystal except for pseudo-symmetric cells.
"""
from __future__ import annotations

import numpy as np

from .cif import MolecularCrystal
from .slabs import d_spacing, primitive_miller


def _structure(crystal):
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = crystal.atoms.copy()
    atoms.arrays = {k: v for k, v in atoms.arrays.items() if k in ("numbers", "positions")}
    atoms.wrap()
    return AseAtomsAdaptor.get_structure(atoms)


def distinct_miller_indices(crystal: MolecularCrystal, max_index=2):
    """Symmetry-distinct primitive Miller indices up to max_index (pymatgen, spglib symmetry)."""
    from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
    found = get_symmetrically_distinct_miller_indices(_structure(crystal), int(max_index))
    return sorted({primitive_miller(h) for h in found}, key=lambda h: (sum(map(abs, h)), [-x for x in h]))


def systematically_absent(hkl, symmetry, tol=1e-6):
    """True when reflection hkl is extinct by the symmetry operators (Symmetry from read_cif)."""
    h = np.asarray(hkl, dtype=float)
    for R, t in zip(symmetry.rotations, symmetry.translations):
        if np.allclose(h @ R, h) and abs(h @ t - round(h @ t)) > tol:
            return True
    return False


def bfdh(crystal: MolecularCrystal, max_index=2, *, indices=None, max_multiple=6):
    """Faces ranked by BFDH effective interplanar spacing, largest first.

    Returns a list of dicts: hkl, d_hkl, multiple n, d_eff (Angstrom),
    relative growth rate d_eff_max / d_eff, and the BFDH area_fraction.
    """
    cell = crystal.atoms.cell.array
    faces = []
    for hkl in indices or distinct_miller_indices(crystal, max_index):
        n = 1
        if crystal.symmetry is not None:
            while n < max_multiple and systematically_absent(np.multiply(n, hkl), crystal.symmetry):
                n += 1
        d = d_spacing(cell, hkl)
        faces.append({"hkl": tuple(hkl), "d_hkl": d, "multiple": n, "d_eff": d / n})
    faces.sort(key=lambda f: -f["d_eff"])
    top = faces[0]["d_eff"]
    shape = wulff(crystal, {f["hkl"]: 1.0 / f["d_eff"] for f in faces})
    for f in faces:
        f["relative_growth_rate"] = top / f["d_eff"]
        f["area_fraction"] = shape["area_fractions"].get(f["hkl"], 0.0)
    return faces


def wulff(crystal_or_lattice, energies, *, symprec=1e-5):
    """Wulff shape from {hkl: energy}; returns area fractions per family and shape descriptors."""
    from pymatgen.analysis.wulff import WulffShape
    from pymatgen.core import Lattice
    cell = crystal_or_lattice.atoms.cell.array if isinstance(crystal_or_lattice, MolecularCrystal) \
        else np.asarray(crystal_or_lattice)
    millers = [tuple(int(x) for x in h) for h in energies]
    try:
        shape = WulffShape(Lattice(cell), millers, [float(energies[h]) for h in energies], symprec=symprec)
    except Exception as error:                     # scipy QhullError and friends: the planes do not enclose a body
        raise ValueError(f"faces {millers} (with their symmetry equivalents) do not enclose a closed Wulff "
                         f"shape; add faces from other directions ({type(error).__name__})") from error
    fractions = {tuple(k): float(v) for k, v in shape.area_fraction_dict.items()}
    return {
        "area_fractions": fractions, "shape": shape, "shape_factor": float(shape.shape_factor),
        "anisotropy": float(shape.anisotropy), "weighted_surface_energy": float(shape.weighted_surface_energy),
        "expressed": sorted((h for h, a in fractions.items() if a > 1e-6), key=lambda h: -fractions[h]),
    }


def attachment_energy_shape(crystal_or_lattice, rows, *, symprec=1e-5):
    """Growth morphology in the attachment-energy model.

    rows: the output of surfaces.attachment_energies (or {hkl: E_att}). For each
    face the termination with the smallest |E_att| (the most strongly bound slice)
    sets the growth rate R_hkl ~ |E_att|, and the shape is the Wulff construction
    with |E_att| in place of the surface energy. Returns wulff()'s dict plus
    "chosen": {hkl: row of the termination used} when rows are given.
    """
    if isinstance(rows, dict):
        best = {tuple(int(x) for x in h): abs(float(e)) for h, e in rows.items()}
        chosen = None
    else:
        chosen = {}
        for r in rows:
            h = tuple(int(x) for x in r["hkl"])
            if h not in chosen or abs(r["attachment_energy_eV"]) < abs(chosen[h]["attachment_energy_eV"]):
                chosen[h] = r
        best = {h: abs(r["attachment_energy_eV"]) for h, r in chosen.items()}
    out = wulff(crystal_or_lattice, best, symprec=symprec)
    if chosen is not None:
        out["chosen"] = chosen
    return out
