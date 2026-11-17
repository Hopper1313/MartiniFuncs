"""Diels-Alder alignment across an anthracene / maleic anhydride crystal interface.

In an interface from crystals.interfaces.build_interface, the contact layers
are anthracene's top layer (slab 0) and maleic anhydride's bottom layer
(slab 1). For each molecule there the reacting group is reduced to a site:
its midpoint and in-plane axis, C9...C10 for anthracene (the two carbons
that bond) and C=C for maleic anhydride (sites.anthracene_sites /
anhydride_sites, from bonding only).

An anthracene site is *lined up* when some C=C midpoint lies within
`threshold` (Angstrom) of its C9/C10 midpoint in plane (periodic) and the two
axes are within `max_angle` degrees of parallel, the geometry of the
Diels-Alder approach, where C9 and C10 bond to the two alkene carbons.
*Orientation-compatible* sites have a C=C within max_angle anywhere in the
cell: that count is fixed by the lattice match (its rotation) and no lateral
shift can change it. The lateral registry, build_interface(shift=...), decides
how many of those are lined up; best_registry scans it on a grid of the cell.

This is a geometric screen, not an energy: it ignores the vertical
separation (about 8 Angstrom between the groups at a 3 Angstrom atom-atom gap
for the (1 1 0)/(1 1 0) contact) and any relaxation.

    contact_sites(interface)           -> (anthracene sites, dienophile sites), each N x 5:
                                          x, y, z of the midpoint and the in-plane axis
    alignment(interface, shift=(0, 0)) -> lined_up, orientation_compatible, n_sites, mean_offset
    best_registry(interface)           -> best shift on a grid and its alignment
    contact_alignment(interface)       -> both, plus "score" (lined-up fraction at the best
                                          registry): a scorer for crystals.interface_options
"""
from __future__ import annotations

import numpy as np

from MartiniFuncs.chemistry.anthracene_ma.sites import anhydride_sites, anthracene_sites
from MartiniFuncs.structure.graph import build_bond_graph


def contact_sites(interface):
    """Sites of the two contact layers: (anthracene C9...C10, maleic anhydride C=C), each an N x 5 array."""
    graph = build_bond_graph(interface, index_array=None)
    els = dict(enumerate(interface.get_chemical_symbols()))
    mol, slab, layer, pos = (interface.arrays[k] for k in ("mol_id", "slab", "layer", "positions"))
    top_a = layer[slab == 0].max()
    diene, dienophile = [], []
    for k in np.unique(mol):
        idx = np.flatnonzero(mol == k)
        s, ly = slab[idx[0]], layer[idx[0]]
        if s == 0 and ly == top_a:
            pair, _ = anthracene_sites(graph, els, idx.tolist())
            target = diene
        elif s == 1 and ly == 0:
            pair, _, _ = anhydride_sites(graph, els, idx.tolist())
            target = dienophile
        else:
            continue
        p, q = pos[pair[0]], pos[pair[1]]
        target.append([*((p + q) / 2), *(q - p)[:2]])
    if not diene or not dienophile:
        raise ValueError("no anthracene C9/C10 or maleic anhydride C=C sites in the contact layers")
    return np.array(diene), np.array(dienophile)


def _axis_angles(diene, dienophile):
    u, v = diene[:, 3:5], dienophile[:, 3:5]
    cos = np.abs(u @ v.T) / (np.linalg.norm(u, axis=1)[:, None] * np.linalg.norm(v, axis=1)[None, :])
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))            # (n_diene, n_dienophile), 0..90


def _offsets(diene, dienophile, cell2, shifts):
    """In-plane periodic distances (n_shifts, n_diene, n_dienophile) for dienophile shifts (fractions of cell2)."""
    inv = np.linalg.inv(cell2)
    d0 = (dienophile[None, :, :2] - diene[:, None, :2]) @ inv                  # fractional, (nA, nB, 2)
    f = d0[None] + np.asarray(shifts, float)[:, None, None, :]
    f -= np.round(f)
    return np.linalg.norm(f @ cell2, axis=-1)


def _score(diene, dienophile, cell2, shifts, threshold, max_angle):
    aligned = _axis_angles(diene, dienophile) <= max_angle
    dist = _offsets(diene, dienophile, cell2, shifts)
    lined = ((dist <= threshold) & aligned[None]).any(axis=2).sum(axis=1)
    mean = dist.min(axis=2).mean(axis=1)
    return lined, mean, int(aligned.any(axis=1).sum())


def alignment(interface, shift=(0.0, 0.0), threshold=1.5, max_angle=30.0, sites=None):
    """Alignment of the contact layers with the dienophile layer shifted by `shift` (fractions of the cell)."""
    diene, dienophile = contact_sites(interface) if sites is None else sites
    cell2 = interface.cell.array[:2, :2]
    lined, mean, compatible = _score(diene, dienophile, cell2, [shift], threshold, max_angle)
    return {"lined_up": int(lined[0]), "orientation_compatible": compatible, "n_sites": len(diene),
            "mean_offset": float(mean[0]), "shift": [float(x) for x in shift],
            "threshold": threshold, "max_angle": max_angle}


def best_registry(interface, threshold=1.5, max_angle=30.0, grid=100, sites=None):
    """The lateral shift (on a grid x grid mesh of the cell, relative to this interface) that lines up most sites.

    Ties go to the smaller mean in-plane offset. The returned shift is relative to
    the interface as built: for build_interface(..., shift=s0) the shift to use is
    s0 + result["shift"] (mod 1); for an interface built with the default shift it is
    result["shift"] itself.
    """
    diene, dienophile = contact_sites(interface) if sites is None else sites
    cell2 = interface.cell.array[:2, :2]
    g = np.arange(grid) / grid
    shifts = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    best = None
    for chunk in np.array_split(shifts, max(1, len(shifts) // 400)):
        lined, mean, compatible = _score(diene, dienophile, cell2, chunk, threshold, max_angle)
        order = np.lexsort((mean, -lined))
        k = order[0]
        if best is None or (lined[k], -mean[k]) > (best[0], -best[1]):
            best = (int(lined[k]), float(mean[k]), chunk[k])
    return {"shift": [round(float(x), 6) for x in best[2]], "lined_up": best[0], "mean_offset": best[1],
            "orientation_compatible": compatible, "n_sites": len(diene), "threshold": threshold,
            "max_angle": max_angle, "grid": grid}


def contact_alignment(interface, threshold=1.5, max_angle=30.0, grid=100):
    """Alignment as built and at the best registry; "score" is the lined-up fraction at the best registry.

    Use as a scorer: crystals.interface_options(..., scorer=contact_alignment, sort="score").
    """
    sites = contact_sites(interface)
    here = alignment(interface, (0.0, 0.0), threshold, max_angle, sites)
    best = best_registry(interface, threshold, max_angle, grid, sites)
    n = best["n_sites"]
    return {"score": best["lined_up"] / n + 1e-3 * best["orientation_compatible"] / n,
            "n_sites": n, "orientation_compatible": best["orientation_compatible"],
            "lined_up_as_built": here["lined_up"], "lined_up_best": best["lined_up"],
            "best_shift": best["shift"], "mean_offset_best": best["mean_offset"],
            "threshold": threshold, "max_angle": max_angle}


def best_shift(interface, **kwargs):
    """Just the best registry shift: for InterfaceOption.build(shift=best_shift)."""
    return tuple(best_registry(interface, **kwargs)["shift"])
