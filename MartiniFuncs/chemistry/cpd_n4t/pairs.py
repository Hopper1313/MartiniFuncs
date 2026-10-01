"""CPD + N4T pair screening: molecule classes, dienophile C=C sites and pair geometry.
"""
from __future__ import annotations

from itertools import zip_longest

import numpy as np
from ase import Atoms

from MartiniFuncs.pairs.lifetimes import pair_identity
from MartiniFuncs.statistics.md import _times
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.structure.molecules import add_global_index


def get_mal_CC(bond_graph, elements):
    """Return the maleimide C=C atoms from element-pattern intersections."""
    hcch = mfg.find_element_chains(bond_graph, elements, "HCCH")
    occcco = mfg.find_element_chains(bond_graph, elements, "OCCCCO")
    hcch_atoms = {i for chain in hcch for i in chain}
    occcco_atoms = {i for chain in occcco for i in chain}
    return hcch_atoms & occcco_atoms


@add_global_index
def get_all_prereactive_pairs(
    atoms: Atoms,
    scale: float = 1.3,
    reactive_radius: float = 3.25,
    printing: bool = False,
    *,
    criteria=None,
    return_candidates=False,
):
    """Find all CPD / dienophile molecular pairs inside ``reactive_radius``.

    Returns the same API as the historical implementation:

    ``pairs``
        ``[[cpd_atom_ids, mal_atom_ids], ...]`` using persistent atom IDs.

    ``cpd_shift``
        Integer lattice translation to apply to the *unwrapped CPD molecule*
        so it lies beside the corresponding unwrapped MAL molecule.

    Unlike the previous implementation, molecular and reactive-site centroids
    are calculated in unwrapped molecular coordinate frames, so the function
    does not require ``repair_fragmented_molecules`` first.

    ``criteria`` optionally accepts CpdN4tCriteria. The default preserves the
    distance screen; use assess_prereactive_pairs separately to retain reasons
    and geometry measurements for rejected candidates.

    Screen d=min_n |R_CPD+n @ cell-R_CC| < reactive_radius, where
    R_CPD is the mass-weighted COM of all CPD atoms and R_CC is the
    mass-weighted COM of the dienophile reactive C=C. Both are computed from
    whole molecules. Keep the nearest periodic image; returned shift is
    S+mal_image-cpd_image. This is not a forming-bond distance criterion.
    The generic engine is pairs.assessment.screen_frame with the cpd_n4t model.
    """
    from MartiniFuncs.pairs.assessment import screen_frame
    found = screen_frame(atoms, "cpd_n4t", scale=scale, reactive_radius=reactive_radius, criteria=criteria,
                         return_candidates=return_candidates)
    if printing:
        print(len(found[0]), "pre-reactive pairs found")
    return found


def get_pair_geometry(atoms, pair, shift, *, scale=1.3):
    """Existing CPD/dienophile diagnostics, distances in Angstrom, angles in degrees.

    Delegates to chemistry.cpd_n4t.selection.measure_pair_geometry: bond distances
    are Euclidean in one placed image; angles are degrees(arccos(|a dot b| /
    (|a||b|))). See that function for the reactive axes and plane definitions.
    """
    from MartiniFuncs.chemistry.cpd_n4t.selection import measure_pair_geometry
    return measure_pair_geometry(atoms, pair, shift, scale=scale)


def get_pair_geometry_timeseries(atoms_by_frame, pairs_by_frame, shifts_by_frame,
                                 times_ps, *, pair, scale=1.3):
    """Track one pair's existing geometry diagnostics across synchronized frames.

    Pair/shift lists are the outputs of get_all_prereactive_pairs for each frame.
    Absence from the candidate list returns geometry=None; invalid geometry has
    an explicit error string. Absence is not evidence of molecular destruction.
    No threshold or orientation criterion is changed here.
    """
    key = pair_identity(pair)
    times = np.asarray(times_ps, dtype=float)
    _times(times, len(times))
    result, sentinel = [], object()
    for atoms, pairs, shifts, time in zip_longest(
            atoms_by_frame, pairs_by_frame, shifts_by_frame, times, fillvalue=sentinel):
        if any(value is sentinel for value in (atoms, pairs, shifts, time)):
            raise ValueError("Supply geometry, pairs, shifts, and time for every frame")
        if len(pairs) != len(shifts):
            raise ValueError("Each candidate pair requires one periodic shift")
        matches = [(p, s) for p, s in zip(pairs, shifts) if pair_identity(p) == key]
        if len(matches) > 1:
            raise ValueError("Duplicate pair identities in a frame")
        geometry, error = None, None
        if matches:
            try:
                geometry = get_pair_geometry(atoms, *matches[0], scale=scale)
            except ValueError as exc:
                error = str(exc)
        result.append({"time_ps": float(time), "pair": key, "present": bool(matches),
                       "geometry": geometry, "error": error})
    return result
