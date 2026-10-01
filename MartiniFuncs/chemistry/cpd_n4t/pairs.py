"""CPD + N4T pair screening: molecule classes, dienophile C=C sites and pair geometry.
"""
from __future__ import annotations

from itertools import zip_longest

import numpy as np
from ase import Atoms

from MartiniFuncs.pairs.lifetimes import pair_identity
from MartiniFuncs.pairs.screening import _pairs_from_centers
from MartiniFuncs.statistics.md import _times
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.structure.molecules import _site_center_of_mass, add_global_index, get_element_map


def get_mal_CC(bond_graph, elements):
    """Return the maleimide C=C atoms from element-pattern intersections."""
    hcch = mfg.find_element_chains(bond_graph, elements, "HCCH")
    occcco = mfg.find_element_chains(bond_graph, elements, "OCCCCO")
    hcch_atoms = {i for chain in hcch for i in chain}
    occcco_atoms = {i for chain in occcco for i in chain}
    return hcch_atoms & occcco_atoms


def _mal_site_centers(atoms: Atoms, graph, mal_components, elements) -> np.ndarray:
    """Reactive C=C COM of each dienophile, in its unwrapped molecular frame.

    Sites come from the molecule's induced subgraph of the frame bond graph,
    which equals a bond graph rebuilt on the isolated molecule.
    """
    centers = []
    for comp in mal_components:
        mal_cc = get_mal_CC(mfg.induced_subgraph(graph, comp), elements)
        if len(mal_cc) != 2:
            raise ValueError(
                f"Could not identify MAL reactive C=C for component {comp[:5]}..."
            )
        centers.append(_site_center_of_mass(atoms, graph, comp, mal_cc))
    return np.asarray(centers)


def _reaction_components(atoms: Atoms, scale: float):
    graph = mfg.build_bond_graph(
        atoms,
        scale=scale,
        index_array="global_index",
    )
    components = mfg.connected_components(
        graph,
        atom_ids=mfg.atom_ids(atoms),
    )

    cpd = [c for c in components if len(c) == 11]
    mal = [c for c in components if len(c) == 23]
    product = [c for c in components if len(c) == 34]
    other = [c for c in components if len(c) not in (11, 23, 34)]
    return graph, components, cpd, mal, product, other


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

    ``criteria`` optionally accepts PairCriteria. The default preserves the
    distance screen; use assess_prereactive_pairs separately to retain reasons
    and geometry measurements for rejected candidates.

    Screen d=min_n |R_CPD+n @ cell-R_CC| < reactive_radius, where
    R_CPD is the mass-weighted COM of all CPD atoms and R_CC is the
    mass-weighted COM of the dienophile reactive C=C. Both are computed from
    whole molecules. Keep the nearest periodic image; returned shift is
    S+mal_image-cpd_image. This is not a forming-bond distance criterion.
    """
    if not np.isfinite(reactive_radius) or reactive_radius <= 0:
        raise ValueError("reactive_radius must be > 0")

    elements = get_element_map(atoms, index_array="global_index")
    graph, _, cpd_components, mal_components, _, other = _reaction_components(
        atoms, scale
    )

    if other:
        raise ValueError(
            f"There are {len(other)} molecules that do not match expected sizes "
            "(11, 23, 34 atoms). Check bond scale / chemistry."
        )

    # CPD search point = COM of the entire unwrapped CPD molecule.
    cpd_centers = np.asarray([
        mfg.component_center_of_mass(atoms, graph, comp)
        for comp in cpd_components
    ])

    # MAL search point = COM of the reactive C=C in the unwrapped MAL frame.
    mal_centers = _mal_site_centers(atoms, graph, mal_components, elements)

    if printing:
        print(len(mal_centers), "mal CC double bonds found")
        print(len(cpd_centers), "CPD molecules found")

    if len(cpd_centers) == 0 or len(mal_centers) == 0:
        return ([], [], []) if return_candidates else ([], [])

    pairs, cpd_shift, distances = _pairs_from_centers(
        cpd_centers, mal_centers, cpd_components, mal_components,
        atoms.cell.array, atoms.pbc, reactive_radius, return_distances=True)

    candidates = None
    if criteria is not None or return_candidates:
        from MartiniFuncs.chemistry.cpd_n4t.selection import PairCriteria, assess_prereactive_pairs
        assessments = assess_prereactive_pairs(atoms, pairs, cpd_shift, graph=graph,
                                               criteria=criteria if criteria is not None else PairCriteria(),
                                               scale=scale)
        # Every screened candidate with its distance and measured geometry (or None).
        candidates = [(pair, shift, distance, a.geometry)
                      for pair, shift, distance, a in zip(pairs, cpd_shift, distances, assessments)]
        if criteria is not None:
            keep = [a.pair_index for a in assessments if a.accepted]
            pairs, cpd_shift = [pairs[i] for i in keep], [cpd_shift[i] for i in keep]

    if printing:
        print(len(pairs), "pre-reactive pairs found")
    if return_candidates:
        return pairs, cpd_shift, candidates

    return pairs, cpd_shift


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
