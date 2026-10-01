"""Generic center-distance pair screen with periodic image bookkeeping.
"""
from __future__ import annotations
import numpy as np
from ase.neighborlist import primitive_neighbor_list
from MartiniFuncs.structure.molecules import _wrap_points


def _pairs_from_centers(cpd_centers, mal_centers, cpd_components, mal_components,
                        cell, pbc, reactive_radius, *, return_distances=False):
    """Shared center-distance screen, preserving image choice and candidate order.

    Select |C_CPD + shift @ cell - C_CC| < reactive_radius (Angstrom).
    Wrapped neighbor shift S becomes shift=S+mal_image-cpd_image for whole
    molecules. Sort by distance, then MAL and CPD component indices. return_distances=True
    also returns each pair's screen distance (Angstrom).
    """
    if not len(cpd_centers) or not len(mal_centers):
        return ([], [], []) if return_distances else ([], [])
    # Put both point sets into one primary periodic representation while
    # remembering which image was removed from each unwrapped centroid.
    mal_wrapped, mal_image = _wrap_points(mal_centers, cell, pbc)
    cpd_wrapped, cpd_image = _wrap_points(cpd_centers, cell, pbc)

    nmal = len(mal_wrapped)
    ncpd = len(cpd_wrapped)
    positions = np.vstack((mal_wrapped, cpd_wrapped))

    # Dummy species labels let ASE discard MAL-MAL / CPD-CPD pairs at cutoff
    # evaluation time. Only 1-2 contacts are relevant.
    numbers = np.concatenate((
        np.ones(nmal, dtype=int),
        np.full(ncpd, 2, dtype=int),
    ))

    i, j, d, S = primitive_neighbor_list(
        "ijdS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff={(1, 2): float(reactive_radius)},
        numbers=numbers,
        self_interaction=False,
    )

    mask = (i < nmal) & (j >= nmal)
    mal_i = i[mask]
    cpd_i = j[mask] - nmal
    distances = d[mask]
    relative_shifts = S[mask]

    if len(mal_i) == 0:
        return ([], [], []) if return_distances else ([], [])

    # Multiple periodic images of the same molecular pair can in principle be
    # inside a large cutoff. Retain only the closest image for each pair.
    best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for m, c, dist, shift in zip(mal_i, cpd_i, distances, relative_shifts):
        key = (int(m), int(c))
        prev = best.get(key)
        if prev is None or float(dist) < prev[0]:
            # Unwrapped centroids satisfy:
            #   C_unwrapped = C_wrapped + cpd_image @ cell
            # The translation applied to the unwrapped CPD is therefore:
            #   T = S + mal_image - cpd_image
            pair_shift = (
                np.asarray(shift, dtype=int)
                + mal_image[int(m)]
                - cpd_image[int(c)]
            )
            best[key] = (float(dist), pair_shift)

    # Deterministic order: distance, then MAL/CPD component indices.
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1]))

    pairs = [
        [list(cpd_components[c]), list(mal_components[m])]
        for (m, c), _ in ordered
    ]
    cpd_shift = [value[1].tolist() for _, value in ordered]
    if return_distances:
        return pairs, cpd_shift, [value[0] for _, value in ordered]
    return pairs, cpd_shift
