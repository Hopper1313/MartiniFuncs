"""Generic center-distance pair screen with periodic image bookkeeping.
"""
from __future__ import annotations
import numpy as np
from ase.neighborlist import primitive_neighbor_list
from MartiniFuncs.structure.periodic import wrap_points


def pairs_from_centers(first_centers, second_centers, first_components, second_components,
                       cell, pbc, reactive_radius, *, return_distances=False):
    """Center-distance screen between the two pair roles, nearest image per pair.

    Select |C_first + shift @ cell - C_second| < reactive_radius (Angstrom).
    Wrapped neighbor shift S becomes shift=S+second_image-first_image for whole
    molecules; the shift applies to the first member. Sort by distance, then
    second and first component indices. return_distances=True also returns each
    pair's screen distance (Angstrom).
    """
    if not len(first_centers) or not len(second_centers):
        return ([], [], []) if return_distances else ([], [])
    # Put both point sets into one primary periodic representation while
    # remembering which image was removed from each unwrapped centroid.
    second_wrapped, second_image = wrap_points(second_centers, cell, pbc)
    first_wrapped, first_image = wrap_points(first_centers, cell, pbc)

    nsecond = len(second_wrapped)
    nfirst = len(first_wrapped)
    positions = np.vstack((second_wrapped, first_wrapped))

    # Dummy species labels let ASE discard same-role pairs at cutoff
    # evaluation time. Only 1-2 contacts are relevant.
    numbers = np.concatenate((
        np.ones(nsecond, dtype=int),
        np.full(nfirst, 2, dtype=int),
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

    mask = (i < nsecond) & (j >= nsecond)
    second_i = i[mask]
    first_i = j[mask] - nsecond
    distances = d[mask]
    relative_shifts = S[mask]

    if len(second_i) == 0:
        return ([], [], []) if return_distances else ([], [])

    # Multiple periodic images of the same molecular pair can in principle be
    # inside a large cutoff. Retain only the closest image for each pair.
    best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for m, c, dist, shift in zip(second_i, first_i, distances, relative_shifts):
        key = (int(m), int(c))
        prev = best.get(key)
        if prev is None or float(dist) < prev[0]:
            # Unwrapped centroids satisfy:
            #   C_unwrapped = C_wrapped + first_image @ cell
            # The translation applied to the unwrapped first member is therefore:
            #   T = S + second_image - first_image
            pair_shift = (
                np.asarray(shift, dtype=int)
                + second_image[int(m)]
                - first_image[int(c)]
            )
            best[key] = (float(dist), pair_shift)

    # Deterministic order: distance, then second/first component indices.
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1]))

    pairs = [
        [list(first_components[c]), list(second_components[m])]
        for (m, c), _ in ordered
    ]
    shifts = [value[1].tolist() for _, value in ordered]
    if return_distances:
        return pairs, shifts, [value[0] for _, value in ordered]
    return pairs, shifts
