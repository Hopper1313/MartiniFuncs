"""Cached pair candidates and selection with any reactive radius / PairCriteria after the fact.

analyze_run(compute_pairs=True) stores every CPD/dienophile candidate inside its
reactive_radius (the screen: CPD COM to dienophile C=C COM, minimum image), with
its measured geometry. select_pairs then applies a smaller radius and/or any
PairCriteria thresholds to those candidates, so pair counts, distributions and
lifetimes can be recomputed without rereading the dump.

Candidate arrays (one row per candidate, frames delimited by offsets):
  offsets (frames+1), molecules (n, 2) [CPD, dienophile] indices into
  molecule_atom_ids (-1 if the molecule is not an initial molecule), shifts
  (n, 3) CPD periodic image shift, and metrics (n, 7) with columns METRICS.
"""
from dataclasses import fields

import numpy as np

METRICS = ("screen_distance", "max_forming_distance", "distance_imbalance",
           "axis_angle", "plane_angle", "approach_angle", "geometry_ok")
# PairCriteria field -> metric column it thresholds
CRITERIA_COLUMNS = {"max_forming_distance": "max_forming_distance",
                    "max_distance_imbalance": "distance_imbalance",
                    "max_axis_angle": "axis_angle", "max_plane_angle": "plane_angle",
                    "max_approach_angle": "approach_angle"}


def metric_row(distance, geometry):
    """One metrics row from a screen distance and a PairGeometry (or None)."""
    if geometry is None:
        return [float(distance)] + [np.nan] * 5 + [0.]
    return [float(distance), max(geometry.forming_distances), geometry.distance_imbalance,
            geometry.axis_angle, geometry.plane_angle, geometry.approach_angle, 1.]


def frame_candidates(rows):
    """Pack a frame's candidate rows [(cpd_mol, mal_mol, shift, metrics), ...] into arrays."""
    molecules = np.array([r[:2] for r in rows], dtype=np.int32).reshape(-1, 2)
    shifts = np.array([r[2] for r in rows], dtype=np.int32).reshape(-1, 3)
    metrics = np.array([r[3] for r in rows], dtype=float).reshape(-1, len(METRICS))
    return {"molecules": molecules, "shifts": shifts, "metrics": metrics}


def concatenate(per_frame, radius):
    """Join per-frame candidate dicts into the stored (offsets, molecules, shifts, metrics) form."""
    counts = [len(c["molecules"]) for c in per_frame]
    empty = frame_candidates([])
    return {"offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
            "molecules": np.concatenate([c["molecules"] for c in per_frame] or [empty["molecules"]]),
            "shifts": np.concatenate([c["shifts"] for c in per_frame] or [empty["shifts"]]),
            "metrics": np.concatenate([c["metrics"] for c in per_frame] or [empty["metrics"]]),
            "reactive_radius": float(radius)}


def candidate_mask(candidates, *, reactive_radius=None, criteria=None):
    """Boolean mask over all candidate rows for a selection.

    screen_distance < reactive_radius (strict, as in the screen). With criteria
    (a PairCriteria, possibly all None): the geometry must have been measured,
    and each non-None threshold rejects values strictly above it (equality
    passes), exactly as pair_selection._rejection_reasons.
    """
    metrics = np.asarray(candidates["metrics"])
    column = {name: metrics[:, i] for i, name in enumerate(METRICS)}
    keep = np.ones(len(metrics), dtype=bool)
    if reactive_radius is not None:
        cached = candidates["reactive_radius"]
        if not 0 < reactive_radius <= cached:
            raise ValueError(f"reactive_radius must be in (0, {cached:g}] Angstrom, the radius cached by the analysis")
        keep &= column["screen_distance"] < reactive_radius
    if criteria is not None:
        keep &= column["geometry_ok"] > 0
        for field in fields(criteria):
            threshold = getattr(criteria, field.name)
            if threshold is not None:
                keep &= ~(column[CRITERIA_COLUMNS[field.name]] > threshold)
    return keep


def select_pairs(result, *, reactive_radius=None, criteria=None):
    """Per-frame (pairs, shifts) for a selection, in the cached pairs' format.

    With neither option, returns the analysis's own pairs_by_frame and
    pair_shifts_by_frame (its radius and criteria). Otherwise the selection
    starts from all cached candidates, so it can be looser or stricter than the
    analysis criteria, but never wider than the cached radius. Pair order within
    a frame is screen distance, then molecule indices, as in the screen.
    """
    if reactive_radius is None and criteria is None:
        return result["pairs_by_frame"], result["pair_shifts_by_frame"]
    candidates = result.get("pair_candidates")
    if candidates is None:
        raise ValueError("This cache has no pair candidates (made before 2026-10-01 or without pairs); "
                         "rerun the analysis to select pairs by radius/criteria")
    keep = candidate_mask(candidates, reactive_radius=reactive_radius, criteria=criteria)
    if np.any(candidates["molecules"][keep] < 0):
        raise ValueError("Selected candidates include molecules that are not initial molecules")
    ids = [list(map(int, m)) for m in result["molecule_atom_ids"]]
    offsets = candidates["offsets"]
    pairs, shifts = [], []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        rows = np.flatnonzero(keep[start:stop]) + start
        pairs.append([[ids[a], ids[b]] for a, b in candidates["molecules"][rows].tolist()])
        shifts.append(candidates["shifts"][rows].tolist())
    return pairs, shifts


def candidate_table(result, *, reactive_radius=None, criteria=None):
    """All cached candidates (optionally only a selection) as a pandas DataFrame.

    Columns: frame, time_ps, cpd_molecule, dienophile_molecule, shift_x/y/z and
    METRICS. Use it for distributions (e.g. histogram screen_distance).
    """
    import pandas as pd
    candidates = result.get("pair_candidates")
    if candidates is None:
        raise ValueError("This cache has no pair candidates")
    offsets = candidates["offsets"]
    frame_of_row = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    frames = np.asarray([r["frame"] for r in result["frames"]])
    times = np.asarray([r["time_ps"] for r in result["frames"]])
    table = pd.DataFrame({"frame": frames[frame_of_row], "time_ps": times[frame_of_row],
                          "cpd_molecule": candidates["molecules"][:, 0],
                          "dienophile_molecule": candidates["molecules"][:, 1],
                          "shift_x": candidates["shifts"][:, 0], "shift_y": candidates["shifts"][:, 1],
                          "shift_z": candidates["shifts"][:, 2],
                          **{name: candidates["metrics"][:, i] for i, name in enumerate(METRICS)}})
    if reactive_radius is not None or criteria is not None:
        table = table[candidate_mask(candidates, reactive_radius=reactive_radius, criteria=criteria)]
    return table.reset_index(drop=True)


def metric_histograms(candidates, mask, *, bins=50):
    """Histogram rows (metric, bin_low, bin_high, count) of the selected candidates."""
    metrics = np.asarray(candidates["metrics"])[mask]
    ranges = {"screen_distance": (0, candidates["reactive_radius"]), "max_forming_distance": (0, 8),
              "distance_imbalance": (0, 4), "axis_angle": (0, 90), "plane_angle": (0, 90), "approach_angle": (0, 90)}
    rows = []
    for i, name in enumerate(METRICS[:-1]):
        values = metrics[:, i][np.isfinite(metrics[:, i])]
        counts, edges = np.histogram(values, bins=bins, range=ranges[name])
        rows += [dict(metric=name, bin_low=float(lo), bin_high=float(hi), count=int(n))
                 for lo, hi, n in zip(edges[:-1], edges[1:], counts)]
    return rows
