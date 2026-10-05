"""Cached pair candidates and selection with any reactive radius / PairCriteria after the fact.

analyze_run(compute_pairs=True) stores every pair candidate inside its
reactive_radius (the pair model's center-distance screen, minimum image), with
the model's measured geometry metrics. select_pairs then applies a smaller
radius and/or any of the model's PairCriteria thresholds to those candidates,
so pair counts, distributions and lifetimes can be recomputed without
rereading the dump.

Candidate arrays (one row per candidate, frames delimited by offsets):
  offsets (frames+1), molecules (n, 2) [first, second member] indices into
  molecule_atom_ids (-1 if the molecule is not an initial molecule), shifts
  (n, 3) first-member periodic image shift, and metrics (n, k) with columns
  metric_columns(candidates): screen_distance, the model's metric_names, geometry_ok.
The dict also records reactive_radius and the pair model name.
"""
import numpy as np

from MartiniFuncs.pairs.model import get_pair_model


def metric_columns(candidates_or_model):
    """Metric column names of a candidate set (or a model): screen_distance, metrics, geometry_ok."""
    if isinstance(candidates_or_model, dict):
        names = candidates_or_model.get("metric_names")
        if names is None:
            names = get_pair_model(candidates_or_model.get("model")).metric_names
    else:
        names = get_pair_model(candidates_or_model).metric_names
    return ("screen_distance", *names, "geometry_ok")


def frame_candidates(rows, model=None):
    """Pack a frame's candidate rows [(first_mol, second_mol, shift, metrics), ...] into arrays."""
    width = len(metric_columns(model))
    molecules = np.array([r[:2] for r in rows], dtype=np.int32).reshape(-1, 2)
    shifts = np.array([r[2] for r in rows], dtype=np.int32).reshape(-1, 3)
    metrics = np.array([r[3] for r in rows], dtype=float).reshape(-1, width)
    return {"molecules": molecules, "shifts": shifts, "metrics": metrics}


def concatenate(per_frame, radius, model=None):
    """Join per-frame candidate dicts into the stored (offsets, molecules, shifts, metrics) form."""
    model = get_pair_model(model)
    counts = [len(c["molecules"]) for c in per_frame]
    empty = frame_candidates([], model)
    return {"offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
            "molecules": np.concatenate([c["molecules"] for c in per_frame] or [empty["molecules"]]),
            "shifts": np.concatenate([c["shifts"] for c in per_frame] or [empty["shifts"]]),
            "metrics": np.concatenate([c["metrics"] for c in per_frame] or [empty["metrics"]]),
            "reactive_radius": float(radius), "model": model.name,
            "metric_names": list(model.metric_names)}


def candidate_mask(candidates, *, reactive_radius=None, criteria=None):
    """Boolean mask over all candidate rows for a selection.

    screen_distance < reactive_radius (strict, as in the screen). With criteria
    (the model's PairCriteria, possibly all None): the geometry must have been
    measured, and each non-None threshold rejects values strictly above it
    (equality passes), exactly as PairCriteria.rejection_reasons.
    """
    metrics = np.asarray(candidates["metrics"])
    column = {name: metrics[:, i] for i, name in enumerate(metric_columns(candidates))}
    keep = np.ones(len(metrics), dtype=bool)
    if reactive_radius is not None:
        cached = candidates["reactive_radius"]
        if not 0 < reactive_radius <= cached:
            raise ValueError(f"reactive_radius must be in (0, {cached:g}] Angstrom, the radius cached by the analysis")
        keep &= column["screen_distance"] < reactive_radius
    if criteria is not None:
        keep &= column["geometry_ok"] > 0
        for metric, threshold in criteria.active().items():
            if metric not in column:
                raise ValueError(f"{type(criteria).__name__} thresholds metric {metric!r}, "
                                 f"which these candidates do not have")
            keep &= ~(column[metric] > threshold)
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
        raise ValueError("This cache has no pair candidates (made without pairs, or by a version that did not "
                         "store candidates); rerun the analysis to select pairs by radius/criteria")
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

    Columns: frame, time_ps, <role>_molecule for the model's two roles (for
    cpd_n4t: cpd_molecule, dienophile_molecule), shift_x/y/z and the metric
    columns. Use it for distributions (e.g. histogram screen_distance).
    """
    import pandas as pd
    candidates = result.get("pair_candidates")
    if candidates is None:
        raise ValueError("This cache has no pair candidates")
    roles = get_pair_model(candidates.get("model")).roles
    offsets = candidates["offsets"]
    frame_of_row = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    frames = np.asarray([r["frame"] for r in result["frames"]])
    times = np.asarray([r["time_ps"] for r in result["frames"]])
    table = pd.DataFrame({"frame": frames[frame_of_row], "time_ps": times[frame_of_row],
                          f"{roles[0]}_molecule": candidates["molecules"][:, 0],
                          f"{roles[1]}_molecule": candidates["molecules"][:, 1],
                          "shift_x": candidates["shifts"][:, 0], "shift_y": candidates["shifts"][:, 1],
                          "shift_z": candidates["shifts"][:, 2],
                          **{name: candidates["metrics"][:, i]
                             for i, name in enumerate(metric_columns(candidates))}})
    if reactive_radius is not None or criteria is not None:
        table = table[candidate_mask(candidates, reactive_radius=reactive_radius, criteria=criteria)]
    return table.reset_index(drop=True)


def metric_histograms(candidates, mask, *, bins=50):
    """Histogram rows (metric, bin_low, bin_high, count) of the selected candidates.

    Ranges: screen_distance (0, cached radius); metrics from the model's
    metric_ranges, else the range of the selected values.
    """
    metrics = np.asarray(candidates["metrics"])[mask]
    ranges = dict(get_pair_model(candidates.get("model")).metric_ranges,
                  screen_distance=(0, candidates["reactive_radius"]))
    rows = []
    for i, name in enumerate(metric_columns(candidates)[:-1]):
        values = metrics[:, i][np.isfinite(metrics[:, i])]
        counts, edges = np.histogram(values, bins=bins, range=ranges.get(name))
        rows += [dict(metric=name, bin_low=float(lo), bin_high=float(hi), count=int(n))
                 for lo, hi, n in zip(edges[:-1], edges[1:], counts)]
    return rows
