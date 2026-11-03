"""Collected pair candidates (stage 1) and selection from them (stage 2).

analyze_run(compute_pairs=True) collects every pair inside its PairCollection's
reactive_radius (the pair model's center-distance screen, minimum image) and
stores it with the model's measured metrics. select_pairs applies a model's
PairCriteria (a smaller max_screen_distance and/or geometric thresholds) to
those candidates, so pair counts, distributions and lifetimes can be recomputed
for any selection without rereading the dump.

Candidate arrays (one row per candidate, frames delimited by offsets):
  offsets (frames+1), molecules (n, 2) [first, second member] indices into
  molecule_atom_ids (-1 if the molecule is not an initial molecule), shifts
  (n, 3) first-member periodic image shift, and metrics (n, k) with columns
  metric_columns(candidates): screen_distance, the model's metric_names, geometry_ok.
The dict also records the collected reactive_radius and the pair model name.
"""
import numpy as np

from MartiniFuncs.pairs.model import PairCriteria, get_pair_model


def metric_columns(candidates_or_model):
    """Metric column names of a candidate set (or a model): screen_distance, metrics, geometry_ok."""
    if isinstance(candidates_or_model, dict):
        names = candidates_or_model.get("metric_names")
        if names is None:
            names = get_pair_model(candidates_or_model.get("model")).metric_names
    else:
        names = get_pair_model(candidates_or_model).metric_names
    return ("screen_distance", *names, "geometry_ok")


def frame_candidates(rows, model):
    """Pack a frame's candidate rows [(first_mol, second_mol, shift, metrics), ...] into arrays."""
    width = len(metric_columns(model))
    molecules = np.array([r[:2] for r in rows], dtype=np.int32).reshape(-1, 2)
    shifts = np.array([r[2] for r in rows], dtype=np.int32).reshape(-1, 3)
    metrics = np.array([r[3] for r in rows], dtype=float).reshape(-1, width)
    return {"molecules": molecules, "shifts": shifts, "metrics": metrics}


def concatenate(per_frame, radius, model):
    """Join per-frame candidate dicts into the stored form; radius is the collected reactive_radius."""
    model = get_pair_model(model)
    counts = [len(c["molecules"]) for c in per_frame]
    empty = frame_candidates([], model)
    return {"offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
            "molecules": np.concatenate([c["molecules"] for c in per_frame] or [empty["molecules"]]),
            "shifts": np.concatenate([c["shifts"] for c in per_frame] or [empty["shifts"]]),
            "metrics": np.concatenate([c["metrics"] for c in per_frame] or [empty["metrics"]]),
            "reactive_radius": float(radius), "model": model.name,
            "metric_names": list(model.metric_names)}


def candidate_mask(candidates, criteria=None):
    """Boolean mask over all candidate rows for a selection (criteria None: all rows).

    Exactly PairCriteria.rejection_reasons, vectorized: each non-None threshold
    rejects values strictly above it (equality passes); unmeasured geometry is
    rejected only when criteria.requires_geometry(). max_screen_distance may
    not exceed the collected reactive_radius.
    """
    metrics = np.asarray(candidates["metrics"])
    column = {name: metrics[:, i] for i, name in enumerate(metric_columns(candidates))}
    keep = np.ones(len(metrics), dtype=bool)
    if criteria is not None:
        model = get_pair_model(candidates.get("model"))
        if not isinstance(criteria, model.criteria_class) and type(criteria) is not PairCriteria:
            raise ValueError(f"criteria must be {model.criteria_class.__name__} (or the generic PairCriteria) "
                             f"for candidates of pair model {model.name!r}")
        collected = candidates["reactive_radius"]
        if criteria.max_screen_distance is not None and criteria.max_screen_distance > collected:
            raise ValueError(f"max_screen_distance must be at most {collected:g} Angstrom, "
                             "the reactive_radius collected by the analysis")
        if criteria.requires_geometry():
            keep &= column["geometry_ok"] > 0
        for metric, threshold in criteria.active().items():
            if metric not in column:
                raise ValueError(f"{type(criteria).__name__} thresholds metric {metric!r}, "
                                 f"which these candidates do not have")
            keep &= ~(column[metric] > threshold)
    return keep


def select_pairs(result, criteria=None):
    """Per-frame (pairs, shifts) selected by criteria (the model's PairCriteria), in the cached format.

    criteria=None returns every collected pair (pairs_by_frame and
    pair_shifts_by_frame). Otherwise the selection applies criteria to the
    cached candidates; max_screen_distance cannot exceed the collected radius.
    Pair order within a frame is screen distance, then molecule indices, as in
    the screen.
    """
    if criteria is None:
        return result["pairs_by_frame"], result["pair_shifts_by_frame"]
    candidates = result.get("pair_candidates")
    if candidates is None:
        raise ValueError("This cache has no pair candidates (made without pairs, or by a version that did not "
                         "store candidates); rerun the analysis to select pairs with criteria")
    keep = candidate_mask(candidates, criteria)
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


def candidate_table(result, criteria=None):
    """All collected candidates (optionally only a selection by criteria) as a pandas DataFrame.

    Columns: frame, time_ps, <role>_molecule for each of the model's two
    roles (e.g. diene_molecule, dienophile_molecule), shift_x/y/z and the metric
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
    if criteria is not None:
        table = table[candidate_mask(candidates, criteria)]
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
