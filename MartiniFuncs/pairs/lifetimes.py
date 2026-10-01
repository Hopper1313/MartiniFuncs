"""Pair identities and encounter lifetimes from per-frame pair lists (any pair model).
"""
from dataclasses import dataclass
from itertools import zip_longest
import numpy as np
from MartiniFuncs.statistics.md import _times


def pair_identity(pair):
    """Canonical unordered pair of molecule atom-ID tuples; independent of rows."""
    if len(pair) != 2:
        raise ValueError("A pair must contain two molecule ID collections")
    first, second = (tuple(sorted(group)) for group in pair)
    if (not first or not second or len(set(first)) != len(first)
            or len(set(second)) != len(second) or set(first) & set(second)):
        raise ValueError("Pair molecules need nonempty, distinct, disjoint atom IDs")
    return tuple(sorted((first, second)))


@dataclass(frozen=True)
class PairLifetime:
    pair: tuple
    first_frame: int
    last_frame: int
    first_time_ps: float
    last_time_ps: float
    observed_span_ps: float
    detected_frames: int
    left_censored: bool
    right_censored: bool


def get_pair_lifetime(pairs_by_frame, times_ps, *, max_gap_frames=0, progress=None):
    """Track encounter episodes from per-frame lists of molecule-ID pairs.

    Directly accepts pair lists from get_all_prereactive_pairs, or filtered lists.
    Use one fixed selection criterion throughout. Molecule IDs here are atom-ID
    sets, not per-frame candidate numbers. Pair order/atom order is immaterial.

    Default episodes are continuous detections. max_gap_frames bridges that many
    missing samples (intermittent definition). Span = last detection minus first
    detection: a single-sample event has span zero, not a known zero lifetime.
    Entry/exit occur between samples and are not interpolated. Boundary episodes
    are censored with the same gap allowance at both ends: an episode is
    left-censored when its first detection is within max_gap_frames of the first
    supplied frame (it could be bridged to earlier, unseen detections) and
    right-censored when still open at the final frame. Do not average these
    spans as unbiased physical lifetimes.
    The iterable of pair lists is consumed once; trajectory coordinates are not
    retained. Timestamps may be irregular. Returned frame indices are local to
    the supplied trajectory/window. Missing frames are not absent detections.

    observed_span_ps=times_ps[last_frame]-times_ps[first_frame]. Two
    detections at indices i<j remain in one encounter when j-i-1 <=
    max_gap_frames; detected_frames counts observations, not bridged gaps.
    Optional progress reports completed frame counts and the final sorting stage.
    """
    if (isinstance(max_gap_frames, (bool, np.bool_)) or not isinstance(max_gap_frames, (int, np.integer))
            or max_gap_frames < 0):
        raise ValueError("max_gap_frames must be a nonnegative integer")
    times = np.asarray(times_ps, dtype=float)
    if times.ndim != 1:
        raise ValueError("Timestamps must be one-dimensional")
    _times(times, len(times))
    active, result = {}, []

    def finish(key, state, censored):
        start, end, count = state
        result.append(PairLifetime(key, start, end, float(times[start]), float(times[end]),
                                   float(times[end] - times[start]), count, start <= max_gap_frames, censored))

    sentinel = object()
    for index, (pairs, time) in enumerate(zip_longest(pairs_by_frame, times, fillvalue=sentinel)):
        if pairs is sentinel or time is sentinel:
            raise ValueError("One pair list is required for every timestamp")
        current = {pair_identity(pair) for pair in pairs}
        for key in list(active):
            # On a detected frame the gap excludes this frame; on an absent
            # frame it includes it. Close only when the allowed gap is exceeded.
            gap = index - active[key][1] - (1 if key in current else 0)
            if gap > max_gap_frames:
                finish(key, active.pop(key), False)
        for key in sorted(current):
            if key in active:
                active[key][1] = index
                active[key][2] += 1
            else:
                active[key] = [index, index, 1]
        if progress and ((index+1) % max(1, len(times)//20) == 0 or index+1 == len(times)):
            progress(f'Pair lifetimes: {index+1}/{len(times)} frames processed; {len(result)} closed encounters')
    for key, state in active.items():
        finish(key, state, True)
    if progress:
        progress(f'Pair lifetimes: sorting {len(result)} encounters')
    return sorted(result, key=lambda episode: (episode.first_frame, episode.pair))
