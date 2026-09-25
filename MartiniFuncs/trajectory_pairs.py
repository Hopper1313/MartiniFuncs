"""Cached fixed-topology pair analysis and bounded process batches.

Used by trajectory_analysis; standalone, changing-topology frame extraction stays
in geom_utils. This module never unwraps time series or loads a force calculator.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing

import numpy as np

from .geom_utils import _pairs_from_centers, get_mal_CC
from .pair_selection import (
    PairAssessment, _cpd_sites, _dienophile_sites, _subgraph,
    _measure_placed_pair, _rejection_reasons,
)


@dataclass
class FixedPairTopology:
    """Atom identity, masses and reactive sites cached from the initial graph.

    Coordinates supplied to evaluate must follow the initial Atoms row order and
    be whole molecules rooted at each component's first atom. The caller checks
    IDs, elements, masses and fixed-bond reconstruction on every source frame.
    New bonds are not detected here; this is a nonreactive-trajectory analysis.
    """

    cpd: list
    mal: list
    rows: dict
    masses: np.ndarray
    cpd_rows: list
    mal_rows: list
    screen_rows: list
    cpd_sites: list
    mal_sites: list

    @classmethod
    def from_frame(cls, atoms, graph, components):
        """Identify each molecule's reactive atoms once; do not cache coordinates."""
        ids = atoms.arrays["global_index"]
        rows = {int(i): row for row, i in enumerate(ids)}
        elements = dict(zip(map(int, ids), atoms.get_chemical_symbols()))
        cpd = [list(c) for c in components if len(c) == 11]
        mal = [list(c) for c in components if len(c) == 23]
        if len(cpd) + len(mal) != len(components):
            raise ValueError("Cached pair analysis requires CPD/dienophile components")

        def sites(function, component):
            try:
                return function(graph, elements, component)
            except ValueError as exc:
                return str(exc)  # Reject candidates containing this molecule.

        screen_rows = []
        for component in mal:
            cc = get_mal_CC(_subgraph(graph, component), elements)
            if len(cc) != 2:
                raise ValueError("Could not identify MAL reactive C=C")
            screen_rows.append(np.array([rows[i] for i in cc]))
        return cls(cpd, mal, rows, atoms.get_masses().copy(),
                   [np.array([rows[i] for i in c]) for c in cpd],
                   [np.array([rows[i] for i in c]) for c in mal], screen_rows,
                   [sites(_cpd_sites, c) for c in cpd],
                   [sites(_dienophile_sites, c) for c in mal])

    def evaluate(self, whole, cell, pbc, *, criteria=None, reactive_radius=3.25):
        """Return selected pairs, shifts and candidate assessments for one frame.

        COMs are sum(m_i*r_i)/sum(m_i) from whole (Angstrom). Screen whole-CPD
        COM to C=C COM with the same image search as get_all_prereactive_pairs.
        With criteria, use _measure_placed_pair: distance=|r_i-r_j| and acute
        angle=degrees(arccos(|a dot b|/(|a||b|))). Assessments precede filtering;
        criteria=None skips geometry checks and returns an empty assessment list.
        """
        if not np.isfinite(reactive_radius) or reactive_radius <= 0:
            raise ValueError("reactive_radius must be positive and finite")
        whole = np.asarray(whole, dtype=float)
        if whole.shape != (len(self.masses), 3) or not np.isfinite(whole).all():
            raise ValueError("whole must be finite coordinates in initial atom-row order")
        cpd_centers = [np.average(whole[r], axis=0, weights=self.masses[r]) for r in self.cpd_rows]
        mal_centers = [np.average(whole[r], axis=0, weights=self.masses[r]) for r in self.screen_rows]
        pairs, shifts = _pairs_from_centers(
            np.asarray(cpd_centers), np.asarray(mal_centers), self.cpd, self.mal,
            np.asarray(cell), pbc, reactive_radius)
        if criteria is None:
            return pairs, shifts, []
        cpd_index = {tuple(c): i for i, c in enumerate(self.cpd)}
        mal_index = {tuple(c): i for i, c in enumerate(self.mal)}
        assessments = []
        for index, ((cpd, mal), shift) in enumerate(zip(pairs, shifts)):
            c, m = cpd_index[tuple(cpd)], mal_index[tuple(mal)]
            try:
                cs, ms = self.cpd_sites[c], self.mal_sites[m]
                if isinstance(cs, str) or isinstance(ms, str):
                    raise ValueError(cs if isinstance(cs, str) else ms)
                geometry = _measure_placed_pair(
                    cpd, whole[self.cpd_rows[c]] + np.asarray(shift) @ cell,
                    mal, whole[self.mal_rows[m]], (*cs, *ms), self.rows)
                reasons = _rejection_reasons(geometry, criteria)
            except ValueError as exc:
                geometry, reasons = None, (str(exc),)
            assessments.append(PairAssessment(index, geometry, reasons))
        keep = [a.pair_index for a in assessments if a.accepted]
        return [pairs[i] for i in keep], [shifts[i] for i in keep], assessments


def _initialize_worker(topology, criteria):
    """Install static, picklable metadata once per spawned worker process."""
    global _worker_topology, _worker_criteria
    _worker_topology, _worker_criteria = topology, criteria


def _evaluate_batch(batch):
    """Evaluate ordered (frame index, whole positions, cell, PBC) payloads."""
    result = []
    for index, whole, cell, pbc in batch:
        try:
            pairs, shifts, _ = _worker_topology.evaluate(whole, cell, pbc, criteria=_worker_criteria)
        except Exception as exc:
            raise ValueError(f"Pair analysis failed at source frame {index}: {exc}") from exc
        result.append((pairs, shifts))
    return result


class PairBatchRunner:
    """Ordered pair results with at most 2*workers submitted batches in memory.

    workers=1 runs synchronously without serialization. Multiple workers use
    spawn (also on Windows); worker functions live in this importable module.
    Results contain only pairs/shifts, not frame geometry. Exceptions propagate;
    context exit shuts down workers even if reading or evaluation fails.
    """

    def __init__(self, topology, criteria, *, workers, batch_size):
        self.topology, self.criteria = topology, criteria
        self.workers, self.batch_size = workers, batch_size
        self.batch, self.pending, self.results = [], deque(), []
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker, initargs=(self.topology, self.criteria))
        return self

    def add(self, index, whole, cell, pbc):
        """Submit one retained frame; block when the bounded queue fills."""
        if self.pool is None:
            pairs, shifts, _ = self.topology.evaluate(whole, cell, pbc, criteria=self.criteria)
            self.results.append((pairs, shifts))
        else:
            self.batch.append((index, whole.copy(), cell.copy(), pbc.copy()))
            if len(self.batch) >= self.batch_size:
                self._submit()

    def _submit(self):
        if self.batch:
            self.pending.append(self.pool.submit(_evaluate_batch, self.batch))
            self.batch = []
        if len(self.pending) >= 2*self.workers:
            self.results.extend(self.pending.popleft().result())

    def finish(self):
        """Drain results in submission order, preserving temporal pair identity."""
        if self.pool is not None:
            self._submit()
            while self.pending:
                self.results.extend(self.pending.popleft().result())
        return self.results

    def __exit__(self, exc_type, exc, traceback):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
