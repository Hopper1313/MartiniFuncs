"""Cached fixed-topology pair analysis.

Used by trajectory.analysis; standalone, changing-topology frame extraction stays
in chemistry.cpd_n4t.pairs. This module never unwraps time series or loads a force
calculator. Parallel evaluation runs through trajectory.parallel (indexed frame chunks).
"""
from dataclasses import dataclass

import numpy as np

from MartiniFuncs.pairs.screening import _pairs_from_centers
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_mal_CC
from MartiniFuncs.chemistry.cpd_n4t.selection import (PairAssessment, _cpd_sites, _dienophile_sites,
                                                      _subgraph, _measure_placed_pair, _rejection_reasons)


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
    cpd_molecules: list = None   # indices of cpd/mal in the components list (molecule_atom_ids)
    mal_molecules: list = None

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
                   [sites(_dienophile_sites, c) for c in mal],
                   [k for k, c in enumerate(components) if len(c) == 11],
                   [k for k, c in enumerate(components) if len(c) == 23])

    def evaluate(self, whole, cell, pbc, *, criteria=None, reactive_radius=3.25, candidates=False):
        """Return selected pairs, shifts and candidate assessments for one frame.

        COMs are sum(m_i*r_i)/sum(m_i) from whole (Angstrom). Screen whole-CPD
        COM to C=C COM with the same image search as get_all_prereactive_pairs.
        With criteria, use _measure_placed_pair: distance=|r_i-r_j| and acute
        angle=degrees(arccos(|a dot b|/(|a||b|))). Assessments precede filtering;
        criteria=None skips geometry checks and returns an empty assessment list.
        candidates=True also measures every screened pair and returns a fourth
        value: rows (cpd_molecule, mal_molecule, shift, metrics) for
        pairs.candidates.frame_candidates, independent of criteria.
        """
        if not np.isfinite(reactive_radius) or reactive_radius <= 0:
            raise ValueError("reactive_radius must be positive and finite")
        whole = np.asarray(whole, dtype=float)
        if whole.shape != (len(self.masses), 3) or not np.isfinite(whole).all():
            raise ValueError("whole must be finite coordinates in initial atom-row order")
        cpd_centers = [np.average(whole[r], axis=0, weights=self.masses[r]) for r in self.cpd_rows]
        mal_centers = [np.average(whole[r], axis=0, weights=self.masses[r]) for r in self.screen_rows]
        pairs, shifts, distances = _pairs_from_centers(
            np.asarray(cpd_centers), np.asarray(mal_centers), self.cpd, self.mal,
            np.asarray(cell), pbc, reactive_radius, return_distances=True)
        if criteria is None and not candidates:
            return pairs, shifts, []
        from MartiniFuncs.pairs.candidates import metric_row
        from MartiniFuncs.chemistry.cpd_n4t.selection import PairCriteria
        measure = criteria if criteria is not None else PairCriteria()
        cpd_index = {tuple(c): i for i, c in enumerate(self.cpd)}
        mal_index = {tuple(c): i for i, c in enumerate(self.mal)}
        assessments, rows = [], []
        for index, ((cpd, mal), shift) in enumerate(zip(pairs, shifts)):
            c, m = cpd_index[tuple(cpd)], mal_index[tuple(mal)]
            try:
                cs, ms = self.cpd_sites[c], self.mal_sites[m]
                if isinstance(cs, str) or isinstance(ms, str):
                    raise ValueError(cs if isinstance(cs, str) else ms)
                geometry = _measure_placed_pair(
                    cpd, whole[self.cpd_rows[c]] + np.asarray(shift) @ cell,
                    mal, whole[self.mal_rows[m]], (*cs, *ms), self.rows)
                reasons = _rejection_reasons(geometry, measure)
            except ValueError as exc:
                geometry, reasons = None, (str(exc),)
            assessments.append(PairAssessment(index, geometry, reasons))
            rows.append((self.cpd_molecules[c], self.mal_molecules[m], shift, metric_row(distances[index], geometry)))
        if criteria is None:
            kept_pairs, kept_shifts = pairs, shifts
        else:
            keep = [a.pair_index for a in assessments if a.accepted]
            kept_pairs, kept_shifts = [pairs[i] for i in keep], [shifts[i] for i in keep]
        if candidates:
            return kept_pairs, kept_shifts, assessments, rows
        return kept_pairs, kept_shifts, assessments
