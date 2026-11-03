"""Cached fixed-topology pair analysis for any pair model.

Used by trajectory.analysis: roles, screen atoms and reactive sites are found
once from the first frame's bond graph, then every frame only computes centers
and measures screened pairs. The per-frame reference path (rebuilding the bond
graph) is pairs.assessment.screen_frame. This module never unwraps time series
or loads a force calculator. Parallel evaluation runs through trajectory.parallel
(indexed frame chunks).
"""
from dataclasses import dataclass

import numpy as np

from MartiniFuncs.pairs.assessment import PairAssessment
from MartiniFuncs.pairs.model import SCREEN_METRIC, get_collection, get_pair_model
from MartiniFuncs.pairs.screening import pairs_from_centers


@dataclass
class FixedPairTopology:
    """Atom identity, masses, roles and reactive sites cached from the initial graph.

    Coordinates supplied to evaluate must follow the initial Atoms row order and
    be whole molecules rooted at each component's first atom. The caller checks
    IDs, elements, masses and fixed-bond reconstruction on every source frame.
    New bonds are not detected here; this is a nonreactive-trajectory analysis.
    first/second hold the components of the model's two roles.
    """

    model: object
    first: list
    second: list
    rows: dict
    masses: np.ndarray
    first_rows: list
    second_rows: list
    screen_rows: tuple        # per role: row arrays of the screen atoms
    first_sites: list         # sites, or the error message for molecules without them
    second_sites: list
    first_molecules: list     # indices of first/second in the components list (molecule_atom_ids)
    second_molecules: list

    @classmethod
    def from_frame(cls, atoms, graph, components, model):
        """Identify each molecule's role, screen atoms and sites once; do not cache coordinates."""
        model = get_pair_model(model)
        ids = atoms.arrays["global_index"]
        rows = {int(i): row for row, i in enumerate(ids)}
        elements = dict(zip(map(int, ids), atoms.get_chemical_symbols()))
        roles = model.classify([list(c) for c in components], elements)
        if any(role is None for role in roles):
            raise ValueError(f"Cached pair analysis requires every molecule to be a {model.name} pair member")
        members = [[list(c) for c, r in zip(components, roles) if r == role] for role in (0, 1)]
        indices = [[k for k, r in enumerate(roles) if r == role] for role in (0, 1)]

        def sites(component, role):
            try:
                return model.sites(graph, elements, component, role)
            except ValueError as exc:
                return str(exc)  # Reject candidates containing this molecule.

        screen_rows = tuple(
            [np.array([rows[i] for i in (c if site is None else site)])
             for c, site in ((c, model.screen_atoms(graph, elements, c, role)) for c in members[role])]
            for role in (0, 1))
        return cls(model, members[0], members[1], rows, atoms.get_masses().copy(),
                   [np.array([rows[i] for i in c]) for c in members[0]],
                   [np.array([rows[i] for i in c]) for c in members[1]], screen_rows,
                   [sites(c, 0) for c in members[0]], [sites(c, 1) for c in members[1]],
                   indices[0], indices[1])

    def evaluate(self, whole, cell, pbc, *, collection=None, criteria=None, candidates=False):
        """Return one frame's pairs, shifts and candidate assessments (collect, then select).

        Screen centers are sum(m_i*r_i)/sum(m_i) over each molecule's screen atoms
        in whole (Angstrom), with the same image search as screen_frame; the
        collection (None: the model's default) sets the radius. With criteria,
        the model measures each collected pair and the criteria select from them
        exactly as pairs.candidates.select_pairs; assessments precede filtering.
        criteria=None returns every collected pair and, unless candidates, an
        empty assessment list. candidates=True also measures every collected
        pair and returns a fourth value: rows (first_molecule, second_molecule,
        shift, metrics) for pairs.candidates.frame_candidates.
        """
        reactive_radius = get_collection(self.model, collection).reactive_radius
        whole = np.asarray(whole, dtype=float)
        if whole.shape != (len(self.masses), 3) or not np.isfinite(whole).all():
            raise ValueError("whole must be finite coordinates in initial atom-row order")
        centers = [np.asarray([np.average(whole[r], axis=0, weights=self.masses[r]) for r in rows])
                   for rows in self.screen_rows]
        pairs, shifts, distances = pairs_from_centers(
            centers[0], centers[1], self.first, self.second,
            np.asarray(cell), pbc, reactive_radius, return_distances=True)
        if criteria is None and not candidates:
            return pairs, shifts, []
        model = self.model
        first_index = {tuple(c): i for i, c in enumerate(self.first)}
        second_index = {tuple(c): i for i, c in enumerate(self.second)}
        assessments, rows = [], []
        for index, ((first, second), shift) in enumerate(zip(pairs, shifts)):
            a, b = first_index[tuple(first)], second_index[tuple(second)]
            geometry = error = None
            try:
                fs, ss = self.first_sites[a], self.second_sites[b]
                if isinstance(fs, str) or isinstance(ss, str):
                    raise ValueError(fs if isinstance(fs, str) else ss)
                geometry = model.measure(first, whole[self.first_rows[a]] + np.asarray(shift) @ cell,
                                         second, whole[self.second_rows[b]], fs, ss, self.rows)
            except ValueError as exc:
                error = str(exc)
            metrics = dict(model.metrics(geometry)) if geometry is not None else {}
            metrics[SCREEN_METRIC] = float(distances[index])
            reasons = (criteria.rejection_reasons(metrics, geometry_ok=geometry is not None, error=error)
                       if criteria is not None else ())
            assessments.append(PairAssessment(index, geometry, reasons, error))
            rows.append((self.first_molecules[a], self.second_molecules[b], shift,
                         model.metric_row(distances[index], geometry)))
        if criteria is None:
            kept_pairs, kept_shifts = pairs, shifts
        else:
            keep = [a.pair_index for a in assessments if a.accepted]
            kept_pairs, kept_shifts = [pairs[i] for i in keep], [shifts[i] for i in keep]
        if candidates:
            return kept_pairs, kept_shifts, assessments, rows
        return kept_pairs, kept_shifts, assessments
