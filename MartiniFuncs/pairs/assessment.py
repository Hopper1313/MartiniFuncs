"""Model-driven pair screening and measurement on one frame (rebuilding the bond graph).

This is the per-frame reference path: molecules, roles and sites are found
from the frame's own bond graph, and molecules are made whole along that
graph's stored bond shifts (structure.graph.unwrap_component). The cached path
for fixed-topology trajectories is pairs.topology.FixedPairTopology, fed by
trajectory.parallel.prepare_frame, which makes molecules whole by minimum-image
bond vectors along the first frame's spanning tree. Both give the same whole
molecules whenever every bond is shorter than half the smallest cell spacing,
and the screen and the model's measurements then use the same equations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MartiniFuncs.pairs.model import SCREEN_METRIC, get_collection, get_pair_model, model_sites
from MartiniFuncs.pairs.screening import pairs_from_centers
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.structure.molecules import get_element_map


@dataclass(frozen=True)
class PairAssessment:
    """One candidate: its measured geometry (None if it could not be measured), the
    selection's rejection reasons (() = selected) and the measurement error, if any."""
    pair_index: int
    geometry: object | None
    reasons: tuple[str, ...]
    error: str | None = None

    @property
    def accepted(self):
        return not self.reasons


def measure_pair(atoms, pair, shift, model=None, *, graph, elements=None, frame_rows=None):
    """Identify both members' sites, place the pair jointly and measure it with the model.

    The first member is translated by shift @ cell; both are made whole along
    graph bonds (strict). Raises ValueError for missing sites or broken molecules.
    """
    model = get_pair_model(model)
    if elements is None:
        elements = dict(zip(mfg.atom_ids(atoms), atoms.get_chemical_symbols()))
    first, second = pair
    first_sites = model_sites(model, graph, elements, first, 0)
    second_sites = model_sites(model, graph, elements, second, 1)
    fi, fx, _ = mfg.place_component(atoms, graph, first, image_shift=shift, strict=True)
    si, sx, _ = mfg.place_component(atoms, graph, second, strict=True)
    if frame_rows is None:
        frame_rows = {int(atom_id): row for row, atom_id in enumerate(mfg.atom_ids(atoms))}
    return model.measure(fi, fx, si, sx, first_sites, second_sites, frame_rows)


def assess_pairs(atoms, pairs, shifts, model=None, *, criteria=None, scale=1.3, graph=None,
                 screen_distances=None):
    """Measurements and selection for every candidate, in input order.

    criteria (the model's PairCriteria; None selects everything) is applied
    exactly as to cached candidates (PairCriteria.rejection_reasons). A pair
    whose geometry cannot be measured keeps the message in error and is
    rejected only if the criteria require geometry. screen_distances (one per
    pair) is needed only for a max_screen_distance threshold. An optional
    graph must describe this frame and bond scale.
    """
    if len(pairs) != len(shifts):
        raise ValueError("pairs and shifts must have the same length")
    if screen_distances is not None and len(screen_distances) != len(pairs):
        raise ValueError("screen_distances needs one value per pair")
    model = get_pair_model(model)
    if graph is None:
        graph = mfg.build_bond_graph(atoms, scale=scale)
    elements = dict(zip(mfg.atom_ids(atoms), atoms.get_chemical_symbols()))
    frame_rows = {int(atom_id): row for row, atom_id in enumerate(mfg.atom_ids(atoms))}
    results = []
    for index, (pair, shift) in enumerate(zip(pairs, shifts)):
        geometry = error = None
        try:
            geometry = measure_pair(atoms, pair, shift, model, graph=graph, elements=elements,
                                    frame_rows=frame_rows)
        except ValueError as exc:
            error = str(exc)
        metrics = model.metrics(geometry) if geometry is not None else {}
        if screen_distances is not None:
            metrics[SCREEN_METRIC] = float(screen_distances[index])
        reasons = (criteria.rejection_reasons(metrics, geometry_ok=geometry is not None, error=error)
                   if criteria is not None else ())
        results.append(PairAssessment(index, geometry, reasons, error))
    return results


def _screen_center(atoms, graph, elements, model, component, role):
    site = model.screen_atoms(graph, elements, component, role)
    return mfg.component_center_of_mass(atoms, graph, component, subset=site)


def screen_frame(atoms, model=None, *, scale=1.3, collection=None, criteria=None, return_candidates=False):
    """One frame's pairs: collect by the screen, then optionally select by criteria.

    collection (the model's PairCollection; None: its default) collects every
    pair whose screen centers are closer than collection.reactive_radius
    (Angstrom, nearest image). Screen points are mass-weighted centers of the
    model's screen atoms (or whole molecules) in each molecule's unwrapped
    frame. Returns pairs [[first IDs, second IDs], ...] and the integer image
    shift of the first member. With criteria, only selected pairs are returned,
    by the same rules as pairs.candidates.select_pairs. return_candidates also
    returns every collected candidate as (pair, shift, distance, geometry or None).
    """
    model = get_pair_model(model)
    reactive_radius = get_collection(model, collection).reactive_radius
    elements = get_element_map(atoms, index_array="global_index")
    graph = mfg.build_bond_graph(atoms, scale=scale, index_array="global_index")
    components = mfg.connected_components(graph, atom_ids=mfg.atom_ids(atoms))
    roles = model.classify(components, elements)
    members = [[c for c, r in zip(components, roles) if r == role] for role in (0, 1)]
    centers = [np.asarray([_screen_center(atoms, graph, elements, model, c, role) for c in members[role]])
               for role in (0, 1)]
    if len(centers[0]) == 0 or len(centers[1]) == 0:
        return ([], [], []) if return_candidates else ([], [])
    pairs, shifts, distances = pairs_from_centers(centers[0], centers[1], members[0], members[1],
                                                  atoms.cell.array, atoms.pbc, reactive_radius,
                                                  return_distances=True)
    candidates = None
    if criteria is not None or return_candidates:
        assessments = assess_pairs(atoms, pairs, shifts, model, graph=graph, criteria=criteria, scale=scale,
                                   screen_distances=distances)
        candidates = [(pair, shift, distance, a.geometry)
                      for pair, shift, distance, a in zip(pairs, shifts, distances, assessments)]
        if criteria is not None:
            keep = [a.pair_index for a in assessments if a.accepted]
            pairs, shifts = [pairs[i] for i in keep], [shifts[i] for i in keep]
    if return_candidates:
        return pairs, shifts, candidates
    return pairs, shifts
