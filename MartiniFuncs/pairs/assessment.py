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

from MartiniFuncs.pairs.model import DEFAULT_REACTIVE_RADIUS, get_pair_model, model_sites
from MartiniFuncs.pairs.screening import pairs_from_centers
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.structure.molecules import get_element_map


@dataclass(frozen=True)
class PairAssessment:
    pair_index: int
    geometry: object | None
    reasons: tuple[str, ...]

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


def assess_pairs(atoms, pairs, shifts, model=None, *, criteria=None, scale=1.3, graph=None):
    """Measurements and rejection reasons for every candidate, in input order.

    criteria (the model's PairCriteria; default all None) rejects metrics
    strictly above non-None thresholds; equality passes. Unmeasurable geometry
    is always rejected. An optional graph must describe this frame and bond scale.
    """
    if len(pairs) != len(shifts):
        raise ValueError("pairs and shifts must have the same length")
    model = get_pair_model(model)
    criteria = model.criteria_class() if criteria is None else criteria
    if graph is None:
        graph = mfg.build_bond_graph(atoms, scale=scale)
    elements = dict(zip(mfg.atom_ids(atoms), atoms.get_chemical_symbols()))
    frame_rows = {int(atom_id): row for row, atom_id in enumerate(mfg.atom_ids(atoms))}
    results = []
    for index, (pair, shift) in enumerate(zip(pairs, shifts)):
        try:
            geometry = measure_pair(atoms, pair, shift, model, graph=graph, elements=elements,
                                    frame_rows=frame_rows)
        except ValueError as exc:
            results.append(PairAssessment(index, None, (str(exc),)))
            continue
        results.append(PairAssessment(index, geometry, criteria.rejection_reasons(model.metrics(geometry))))
    return results


def _screen_center(atoms, graph, elements, model, component, role):
    site = model.screen_atoms(graph, elements, component, role)
    return mfg.component_center_of_mass(atoms, graph, component, subset=site)


def screen_frame(atoms, model=None, *, scale=1.3, reactive_radius=DEFAULT_REACTIVE_RADIUS, criteria=None,
                 return_candidates=False):
    """All pairs whose screen centers are closer than reactive_radius (Angstrom), nearest image.

    Screen points are mass-weighted centers of the model's screen atoms (or
    whole molecules) in each molecule's unwrapped frame. Returns pairs
    [[first IDs, second IDs], ...] and the integer image shift of the first
    member. With criteria, only accepted pairs are returned. return_candidates
    also returns every screened candidate as (pair, shift, distance, geometry or None).
    """
    model = get_pair_model(model)
    if not np.isfinite(reactive_radius) or reactive_radius <= 0:
        raise ValueError("reactive_radius must be > 0")
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
        assessments = assess_pairs(atoms, pairs, shifts, model, graph=graph, criteria=criteria, scale=scale)
        candidates = [(pair, shift, distance, a.geometry)
                      for pair, shift, distance, a in zip(pairs, shifts, distances, assessments)]
        if criteria is not None:
            keep = [a.pair_index for a in assessments if a.accepted]
            pairs, shifts = [pairs[i] for i in keep], [shifts[i] for i in keep]
    if return_candidates:
        return pairs, shifts, candidates
    return pairs, shifts
