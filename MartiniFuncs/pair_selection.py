"""Optional CPD/dienophile encounter diagnostics, not TS predictions.

All angles are unsigned acute angles (0--90 degrees). Thresholds are research
choices: none are imposed by default. Distances use the jointly placed pair,
never independent minimum images for the two forming bonds.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np

from . import graph as mfg


@dataclass(frozen=True)
class PairCriteria:
    max_forming_distance: float | None = None
    max_distance_imbalance: float | None = None
    max_axis_angle: float | None = None
    max_plane_angle: float | None = None
    max_approach_angle: float | None = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{field.name} must be finite and nonnegative")
            if value is not None and "angle" in field.name and value > 90:
                raise ValueError(f"{field.name} must be at most 90 degrees")


@dataclass(frozen=True)
class PairGeometry:
    """Proposed bonds with explicitly scoped zero-based index mappings.

    ``forming_bonds_local`` indexes the extracted pair: CPD then dienophile,
    each in place_component order. This matches get_pair_shell_clusters, which
    appends solvent after the pair. ``forming_bonds_frame_local`` indexes the
    input Atoms. Neither local mapping survives subsequent atom reordering.
    """
    forming_bonds: tuple[tuple[int, int], tuple[int, int]]
    forming_distances: tuple[float, float]
    axis_angle: float
    plane_angle: float
    approach_angle: float
    forming_bonds_local: tuple[tuple[int, int], tuple[int, int]]
    forming_bonds_frame_local: tuple[tuple[int, int], tuple[int, int]]

    @property
    def distance_imbalance(self):
        """Return |d_1-d_2| in Angstrom for the chosen forming_distances.
        """
        return abs(self.forming_distances[0] - self.forming_distances[1])


@dataclass(frozen=True)
class PairAssessment:
    pair_index: int
    geometry: PairGeometry | None
    reasons: tuple[str, ...]

    @property
    def accepted(self):
        return not self.reasons


def _subgraph(graph, ids):
    return mfg.induced_subgraph(graph, ids)


def _cpd_sites(graph, elements, cpd):
    """Identify ordered termini and four diene carbons from one CPD graph."""
    if len(cpd) != 11:
        raise ValueError("Expected an 11-atom CPD and a 23-atom dienophile")
    cg = _subgraph(graph, cpd)
    if sorted(elements[i] for i in cpd) != sorted("C" * 5 + "H" * 6):
        raise ValueError("CPD component does not have formula C5H6")
    bridges = {chain[1] for chain in mfg.find_element_chains(cg, elements, "HCH")}
    if len(bridges) != 1:
        raise ValueError("Expected exactly one CPD CH2 bridge")
    bridge = next(iter(bridges))
    ends = sorted({j for j, _ in cg[bridge] if elements[j] == "C"})
    if len(ends) != 2:
        raise ValueError("CPD bridge must have two carbon neighbors")
    paths = [p for p in mfg.find_element_chains(cg, elements, "CCCC")
             if {p[0], p[-1]} == set(ends)]
    if len(paths) != 1:
        raise ValueError("Expected one four-carbon diene path")
    return ends, paths[0]


def _dienophile_sites(graph, elements, mal):
    """Identify reactive C=C and four plane carbons from one dienophile graph."""
    if len(mal) != 23:
        raise ValueError("Expected an 11-atom CPD and a 23-atom dienophile")
    mg = _subgraph(graph, mal)
    hcch = {frozenset(p[1:3]) for p in mfg.find_element_chains(mg, elements, "HCCH")}
    candidates = [p for p in mfg.find_element_chains(mg, elements, "OCCCCO")
                  if frozenset(p[2:4]) in hcch]
    sites = {tuple(sorted(p[2:4])) for p in candidates}
    planes = {tuple(sorted(p[1:5])) for p in candidates}
    if len(sites) != 1 or len(planes) != 1:
        raise ValueError("Dienophile reactive C=C or its carbon plane is ambiguous")
    return next(iter(sites)), next(iter(planes))


def _sites(atoms, graph, cpd, mal):
    elements = dict(zip(mfg.atom_ids(atoms), atoms.get_chemical_symbols()))
    return (*_cpd_sites(graph, elements, cpd), *_dienophile_sites(graph, elements, mal))


def _normal(points):
    """Fit a plane to rows of points (Angstrom): X=points-mean(points),
    X=U S Vt; return Vt[-1], the unit normal minimizing |X n|^2.
    Reject collinear sites when the second singular value is below 1e-10.
    """
    _, singular, vectors = np.linalg.svd(points - points.mean(axis=0))
    if singular[1] < 1e-10:
        raise ValueError("Reactive-site plane is degenerate")
    return vectors[-1]


def _angle(a, b):
    """Return degrees(arccos(clip(|a dot b|/(|a||b|),0,1))).
    The absolute dot product makes axes unoriented: the range is 0--90 degrees.
    """
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("Reactive-site direction is degenerate")
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)) / norm, 0, 1))))


def measure_pair_geometry(atoms, pair, shift, *, scale=1.3, graph=None):
    """Measure both possible forming-bond matchings and choose the minimax one.

    The tie-breakers are total distance and persistent atom IDs. ``axis_angle``
    compares the diene termini axis with C=C; ``plane_angle`` compares fitted
    reactive carbon planes; ``approach_angle`` compares site separation to the
    diene-plane normal. These descriptors do not establish stereochemistry.

    For each of the two termini/C=C matchings, d_j=|r_CPD,j-r_CC,j|.
    Choose lexicographically (max(d), sum(d), persistent bond IDs).
    Let u join the CPD termini, v join the C=C carbons, and
    s=mean(r_CC)-mean(r_CPD_termini). Fit normals n_CPD,n_MAL to their
    four reactive carbons by centered SVD (_normal). With
    A(a,b)=degrees(arccos(|a dot b|/(|a||b|))), axis_angle=A(u,v),
    plane_angle=A(n_CPD,n_MAL), approach_angle=A(n_CPD,s).
    Positions are molecularly unwrapped; shift translates CPD by shift @ cell.
    Distances are Angstrom. These site midpoints differ from the whole-CPD
    COM used for the initial distance screen.
    """
    if graph is None:
        graph = mfg.build_bond_graph(atoms, scale=scale)
    cpd, mal = pair
    ends, diene, cc, mal_plane = _sites(atoms, graph, cpd, mal)
    ci, cx, _ = mfg.place_component(atoms, graph, cpd, image_shift=shift, strict=True)
    mi, mx, _ = mfg.place_component(atoms, graph, mal, strict=True)
    frame_rows = {int(atom_id): row for row, atom_id in enumerate(mfg.atom_ids(atoms))}
    return _measure_placed_pair(ci, cx, mi, mx, (ends, diene, cc, mal_plane), frame_rows)


def _measure_placed_pair(ci, cx, mi, mx, sites, frame_rows):
    """Measure already placed coordinates using measure_pair_geometry's equations.

    ci/mi are ordered persistent IDs; cx/mx are whole, jointly placed Angstrom
    coordinates. sites contains termini, diene carbons, C=C, and plane carbons.
    No graph building or periodic unwrapping occurs here.
    """
    ends, diene, cc, mal_plane = sites
    xyz = dict(zip(np.concatenate((ci, mi)), np.vstack((cx, mx))))
    options = []
    for partner in (cc, cc[::-1]):
        bonds = tuple(zip(ends, partner))
        distances = tuple(float(np.linalg.norm(xyz[i] - xyz[j])) for i, j in bonds)
        options.append((max(distances), sum(distances), bonds, distances))
    _, _, bonds, distances = min(options)
    id_to_local = frame_rows
    frame_bonds = tuple((id_to_local[i], id_to_local[j]) for i, j in bonds)
    pair_rows = {int(atom_id): row for row, atom_id in enumerate(np.concatenate((ci, mi)))}
    local_bonds = tuple((pair_rows[i], pair_rows[j]) for i, j in bonds)
    diene_normal = _normal(np.array([xyz[i] for i in diene]))
    mal_normal = _normal(np.array([xyz[i] for i in mal_plane]))
    separation = (xyz[cc[0]] + xyz[cc[1]] - xyz[ends[0]] - xyz[ends[1]]) / 2
    return PairGeometry(
        bonds, distances,
        _angle(xyz[ends[1]] - xyz[ends[0]], xyz[cc[1]] - xyz[cc[0]]),
        _angle(diene_normal, mal_normal), _angle(diene_normal, separation),
        local_bonds, frame_bonds,
    )


def assess_prereactive_pairs(atoms, pairs, shifts, *, criteria=None, scale=1.3, graph=None):
    """Return diagnostics/rejection reasons for every candidate in input order.

    Reject when max(d_1,d_2), |d_1-d_2|, or a measured acute angle
    exceeds its corresponding non-None threshold. Equality passes. Distances
    are Angstrom, angles degrees; invalid geometry is always rejected.
    An optional graph is reused and must describe this same frame and bond scale.
    """
    if len(pairs) != len(shifts):
        raise ValueError("pairs and shifts must have the same length")
    criteria = PairCriteria() if criteria is None else criteria
    if graph is None:
        graph = mfg.build_bond_graph(atoms, scale=scale)
    results = []
    for index, (pair, shift) in enumerate(zip(pairs, shifts)):
        try:
            geometry = measure_pair_geometry(atoms, pair, shift, graph=graph)
        except ValueError as exc:
            results.append(PairAssessment(index, None, (str(exc),)))
            continue
        reasons = _rejection_reasons(geometry, criteria)
        results.append(PairAssessment(index, geometry, reasons))
    return results


def _rejection_reasons(geometry, criteria):
    """Reject measurements strictly above non-None PairCriteria thresholds.

    Compare max(d1,d2), abs(d1-d2), and the three acute angles; equality passes.
    """
    values = {
        "max_forming_distance": max(geometry.forming_distances),
        "max_distance_imbalance": geometry.distance_imbalance,
        "max_axis_angle": geometry.axis_angle,
        "max_plane_angle": geometry.plane_angle,
        "max_approach_angle": geometry.approach_angle,
    }
    reasons = tuple(f"{name}: {value:.6g} > {getattr(criteria, name):.6g}"
                    for name, value in values.items()
                    if getattr(criteria, name) is not None and value > getattr(criteria, name))
    return reasons
