"""Diels-Alder pair geometry shared by every diene + dienophile pair model.

A Diels-Alder pair model (a DielsAlderModel subclass) supplies the chemistry:
which molecules are the diene (role 0) and the dienophile (role 1), and their
reactive sites. Each molecule's sites start with two fixed entries; anything
after them is model-specific (e.g. an orientation reference atom):

    diene (role 0):       (termini, diene_plane, ...)
    dienophile (role 1):  (alkene, dienophile_plane, ...)

termini are the two diene carbons that bond, alkene the two dienophile C=C
carbons; diene_plane / dienophile_plane are the carbons whose fitted planes
are compared. measure_diels_alder turns a jointly placed pair into a
DielsAlderGeometry with the shared metrics:

    max_forming_distance, distance_imbalance (Angstrom),
    axis_angle, plane_angle, approach_angle (unsigned acute degrees).

Forming bonds: of the two ways to match termini to alkene carbons, the one
minimizing (max distance, sum of distances, persistent bond IDs), so a pair
is measured the same whichever way the dienophile is flipped. Models add
their own metrics on top (e.g. cpd_n4t's ring dihedral). These are encounter
descriptors, not transition-state predictions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from MartiniFuncs.pairs.model import PairCriteria, PairModel, threshold
from MartiniFuncs.structure import graph as mfg

SHARED_METRICS = ("max_forming_distance", "distance_imbalance", "axis_angle", "plane_angle", "approach_angle")
SHARED_METRIC_RANGES = {"max_forming_distance": (0, 8), "distance_imbalance": (0, 4),
                        "axis_angle": (0, 90), "plane_angle": (0, 90), "approach_angle": (0, 90)}


@dataclass(frozen=True)
class DielsAlderGeometry:
    """Forming bonds and shared angles of one measured pair, with explicitly scoped index mappings.

    ``forming_bonds`` holds (diene terminus, alkene carbon) persistent IDs;
    ``forming_bonds_local`` indexes the extracted pair (diene then dienophile,
    each in placement order) and ``forming_bonds_frame_local`` the source
    frame's rows. Neither local mapping survives later atom reordering.
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
        """|d_1 - d_2| in Angstrom for the chosen forming_distances."""
        return abs(self.forming_distances[0] - self.forming_distances[1])


@dataclass(frozen=True)
class DielsAlderCriteria(PairCriteria):
    """Stage-2 thresholds on the shared Diels-Alder metrics (base class of a model's criteria).

    Distances in Angstrom, unsigned acute angles in degrees; max_screen_distance
    and require_geometry come from PairCriteria. Thresholds are research
    choices: none are imposed by default.
    """
    max_forming_distance: float | None = threshold(
        "max_forming_distance", help="longer of the two forming C-C distances")
    max_distance_imbalance: float | None = threshold(
        "distance_imbalance", help="|d1 - d2| of the forming distances")
    max_axis_angle: float | None = threshold(
        "axis_angle", upper=90, unit="degrees", help="diene termini axis vs dienophile C=C")
    max_plane_angle: float | None = threshold(
        "plane_angle", upper=90, unit="degrees", help="diene plane vs dienophile ring-carbon plane")
    max_approach_angle: float | None = threshold(
        "approach_angle", upper=90, unit="degrees", help="diene-plane normal vs site separation")


class DielsAlderModel(PairModel):
    """A pair model whose geometry is measure_diels_alder; subclasses supply the chemistry.

    Subclasses define name, roles, role_labels, collection_class,
    criteria_class (a DielsAlderCriteria subclass), species, classify, sites
    (in the two-entry form above) and, optionally, screen_atoms. To add
    metrics, extend metric_names/metric_ranges and override measure/metrics,
    starting from this class's results.
    """
    metric_names = SHARED_METRICS
    metric_ranges = SHARED_METRIC_RANGES
    criteria_class = DielsAlderCriteria

    def measure(self, first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows):
        return measure_diels_alder(first_ids, first_xyz, second_ids, second_xyz,
                                   first_sites[0], first_sites[1], second_sites[0], second_sites[1], frame_rows)

    def metrics(self, geometry):
        return {"max_forming_distance": max(geometry.forming_distances),
                "distance_imbalance": geometry.distance_imbalance,
                "axis_angle": geometry.axis_angle, "plane_angle": geometry.plane_angle,
                "approach_angle": geometry.approach_angle}


def placed_positions(first_ids, first_xyz, second_ids, second_xyz):
    """{persistent ID: Angstrom position} of a jointly placed pair."""
    return dict(zip(np.concatenate((first_ids, second_ids)), np.vstack((first_xyz, second_xyz))))


def measure_diels_alder(first_ids, first_xyz, second_ids, second_xyz, termini, diene_plane, alkene,
                        dienophile_plane, frame_rows):
    """DielsAlderGeometry of already placed, whole coordinates; no unwrapping happens here.

    For each matching of termini to alkene carbons, d_j=|r_terminus,j - r_alkene,j|;
    choose lexicographically (max(d), sum(d), persistent bond IDs). Let u join
    the termini, v the alkene carbons and s=mean(r_alkene)-mean(r_termini);
    n_diene, n_dienophile are the fitted plane normals (plane_normal). With
    A=acute_angle: axis_angle=A(u,v), plane_angle=A(n_diene,n_dienophile),
    approach_angle=A(n_diene,s). frame_rows maps persistent ID to source row.
    """
    xyz = placed_positions(first_ids, first_xyz, second_ids, second_xyz)
    options = []
    for partner in (alkene, alkene[::-1]):
        bonds = tuple(zip(termini, partner))
        distances = tuple(float(np.linalg.norm(xyz[i] - xyz[j])) for i, j in bonds)
        options.append((max(distances), sum(distances), bonds, distances))
    _, _, bonds, distances = min(options)
    frame_bonds = tuple((frame_rows[i], frame_rows[j]) for i, j in bonds)
    pair_rows = {int(atom_id): row for row, atom_id in enumerate(np.concatenate((first_ids, second_ids)))}
    local_bonds = tuple((pair_rows[i], pair_rows[j]) for i, j in bonds)
    diene_normal = plane_normal(np.array([xyz[i] for i in diene_plane]))
    dienophile_normal = plane_normal(np.array([xyz[i] for i in dienophile_plane]))
    separation = (xyz[alkene[0]] + xyz[alkene[1]]) / 2 - (xyz[termini[0]] + xyz[termini[1]]) / 2
    return DielsAlderGeometry(
        bonds, distances,
        acute_angle(xyz[termini[1]] - xyz[termini[0]], xyz[alkene[1]] - xyz[alkene[0]]),
        acute_angle(diene_normal, dienophile_normal), acute_angle(diene_normal, separation),
        local_bonds, frame_bonds,
    )


def cyclic_dienophile_chains(graph, elements):
    """O=C-C=C-C=O element paths whose middle C-C pair is the middle of an H-C=C-H path.

    The reactive alkene of a cyclic 1,4-dicarbonyl dienophile (maleimides,
    maleic anhydride). graph should already be restricted to one molecule.
    Atoms 2 and 3 of each returned path are the C=C and atoms 1..4 its four
    ring carbons.
    """
    hcch = {frozenset(p[1:3]) for p in mfg.find_element_chains(graph, elements, "HCCH")}
    return [p for p in mfg.find_element_chains(graph, elements, "OCCCCO") if frozenset(p[2:4]) in hcch]


def plane_normal(points):
    """Fit a plane to rows of points (Angstrom): X=points-mean(points),
    X=U S Vt; return Vt[-1], the unit normal minimizing |X n|^2.
    Reject collinear sites when the second singular value is below 1e-10.
    """
    _, singular, vectors = np.linalg.svd(points - points.mean(axis=0))
    if singular[1] < 1e-10:
        raise ValueError("Reactive-site plane is degenerate")
    return vectors[-1]


def acute_angle(a, b):
    """Return degrees(arccos(clip(|a dot b|/(|a||b|),0,1))).
    The absolute dot product makes axes unoriented: the range is 0--90 degrees.
    """
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("Reactive-site direction is degenerate")
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(a, b)) / norm, 0, 1))))


def signed_dihedral(p1, p2, p3, p4):
    """Signed dihedral p1-p2-p3-p4 in degrees, (-180, 180], IUPAC sign convention.

    With b1=p2-p1, b2=p3-p2, b3=p4-p3: phi=atan2(|b2| b1.(b2 x b3), (b1 x b2).(b2 x b3)).
    Positive when, viewed along b2 (from p2 towards p3), the near bond p2-p1
    turns clockwise onto the far bond p3-p4. Undefined (ValueError) when
    p1-p2 or p3-p4 is collinear with the central p2-p3 axis.
    """
    b1, b2, b3 = (np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
                  for p, q in ((p1, p2), (p2, p3), (p3, p4)))
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    length = np.linalg.norm(b2)
    if (length < 1e-8 or np.linalg.norm(n1) < 1e-6 * np.linalg.norm(b1) * length
            or np.linalg.norm(n2) < 1e-6 * np.linalg.norm(b3) * length):
        raise ValueError("Dihedral is undefined: a vector is collinear with the central axis")
    return float(np.degrees(np.arctan2(length * np.dot(b1, n2), np.dot(n1, n2))))
