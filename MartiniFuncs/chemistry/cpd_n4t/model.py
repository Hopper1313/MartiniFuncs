"""The CPD + N4T Diels-Alder pair model.

Roles: cyclopentadiene (CPD, C5H6, 11 atoms) is the first member and the
N4T maleimide dienophile (C11H9NO2, 23 atoms) the second; 34-atom adducts
take no part. In exported structures the members are labelled "CPD" and
"MAL" (role_labels; MAL = the maleimide, i.e. the N4T dienophile). Dimerization (CPD+CPD) is ignored: it is not observed
experimentally.

Screen: the whole-CPD COM to the dienophile reactive C=C COM.
Sites: CPD bridge-ended diene termini and its four diene carbons; the
dienophile C=C and its four ring carbons. Metrics (selection.measure_pair_geometry):
max_forming_distance, distance_imbalance (Angstrom), axis_angle,
plane_angle, approach_angle (acute degrees).
"""
from __future__ import annotations

from dataclasses import dataclass

from MartiniFuncs.pairs.model import PairCriteria, PairModel, threshold
from MartiniFuncs.structure import graph as mfg

CPD_SIZE, DIENOPHILE_SIZE, ADDUCT_SIZE = 11, 23, 34


@dataclass(frozen=True)
class CpdN4tCriteria(PairCriteria):
    """Optional CPD + N4T thresholds: distances in Angstrom, unsigned acute angles in degrees.

    Thresholds are research choices; none are imposed by default.
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


class CpdN4tModel(PairModel):
    name = "cpd_n4t"
    roles = ("cpd", "dienophile")
    role_labels = ("CPD", "MAL")
    criteria_class = CpdN4tCriteria
    metric_names = ("max_forming_distance", "distance_imbalance", "axis_angle", "plane_angle", "approach_angle")
    metric_ranges = {"max_forming_distance": (0, 8), "distance_imbalance": (0, 4),
                     "axis_angle": (0, 90), "plane_angle": (0, 90), "approach_angle": (0, 90)}
    species = frozenset({"C5H6", "C11H9NO2"})

    def classify(self, components, elements):
        sizes = {CPD_SIZE: 0, DIENOPHILE_SIZE: 1, ADDUCT_SIZE: None}
        other = [c for c in components if len(c) not in sizes]
        if other:
            raise ValueError(
                f"There are {len(other)} molecules that do not match expected sizes "
                f"({CPD_SIZE}, {DIENOPHILE_SIZE}, {ADDUCT_SIZE} atoms). Check bond scale / chemistry.")
        return [sizes[len(c)] for c in components]

    def screen_atoms(self, graph, elements, component, role):
        if role == 0:
            return None
        from MartiniFuncs.chemistry.cpd_n4t.selection import dienophile_cc_candidates
        cc = {i for chain in dienophile_cc_candidates(mfg.induced_subgraph(graph, component), elements)
              for i in chain[2:4]}
        if len(cc) != 2:
            raise ValueError(f"Could not identify the dienophile reactive C=C for component {component[:5]}...")
        return cc

    def sites(self, graph, elements, component, role):
        from MartiniFuncs.chemistry.cpd_n4t.selection import _cpd_sites, _dienophile_sites
        return (_cpd_sites if role == 0 else _dienophile_sites)(graph, elements, component)

    def measure(self, first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows):
        from MartiniFuncs.chemistry.cpd_n4t.selection import _measure_placed_pair
        return _measure_placed_pair(first_ids, first_xyz, second_ids, second_xyz,
                                    (*first_sites, *second_sites), frame_rows)

    def metrics(self, geometry):
        return {"max_forming_distance": max(geometry.forming_distances),
                "distance_imbalance": geometry.distance_imbalance,
                "axis_angle": geometry.axis_angle, "plane_angle": geometry.plane_angle,
                "approach_angle": geometry.approach_angle}
