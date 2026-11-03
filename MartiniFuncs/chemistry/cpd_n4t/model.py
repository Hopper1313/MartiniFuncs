"""The CPD + N4T Diels-Alder pair model.

Roles: cyclopentadiene (CPD, C5H6, 11 atoms) is the first member and the
N4T maleimide dienophile (C11H9NO2, 23 atoms) the second; 34-atom adducts
take no part. In exported structures the members are labelled "CPD" and
"MAL" (role_labels; MAL = the maleimide, i.e. the N4T dienophile).
Dimerization (CPD+CPD) is ignored: it is not observed experimentally.

Collection (CpdN4tCollection): every pair whose whole-CPD COM to dienophile
reactive C=C COM distance (minimum image) is below reactive_radius, default
3.25 Angstrom, a choice for this Diels-Alder pair. Selection (CpdN4tCriteria):
optional thresholds on that screen distance and on the metrics below.
Sites: CPD bridge-ended diene termini, its four diene carbons and the CH2
bridge carbon; the dienophile C=C, its four ring carbons and the imide N.
Metrics: the shared Diels-Alder metrics (chemistry.diels_alder:
max_forming_distance, distance_imbalance in Angstrom; axis_angle, plane_angle,
approach_angle in acute degrees), plus ring_dihedral (signed CH2 - termini
midpoint - C=C midpoint - N dihedral: exo near 0, endo near 180) and
ring_dihedral_deviation (degrees from the nearest of 0 and 180).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from MartiniFuncs.chemistry.diels_alder import (SHARED_METRIC_RANGES, SHARED_METRICS, DielsAlderCriteria,
                                                DielsAlderModel, cyclic_dienophile_chains, placed_positions,
                                                signed_dihedral)
from MartiniFuncs.pairs.model import PairCollection, threshold
from MartiniFuncs.structure import graph as mfg

CPD_SIZE, DIENOPHILE_SIZE, ADDUCT_SIZE = 11, 23, 34


@dataclass(frozen=True)
class CpdN4tCollection(PairCollection):
    """Stage 1 for CPD + N4T: collect and cache every pair with CPD COM to C=C COM < reactive_radius."""
    reactive_radius: float = 3.25   # Angstrom


@dataclass(frozen=True)
class CpdN4tCriteria(DielsAlderCriteria):
    """Stage 2 for CPD + N4T: the shared Diels-Alder thresholds plus the ring-dihedral deviation.

    Distances in Angstrom, unsigned acute angles and the dihedral deviation in
    degrees; inherited max_screen_distance and require_geometry come from
    PairCriteria. Thresholds are research choices; none are imposed by default.
    """
    max_ring_dihedral_deviation: float | None = threshold(
        "ring_dihedral_deviation", upper=90, unit="degrees",
        help="CH2/termini/C=C/N dihedral within this of 0 (exo) or of 180 (endo)")


class CpdN4tModel(DielsAlderModel):
    name = "cpd_n4t"
    roles = ("cpd", "dienophile")
    role_labels = ("CPD", "MAL")
    collection_class = CpdN4tCollection
    criteria_class = CpdN4tCriteria
    metric_names = (*SHARED_METRICS, "ring_dihedral", "ring_dihedral_deviation")
    metric_ranges = {**SHARED_METRIC_RANGES, "ring_dihedral": (-180, 180), "ring_dihedral_deviation": (0, 90)}
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
        cc = {i for chain in cyclic_dienophile_chains(mfg.induced_subgraph(graph, component), elements)
              for i in chain[2:4]}
        if len(cc) != 2:
            raise ValueError(f"Could not identify the dienophile reactive C=C for component {component[:5]}...")
        return cc

    def sites(self, graph, elements, component, role):
        from MartiniFuncs.chemistry.cpd_n4t.selection import _cpd_sites, _dienophile_sites
        return (_cpd_sites if role == 0 else _dienophile_sites)(graph, elements, component)

    def measure(self, first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows):
        """The shared Diels-Alder geometry plus the CH2 - termini - C=C - imide N ring dihedral."""
        from MartiniFuncs.chemistry.cpd_n4t.selection import PairGeometry
        shared = super().measure(first_ids, first_xyz, second_ids, second_xyz, first_sites, second_sites, frame_rows)
        (ends, _, bridge), (cc, _, nitrogen) = first_sites, second_sites
        xyz = placed_positions(first_ids, first_xyz, second_ids, second_xyz)
        dihedral = signed_dihedral(xyz[bridge], (xyz[ends[0]] + xyz[ends[1]]) / 2,
                                   (xyz[cc[0]] + xyz[cc[1]]) / 2, xyz[nitrogen])
        return PairGeometry(**{f.name: getattr(shared, f.name) for f in fields(shared)}, ring_dihedral=dihedral)

    def metrics(self, geometry):
        return {**super().metrics(geometry), "ring_dihedral": geometry.ring_dihedral,
                "ring_dihedral_deviation": geometry.ring_dihedral_deviation}
