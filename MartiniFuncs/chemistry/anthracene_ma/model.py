"""The anthracene + maleic anhydride Diels-Alder pair model.

Roles: anthracene (C14H10, 24 atoms) is the diene, the first member, and
maleic anhydride (C4H2O3, 9 atoms) the dienophile, the second; 33-atom adducts
take no part. In exported structures the members are labelled "ANT" and "MA"
(role_labels, as build.ROLE_LABELS).

Collection (AnthraceneMaCollection): every pair whose anthracene central-ring
centroid (C9, C10 and the four fusion carbons) to anhydride C=C midpoint
distance (minimum image) is below reactive_radius, default 4.0 Angstrom, a
choice for this pair: the UMA reactant complex is at 3.17, the TS at 2.33 and
the adduct at 1.90 (build.structure_metrics' screen_distance, the same
points). The central ring is used, not the whole-molecule COM, which a bent
anthracene (TS) moves about 0.2 Angstrom away from the dienophile.
Selection (AnthraceneMaCriteria): optional thresholds on that screen distance
and on the metrics below.
Sites (sites.py): anthracene C9/C10 termini and the six central-ring carbons;
the anhydride C=C, its four ring carbons and the bridging O.
Metrics: the shared Diels-Alder metrics only (chemistry.diels_alder:
max_forming_distance, distance_imbalance in Angstrom; axis_angle, plane_angle,
approach_angle in acute degrees). The two faces of the central ring and the
two benzo rings the anhydride ring can point toward are equivalent by
symmetry, so there is no endo/exo metric.
"""
from __future__ import annotations

from dataclasses import dataclass

from MartiniFuncs.chemistry.anthracene_ma.sites import (ADDUCT_SIZE, ANHYDRIDE_SIZE, ANTHRACENE_SIZE,
                                                       anhydride_sites, anthracene_sites)
from MartiniFuncs.chemistry.diels_alder import DielsAlderCriteria, DielsAlderModel
from MartiniFuncs.pairs.model import PairCollection


@dataclass(frozen=True)
class AnthraceneMaCollection(PairCollection):
    """Stage 1 for anthracene + MA: collect every pair with central-ring centroid to C=C midpoint < reactive_radius."""
    reactive_radius: float = 4.0   # Angstrom


@dataclass(frozen=True)
class AnthraceneMaCriteria(DielsAlderCriteria):
    """Stage 2 for anthracene + MA: the shared Diels-Alder thresholds (none imposed by default)."""


class AnthraceneMaModel(DielsAlderModel):
    name = "anthracene_ma"
    roles = ("anthracene", "anhydride")
    role_labels = ("ANT", "MA")
    collection_class = AnthraceneMaCollection
    criteria_class = AnthraceneMaCriteria
    species = frozenset({"C14H10", "C4H2O3"})

    def classify(self, components, elements):
        sizes = {ANTHRACENE_SIZE: 0, ANHYDRIDE_SIZE: 1, ADDUCT_SIZE: None}
        other = [c for c in components if len(c) not in sizes]
        if other:
            raise ValueError(
                f"There are {len(other)} molecules that do not match expected sizes "
                f"({ANTHRACENE_SIZE}, {ANHYDRIDE_SIZE}, {ADDUCT_SIZE} atoms). Check bond scale / chemistry.")
        return [sizes[len(c)] for c in components]

    def screen_atoms(self, graph, elements, component, role):
        return set(self.sites(graph, elements, component, role)[1 if role == 0 else 0])

    def sites(self, graph, elements, component, role):
        return (anthracene_sites if role == 0 else anhydride_sites)(graph, elements, component)
