"""Anthracene and maleic anhydride reactive sites, found from each molecule's bond graph.

Sites follow the shared Diels-Alder form (chemistry.diels_alder):

    anthracene (diene):           (termini, central_ring)
    maleic anhydride (dienophile): (alkene, ring_carbons, bridging_oxygen)

termini are C9 and C10, the CH carbons whose two carbon neighbours carry no
hydrogen (ring-fusion carbons); central_ring is C9, C10 and those four fusion
carbons, whose fitted plane is the anthracene plane. The anhydride alkene is
the H-C=C-H inside its O=C-C=C-C=O path; ring_carbons are the four ring
carbons (the dienophile plane) and bridging_oxygen the ring O bonded to both
carbonyl carbons. Only bonding is used, never atom order.
"""
from collections import Counter

from MartiniFuncs.chemistry.diels_alder import cyclic_dienophile_chains
from MartiniFuncs.structure import graph as mfg

ANTHRACENE_SIZE, ANHYDRIDE_SIZE, ADDUCT_SIZE = 24, 9, 33


def _check_formula(elements, component, counts, name):
    if Counter(elements[i] for i in component) != Counter(counts):
        raise ValueError(f"{name} component does not have formula "
                         + "".join(f"{e}{n}" for e, n in counts.items()))


def anthracene_sites(graph, elements, component):
    """(termini, central_ring) of one anthracene: C9/C10 and the six central-ring carbons (sorted IDs)."""
    _check_formula(elements, component, {"C": 14, "H": 10}, "Anthracene")
    g = mfg.induced_subgraph(graph, component)
    hydrogens = {i: sum(elements[j] == "H" for j, _ in g[i]) for i in component if elements[i] == "C"}
    carbon_neighbours = {i: [j for j, _ in g[i] if elements[j] == "C"] for i in hydrogens}
    termini = sorted(i for i, n in hydrogens.items()
                     if n == 1 and len(carbon_neighbours[i]) == 2
                     and all(hydrogens[j] == 0 for j in carbon_neighbours[i]))
    if len(termini) != 2:
        raise ValueError("Anthracene C9/C10 carbons not found (expected two CH carbons between fusion carbons)")
    fusion = {j for i in termini for j in carbon_neighbours[i]}
    if len(fusion) != 4:
        raise ValueError("Anthracene central ring not found")
    return tuple(termini), tuple(sorted(set(termini) | fusion))


def anhydride_sites(graph, elements, component):
    """(alkene, ring_carbons, bridging_oxygen) of one maleic anhydride (sorted IDs)."""
    _check_formula(elements, component, {"C": 4, "H": 2, "O": 3}, "Maleic anhydride")
    g = mfg.induced_subgraph(graph, component)
    chains = cyclic_dienophile_chains(g, elements)
    alkenes = {tuple(sorted(p[2:4])) for p in chains}
    rings = {tuple(sorted(p[1:5])) for p in chains}
    if len(alkenes) != 1 or len(rings) != 1:
        raise ValueError("Maleic anhydride C=C or ring carbons are ambiguous")
    first, last = chains[0][1], chains[0][4]
    oxygens = {j for j, _ in g[first] if elements[j] == "O"} & {j for j, _ in g[last] if elements[j] == "O"}
    if len(oxygens) != 1:
        raise ValueError("Maleic anhydride bridging O (bonded to both carbonyl carbons) not found")
    return next(iter(alkenes)), next(iter(rings)), next(iter(oxygens))
