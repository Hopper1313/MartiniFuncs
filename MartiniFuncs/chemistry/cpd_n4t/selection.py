"""CPD + N4T reactive sites and pair geometry (the measurements behind CpdN4tModel).

The shared Diels-Alder geometry (chemistry.diels_alder: forming bonds,
axis/plane/approach angles) plus this pair's ring_dihedral, a signed dihedral
(-180--180 degrees) that tells exo (near 0) from endo (near 180). Encounter
diagnostics, not TS predictions. Thresholds (model.CpdN4tCriteria) are
research choices: none are imposed by default. Distances use the jointly
placed pair, never independent minimum images for the two forming bonds.
"""
from __future__ import annotations

from dataclasses import dataclass

from MartiniFuncs.chemistry.cpd_n4t.model import CPD_SIZE, DIENOPHILE_SIZE
from MartiniFuncs.chemistry.diels_alder import DielsAlderGeometry, cyclic_dienophile_chains
from MartiniFuncs.pairs.assessment import assess_pairs, measure_pair
from MartiniFuncs.pairs.model import get_pair_model, model_sites
from MartiniFuncs.structure import graph as mfg

_SIZE_ERROR = f"Expected an {CPD_SIZE}-atom CPD and a {DIENOPHILE_SIZE}-atom dienophile"


@dataclass(frozen=True)
class PairGeometry(DielsAlderGeometry):
    """The shared Diels-Alder geometry (forming bonds, angles, index mappings) plus ring_dihedral.

    ``forming_bonds_local`` indexes the extracted pair: CPD then dienophile,
    each in place_component order. This matches get_pair_shell_clusters, which
    appends solvent after the pair. ``forming_bonds_frame_local`` indexes the
    input Atoms. Neither local mapping survives subsequent atom reordering.

    ``ring_dihedral`` is the signed dihedral CH2 carbon - CPD termini midpoint -
    C=C midpoint - imide N, in degrees (-180, 180]: near 0 the imide ring points
    the same way as the CH2 bridge (exo), near +/-180 it lies under the diene
    (endo). The sign is the twist direction (mirror images), not exo/endo.
    """
    ring_dihedral: float

    @property
    def ring_dihedral_deviation(self):
        """Degrees from the nearest of 0 (exo) and 180 (endo): min(|phi|, 180-|phi|), in 0--90."""
        magnitude = abs(self.ring_dihedral)
        return min(magnitude, 180. - magnitude)

    @property
    def approach_type(self):
        """'exo' when |ring_dihedral| < 90 degrees, else 'endo'."""
        return "exo" if abs(self.ring_dihedral) < 90 else "endo"


def _cpd_sites(graph, elements, cpd):
    """Identify ordered termini, the four diene carbons and the CH2 bridge carbon from one CPD graph."""
    if len(cpd) != CPD_SIZE:
        raise ValueError(_SIZE_ERROR)
    cg = mfg.induced_subgraph(graph, cpd)
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
    return ends, paths[0], bridge


def _dienophile_sites(graph, elements, mal):
    """Identify reactive C=C, four plane carbons and the imide N from one dienophile graph.

    The imide N is the nitrogen bonded to both carbonyl carbons (atoms 1 and 4
    of the O=C-C=C-C=O path): the ring atom opposite the C=C.
    """
    if len(mal) != DIENOPHILE_SIZE:
        raise ValueError(_SIZE_ERROR)
    mg = mfg.induced_subgraph(graph, mal)
    candidates = cyclic_dienophile_chains(mg, elements)
    sites = {tuple(sorted(p[2:4])) for p in candidates}
    planes = {tuple(sorted(p[1:5])) for p in candidates}
    if len(sites) != 1 or len(planes) != 1:
        raise ValueError("Dienophile reactive C=C or its carbon plane is ambiguous")
    first, last = candidates[0][1], candidates[0][4]
    nitrogens = ({j for j, _ in mg[first] if elements[j] == "N"}
                 & {j for j, _ in mg[last] if elements[j] == "N"})
    if len(nitrogens) != 1:
        raise ValueError("Dienophile imide nitrogen (bonded to both carbonyl carbons) not found")
    return next(iter(sites)), next(iter(planes)), next(iter(nitrogens))


def _sites(atoms, graph, cpd, mal):
    """(termini, diene carbons, CH2 carbon, C=C, dienophile plane carbons, imide N) of one pair, memoized."""
    model = get_pair_model("cpd_n4t")
    elements = dict(zip(mfg.atom_ids(atoms), atoms.get_chemical_symbols()))
    return (*model_sites(model, graph, elements, cpd, 0), *model_sites(model, graph, elements, mal, 1))


def measure_pair_geometry(atoms, pair, shift, *, scale=1.3, graph=None):
    """Measure both possible forming-bond matchings and choose the minimax one.

    The tie-breakers are total distance and persistent atom IDs. ``axis_angle``
    compares the diene termini axis with C=C; ``plane_angle`` compares fitted
    reactive carbon planes; ``approach_angle`` compares site separation to the
    diene-plane normal; ``ring_dihedral`` is the signed CH2 - termini midpoint -
    C=C midpoint - imide N dihedral (exo near 0, endo near 180). These are
    encounter descriptors; they do not establish product stereochemistry.

    For each of the two termini/C=C matchings, d_j=|r_CPD,j-r_CC,j|.
    Choose lexicographically (max(d), sum(d), persistent bond IDs).
    Let u join the CPD termini, v join the C=C carbons, and
    s=mean(r_CC)-mean(r_CPD_termini). Fit normals n_CPD,n_MAL to their
    four reactive carbons by centered SVD (diels_alder.plane_normal). With
    A(a,b)=degrees(arccos(|a dot b|/(|a||b|))), axis_angle=A(u,v),
    plane_angle=A(n_CPD,n_MAL), approach_angle=A(n_CPD,s), and
    ring_dihedral=dihedral(r_CH2, mean(r_CPD_termini), mean(r_CC), r_N) (diels_alder.signed_dihedral).
    Positions are molecularly unwrapped; shift translates CPD by shift @ cell.
    Distances are Angstrom. These site midpoints differ from the whole-CPD
    COM used for the initial distance screen.
    """
    if graph is None:
        graph = mfg.build_bond_graph(atoms, scale=scale)
    return measure_pair(atoms, pair, shift, "cpd_n4t", graph=graph)


def _measure_placed_pair(ci, cx, mi, mx, sites, frame_rows):
    """Measure already placed coordinates using measure_pair_geometry's equations.

    ci/mi are ordered persistent IDs; cx/mx are whole, jointly placed Angstrom
    coordinates. sites contains termini, diene carbons, the CH2 carbon, C=C,
    plane carbons and the imide N. No graph building or periodic unwrapping occurs here.
    """
    return get_pair_model("cpd_n4t").measure(ci, cx, mi, mx, sites[:3], sites[3:], frame_rows)


def assess_prereactive_pairs(atoms, pairs, shifts, *, criteria=None, scale=1.3, graph=None):
    """Return diagnostics/rejection reasons for every candidate in input order.

    Reject when max(d_1,d_2), |d_1-d_2|, a measured acute angle, or the ring
    dihedral's deviation from 0/180 exceeds its non-None CpdN4tCriteria threshold. Equality passes.
    Distances are Angstrom, angles degrees; invalid geometry is always rejected.
    An optional graph is reused and must describe this same frame and bond scale.
    """
    return assess_pairs(atoms, pairs, shifts, "cpd_n4t", criteria=criteria, scale=scale, graph=graph)
