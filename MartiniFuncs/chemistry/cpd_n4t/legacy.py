"""Legacy helpers kept for compatibility; nothing else in the package imports them.

Superseded by MartiniFuncs.pairs and the rest of chemistry.cpd_n4t. They are
still exercised by the test suite, but new features are not added here.
"""
from __future__ import annotations
import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.chemistry.cpd_n4t.model import ADDUCT_SIZE, CPD_SIZE, DIENOPHILE_SIZE
from MartiniFuncs.chemistry.cpd_n4t.pairs import get_mal_CC
from MartiniFuncs.structure.molecules import _subset, get_element_map


def get_diels_alder_reaction_indices_from_pair(atoms: Atoms, scale: float = 1.3):
    """Identify the reactive atoms in an isolated CPD + dienophile pair."""
    graph_all = mfg.build_bond_graph(atoms, scale=scale, index_array="global_index")
    molecules = mfg.connected_components(
        graph_all,
        atom_ids=mfg.atom_ids(atoms),
    )

    if len(molecules) != 2:
        raise ValueError(
            f"Expected exactly two molecules, got {len(molecules)} with sizes "
            f"{[len(m) for m in molecules]}."
        )

    if len(molecules[0]) == CPD_SIZE and len(molecules[1]) == DIENOPHILE_SIZE:
        cpd_index, n4t_index = 0, 1
    elif len(molecules[1]) == CPD_SIZE and len(molecules[0]) == DIENOPHILE_SIZE:
        cpd_index, n4t_index = 1, 0
    else:
        raise ValueError(
            f"Unexpected molecule sizes, got {[len(m) for m in molecules]}, "
            f"but expected {CPD_SIZE} and {DIENOPHILE_SIZE}."
        )

    n4t_atoms = _subset(atoms, molecules[n4t_index])
    graph_n4t = mfg.build_bond_graph(n4t_atoms, scale=scale, index_array="global_index")
    elements_n4t = get_element_map(n4t_atoms, index_array="global_index")
    occcco = mfg.find_element_chains(graph_n4t, elements_n4t, "OCCCCO")[0]
    n4t_cc = [occcco[2], occcco[3]]

    cpd_atoms = _subset(atoms, molecules[cpd_index])
    graph_cpd = mfg.build_bond_graph(cpd_atoms, scale=scale, index_array="global_index")
    elements_cpd = get_element_map(cpd_atoms, index_array="global_index")
    hch = mfg.find_element_chains(graph_cpd, elements_cpd, "HCH")[0]

    # H-C1-H identifies the bridge carbon. Its one-bond neighborhood contains
    # the two ring carbons that define the diene termini.
    _, c_ch2_c_graph = mfg.extract_neighborhood(graph_cpd, [hch[1]], n=1)
    ccc = mfg.find_element_chains(c_ch2_c_graph, elements_cpd, "CCC")[0]
    cpd_cxxc = [ccc[0], ccc[2]]

    cccc_candidates = mfg.find_element_chains(graph_cpd, elements_cpd, "CCCC")
    cccc = [
        chain
        for chain in cccc_candidates
        if (chain[0] in cpd_cxxc) and (chain[-1] in cpd_cxxc)
    ][0]
    ring = mfg.find_rings(graph_cpd, max_size=5)[0]

    return {
        "CPD": {
            "reactive carbons": cpd_cxxc,
            "c=cc=c": cccc,
            "ring": ring,
        },
        "dienophile": {
            "reactive carbons": n4t_cc,
            "occcco": occcco,
        },
    }


def pair_data_to_xyz(data, filename):
    """Write pair/cluster data with pair identity stored in the XYZ comment.

    Local indices are 0-based row indices in the extracted XYZ structure.
    Global IDs are persistent atom IDs from ``global_index``. Velocities and
    forces are optional; the comment line's ``columns=`` lists what is written.
    Values use 17 significant digits. Units are those of the record (see
    pair_data_to_extxyz); prefer pair_data_to_extxyz, which records them.
    """

    def _index_list(values):
        return "[" + ",".join(str(int(x)) for x in values) + "]"

    def _number(value):
        # Same 17-significant-digit format as pair_data_to_extxyz: %12.8f
        # truncated small force components to zero.
        return format(float(value), ".17g")

    with open(filename, "w") as f:
        for i in range(len(data)):
            positions = np.asarray(data[i]["positions"])
            symbols = data[i]["symbols"]
            vectors = [("x,y,z", positions)]
            if "velocities" in data[i]:
                vectors.append(("vx,vy,vz", np.asarray(data[i]["velocities"])))
            if "forces" in data[i]:
                vectors.append(("fx,fy,fz", np.asarray(data[i]["forces"])))
            for label, values in vectors:
                if values.shape != positions.shape:
                    raise ValueError(f"Cluster {i}: {label} must have shape {positions.shape}")

            cpd_local = data[i].get("first_local_indices", [])
            mal_local = data[i].get("second_local_indices", [])
            cpd_global = data[i].get("first_global_ids", [])
            mal_global = data[i].get("second_global_ids", [])
            cluster_local = data[i].get(
                "cluster_local_indices", np.arange(len(positions), dtype=int)
            )
            cluster_global = data[i].get(
                "cluster_global_ids", data[i].get("atom_ids", [])
            )

            f.write(f"{len(positions)}\n")
            f.write(
                f"Cluster {i + 1}/{len(data)} "
                f"CPD_local={_index_list(cpd_local)} "
                f"MAL_local={_index_list(mal_local)} "
                f"CPD_global={_index_list(cpd_global)} "
                f"MAL_global={_index_list(mal_global)} "
                f"cluster_local={_index_list(cluster_local)} "
                f"cluster_global={_index_list(cluster_global)} "
                f"columns={','.join(label for label, _ in vectors)}\n"
            )

            for row, sym in enumerate(symbols):
                f.write(" ".join([str(sym)] + [_number(v) for _, values in vectors for v in values[row]]) + "\n")


def align_atoms(mobile, reference, indices=None):
    """Align finite clusters by row correspondence; rotate stored vectors too.

    Returns the historical ``(aligned, rmsd)`` tuple. For persistent-ID matching
    and access to the rotation/translation, use ``structure.alignment.align_geometry``.

    Equal-weight Kabsch fit: minimize sum_i |r_mobile,i @ R+t-r_reference,i|^2
    with det(R)=+1. Return the aligned copy and sqrt(mean squared distance);
    matching is by row, as delegated to structure.alignment.align_geometry(id_array=None).
    """
    from MartiniFuncs.structure.alignment import align_geometry

    aligned, rmsd, _ = align_geometry(mobile, reference, indices, id_array=None)
    return aligned, rmsd


def get_pairs_distance_distribution(atoms: Atoms, scale: float = 1.3):
    """Return MIC distances between whole-CPD COMs and dienophile C=C COMs.

    For each whole-CPD COM R_CPD and reactive dienophile C=C COM R_CC,
    return min_n |R_CPD-R_CC+n @ cell| in Angstrom, flattened across all pairs.
    The CPD point is not a reactive C=C midpoint.
    """
    elements = get_element_map(atoms, index_array="global_index")
    graph, _, cpd_components, mal_components, _, other = _reaction_components(
        atoms, scale
    )

    if other:
        raise ValueError(
            f"There are {len(other)} molecules that do not match expected sizes."
        )

    cpd_pos = np.asarray([
        mfg.component_center_of_mass(atoms, graph, comp)
        for comp in cpd_components
    ])

    mal_pos = _mal_site_centers(atoms, graph, mal_components, elements)

    if len(cpd_pos) == 0 or len(mal_pos) == 0:
        return np.empty(0, dtype=float)

    _, distances = get_distances(
        cpd_pos,
        mal_pos,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    return distances.ravel()


def _mal_site_centers(atoms: Atoms, graph, mal_components, elements) -> np.ndarray:
    """Reactive C=C COM of each dienophile, in its unwrapped molecular frame.

    Sites come from the molecule's induced subgraph of the frame bond graph,
    which equals a bond graph rebuilt on the isolated molecule.
    """
    centers = []
    for comp in mal_components:
        mal_cc = get_mal_CC(mfg.induced_subgraph(graph, comp), elements)
        if len(mal_cc) != 2:
            raise ValueError(
                f"Could not identify MAL reactive C=C for component {comp[:5]}..."
            )
        centers.append(mfg.component_center_of_mass(atoms, graph, comp, subset=mal_cc))
    return np.asarray(centers)


def _reaction_components(atoms: Atoms, scale: float):
    graph = mfg.build_bond_graph(
        atoms,
        scale=scale,
        index_array="global_index",
    )
    components = mfg.connected_components(
        graph,
        atom_ids=mfg.atom_ids(atoms),
    )

    cpd = [c for c in components if len(c) == CPD_SIZE]
    mal = [c for c in components if len(c) == DIENOPHILE_SIZE]
    product = [c for c in components if len(c) == ADDUCT_SIZE]
    other = [c for c in components if len(c) not in (CPD_SIZE, DIENOPHILE_SIZE, ADDUCT_SIZE)]
    return graph, components, cpd, mal, product, other
