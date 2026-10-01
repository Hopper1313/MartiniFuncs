"""Deprecated legacy shell tools (local row indices); kept only for old scripts.

Use MartiniFuncs.graph (persistent-ID bond graph, components, unwrapping) and
geom_utils.get_pair_shell_clusters / get_cluster_positions_and_forces, which are
minimum-image correct and do not assume LAMMPS ID - 1 == row. Known problems
here: process_lammps_frame fails on orthogonal boxes and on dumps lacking any of
id/mol/mass/type/q/x..fz; shell repair re-centers per atom and is not
minimum-image correct. This module is frozen and will be removed.
"""
import warnings

warnings.warn("MartiniFuncs.shell is deprecated; use MartiniFuncs.graph and "
              "geom_utils.get_pair_shell_clusters", DeprecationWarning, stacklevel=2)


from collections import deque, defaultdict
from typing import Literal, Optional
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, natural_cutoffs
from ase import Atom, Atoms
import numpy as np
import pandas as pd
from .coreio import get_symbol_from_mass

def build_bond_graph(atoms: Atoms,
                     scale: float = 1.15,
                     skin: float = 0.0,
                     periodic: bool = True
                     ) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Build a bonded graph from distances under PBC.

    atoms : ase.Atoms object
    scale : float
        Bond criterion scale factor on covalent radii sum.
    skin : float
        ASE NeighborList skin.

    Returns
    -------
    graph : dict
        graph[i] = list of (j, S_ij)
        where S_ij is the integer lattice shift such that
        r_j + S_ij @ cell - r_i
        is the bonded minimum-image vector.

    Legacy row-index version: |r_j+S_ij @ cell-r_i| <=
    scale*(rcov_i+rcov_j). periodic=False uses zero shifts and a nonperiodic
    neighbor search. New code should use graph.build_bond_graph for persistent IDs.
    """
    numbers = atoms.get_atomic_numbers()
    cell = atoms.cell.array

    # Pair cutoff = scale * (r_cov_i + r_cov_j)
    # NeighborList expects one radius per atom, and pairs interact if overlaps exist
    cutoffs = [scale * covalent_radii[z] for z in numbers]

    nl = NeighborList(cutoffs, skin=skin, bothways=True, self_interaction=False)
    search_atoms = atoms
    if not periodic:
        search_atoms = atoms.copy()
        search_atoms.pbc = False
    nl.update(search_atoms)

    graph = defaultdict(list)


    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)

        for j, S in zip(indices, offsets):
            if j <= i:
                continue

            if periodic:
                S = np.asarray(S, dtype=int)
                rij = atoms.positions[j] + S @ cell - atoms.positions[i]
            elif not periodic:
                S = np.zeros(3, dtype=int)
                rij = atoms.positions[j] - atoms.positions[i]

            dij = np.linalg.norm(rij)

            cutoff_ij = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])

            if dij <= cutoff_ij:
                graph[i].append((j, S))
                graph[j].append((i, -S))

    return graph


def connected_components(graph: dict,
                         n_atoms: int,
                         ) -> list[list[int]]:
    """
    graph : from build_bond_graph
    Find connected components in the bond graph.
    """
    seen = set()
    components = []

    for start in range(n_atoms):
        if start in seen:
            continue

        comp = []
        queue = deque([start])
        seen.add(start)

        while queue:
            i = queue.popleft()
            comp.append(i)

            for j, _ in graph.get(i, []):
                if j not in seen:
                    seen.add(j)
                    queue.append(j)

        components.append(comp)

    return components


def unwrap_component(atoms: Atoms,
                     graph: dict,
                     component: list[int]
                     ) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Assign integer lattice shifts n_i to atoms in one connected component
    so bonded neighbors become contiguous in Cartesian space.

    If atom i has shift n_i, then neighbor j gets:
        n_j = n_i + S_ij

    Coordinates are r_whole,i=r_i+n_i @ cell, with n_root=0 and
    n_j=n_i+S_ij. This legacy routine ignores inconsistent cycles; use
    graph.unwrap_component(strict=True) to reject them.
    """
    cell = atoms.cell.array
    component_set = set(component)

    shifts = {}
    root = component[0]
    shifts[root] = np.zeros(3, dtype=int)

    queue = deque([root])

    while queue:
        i = queue.popleft()

        for j, Sij in graph.get(i, []):
            if j not in component_set:
                continue

            proposed = shifts[i] + Sij

            if j not in shifts:
                shifts[j] = proposed
                queue.append(j)
            else:
                # Ignore inconsistency here; usually means overconnected graph
                pass

    for i in component:
        if i not in shifts:
            shifts[i] = np.zeros(3, dtype=int)

    unwrapped = np.array([
        atoms.positions[i] + shifts[i] @ cell
        for i in component
    ])

    return shifts, unwrapped

def repair_fragmented_molecules(
    atoms: Atoms,
    bond_scale: float = 1.15,
    neighbor_skin: float = 0.0,
    centroid_position_frac: np.ndarray = None,
    # return_components=False,
    ) -> Atoms:
    """Repair molecules fragmented across periodic boundaries.

    Parameters
    ----------
    atoms : ase.Atoms
    bond_scale : float
        Bond criterion scale factor on covalent radii sum.
    neighbor_skin : float
        ASE NeighborList skin.

    Returns
    -------
    repaired : ase.Atoms
        New Atoms object with repaired positions.

    Use r'_i=r_i+n_i @ cell per bonded component. If a fractional
    centroid is supplied, translate all atoms by centroid_position_frac @ cell
    minus mean(r'). This is a geometric, not mass-weighted, centroid.
    """
    repaired = atoms.copy()

    graph = build_bond_graph(repaired, scale=bond_scale, skin=neighbor_skin)
    components = connected_components(graph, len(repaired))

    new_positions = repaired.positions.copy()

    for comp_idx, comp in enumerate(components):
        if len(comp) == 1:
            continue

        atom_shifts, unwrapped = unwrap_component(repaired, graph, comp)

        for k, atom_idx in enumerate(comp):
            new_positions[atom_idx] = unwrapped[k]

    repaired.positions[:] = new_positions

    if centroid_position_frac is not None:
        cell = repaired.cell.array
        centroid_cart = np.mean(repaired.positions, axis=0)
        centroid_frac = np.linalg.solve(cell.T, centroid_cart)
        shift_frac = centroid_position_frac - centroid_frac
        shift_cart = shift_frac @ cell
        repaired.positions += shift_cart

    return repaired

def get_idxs_for_molid(data, molid):
    """Return LAMMPS atom IDs minus one for rows with the requested molecule ID.
    These values equal ASE local rows only for consecutively numbered, ID-sorted
    frames; they are not a general row mapping.
    """
    return [i-1 for i in data.index[data['mol'] == molid].tolist()]


def center_positions_on_atomindex(atoms, atomidx, repair=True, **repair_kwargs):
    """Translate fractional positions by (0.5,0.5,0.5)-s_atomidx, then wrap.
    Mutates the input Atoms; returns a repaired copy when repair=True, otherwise
    the shifted input. atomidx is a local row, not a persistent ID.
    """
    atoms.wrap()
    scaled = atoms.get_scaled_positions(wrap=False)
    shift_frac = np.array([0.5, 0.5, 0.5]) - scaled[atomidx]

    scaled += shift_frac
    atoms.set_scaled_positions(scaled)
    atoms.wrap()

    if repair:
        return repair_fragmented_molecules(atoms, **repair_kwargs)
    return atoms

def identify_all_atoms_within_radius_of_atomindex(atoms, atomidx, radius, **repair_kwargs):
    """After centering/repair, return local rows j with |r_j-r_atomidx|<=radius
    in Angstrom. This is a Cartesian cutoff in that repaired image, not a
    separate minimum-image calculation for each distance.
    """
    repaired = center_positions_on_atomindex(atoms, atomidx, repair=True, **repair_kwargs)
    center_pos = repaired.get_positions()[atomidx]
    all_pos = repaired.get_positions()
    distances = np.linalg.norm(all_pos - center_pos, axis=1)
    return np.where(distances <= radius)[0]

# def get_all_mols_from_atomidxs(data, atomidxs):
#     mols = set()
#     for idx in atomidxs:
#         for molid, molidxs in zip(data['mol'].unique(), mol_sets):
#             if idx in molidxs:
#                 mols.add(molid)
    # return mols
def get_mol_sets(data):
    mol_sets = []
    for mol in data['mol'].unique():
        mol_idxs = get_idxs_for_molid(data, mol)
        mol_sets.append(set(mol_idxs))
    return mol_sets
def get_all_mols_from_atomidxs(data, atomidxs):
    mols = set()
    for idx in atomidxs:
        for molid, molidxs in zip(data['mol'].unique(), get_mol_sets(data)):
            if idx in molidxs:
                mols.add(molid)
    return mols
def complete_mols_from_atomidxs(data, atomidxs):
    mols = get_all_mols_from_atomidxs(data, atomidxs)
    all_idxs = []
    for mol in mols:
        all_idxs.extend(get_idxs_for_molid(data, mol))
    return np.unique(all_idxs)


def index_map(atomidxs):
    return {idx: i for i, idx in enumerate(atomidxs)}

def process_lammps_frame(file):
    timestep = None
    natoms = None
    box = np.zeros((3,3))
    # print(box)
    with open(file,'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'ITEM: TIMESTEP' in line:
                timestep = int(f.readline().strip())
            if 'ITEM: NUMBER OF ATOMS' in line:
                natoms = int(f.readline().strip())
            if 'ITEM: BOX BOUNDS' in line:
                for i in range(3):
                    line = f.readline()
                    parts = line.strip().split()
                    box[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            if 'ITEM: ATOMS' in line:
                rows = []
                columns = line.strip().split()[2:]
                for i in range(natoms):
                    line = f.readline()
                    parts = line.strip().split()
                    rows.append(parts)

                dtypes = {
                            "id": int,
                            "mol": int,
                            "mass": float,
                            "type": int,
                            "q": float,
                            "x": float,
                            "y": float,
                            "z": float,
                            "ix": int,
                            "iy": int,
                            "iz": int,
                            "vx": float,
                            "vy": float,
                            "vz": float,
                            "fx": float,
                            "fy": float,
                            "fz": float,
                        }
                data = pd.DataFrame(rows, columns=columns)
                data = data.astype(dtypes)
                data['atomid'] = data['id'].copy()
                # data['atomindex'] = data['id'] - 1
                data.set_index('id', inplace=True)

                data['symbol'] = data['mass'].apply(get_symbol_from_mass)

    return timestep, natoms, box, data

def get_shell_for_molid(data,test,molid,radius=3.0,bond_scale=1.15,neighbor_skin=0.0,center_on_mol=True,complete=True):
    """Take the union of local rows within radius of each molecule atom;
    complete=True expands to whole molecule-ID groups. This legacy path assumes
    LAMMPS IDs minus one equal ASE rows; use geom_utils for arbitrary IDs.
    """
    molidx = get_idxs_for_molid(data, molid)
    if center_on_mol:
        test = center_positions_on_atomindex(test, molidx[0], repair=True, bond_scale=bond_scale, neighbor_skin=neighbor_skin)
    all_atoms_in_radius = []
    for atom in molidx:
        idxs = identify_all_atoms_within_radius_of_atomindex(test, atom, radius=radius, bond_scale=bond_scale, neighbor_skin=neighbor_skin)
        all_atoms_in_radius.append(idxs)
    all_atoms_in_radius = np.unique(np.concatenate(all_atoms_in_radius))
    if complete:
        all_atoms_in_radius = complete_mols_from_atomidxs(data, all_atoms_in_radius)
    repaired = repair_fragmented_molecules(test[all_atoms_in_radius], bond_scale=bond_scale, neighbor_skin=neighbor_skin)
    return repaired, all_atoms_in_radius

def get_shell_for_molids(data,test,molids,radius=3.0,bond_scale=1.15,neighbor_skin=0.0,center_on_mol=True):

    """Union the radius shells of all selected molecule IDs, then complete
    each touched molecule. Distances are Angstrom; legacy ID-minus-one row
    assumptions apply as in get_shell_for_molid.
    """
    all_atoms_in_radius = []

    for molid in molids:
        _, atoms_in_radius = get_shell_for_molid(data, test, molid, radius=radius, bond_scale=bond_scale, neighbor_skin=neighbor_skin, center_on_mol=center_on_mol,complete=False)
        all_atoms_in_radius.append(atoms_in_radius)
    all_atoms_in_radius = np.unique(np.concatenate(all_atoms_in_radius))
    all_atoms_in_radius_complete = complete_mols_from_atomidxs(data, all_atoms_in_radius)
    repaired = repair_fragmented_molecules(test[all_atoms_in_radius_complete], bond_scale=bond_scale, neighbor_skin=neighbor_skin)


    return repaired, all_atoms_in_radius_complete

def get_mol_idx_maps(data, all_atoms_in_radius_complete, molz):
    indice_map = index_map(all_atoms_in_radius_complete)
    mol1idx_map = [indice_map[index] for index in get_idxs_for_molid(data, molz[0])]
    mol2idx_map = [indice_map[index] for index in get_idxs_for_molid(data, molz[1])]
    return mol1idx_map, mol2idx_map
