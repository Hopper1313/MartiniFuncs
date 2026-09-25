"""Graph and periodic-geometry utilities for MartiniFuncs.

Design rule
-----------
This module keeps two index namespaces deliberately separate:

1. Local ASE index
   The row in the *current* ``Atoms`` object, always ``0 .. len(atoms)-1``.
   Local indices are used whenever indexing ``atoms.positions``, ``atoms.arrays``,
   masses, symbols, etc.

2. Persistent atom ID
   The value stored in ``atoms.arrays[index_array]`` (normally
   ``"global_index"``).  Persistent IDs survive slicing/reordering and are used
   as bond-graph node IDs and in component lists.

Any function that crosses from a graph/component into an ``Atoms`` object must
convert persistent IDs to local ASE indices explicitly.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList, first_neighbors, neighbor_list



AtomID: TypeAlias = int
Shift: TypeAlias = np.ndarray
BondGraph: TypeAlias = dict[AtomID, list[tuple[AtomID, Shift]]]


# -----------------------------------------------------------------------------
# Index / identity helpers
# -----------------------------------------------------------------------------


def ensure_index_array(
    atoms: Atoms,
    index_array: str | None = "global_index",
) -> np.ndarray:
    """Ensure that an Atoms object has a persistent atom-ID array.

    If ``index_array`` is absent it is initialized once as ``0..N-1``. ASE
    preserves per-atom arrays when an ``Atoms`` object is sliced or reordered,
    so these IDs then remain stable. If ``index_array`` is ``None``, local ASE
    indices are returned without creating an array.
    """
    if index_array is None:
        return np.arange(len(atoms), dtype=int)

    if index_array not in atoms.arrays:
        atoms.new_array(index_array, np.arange(len(atoms), dtype=int))

    ids = np.asarray(atoms.arrays[index_array], dtype=int)
    if ids.shape != (len(atoms),):
        raise ValueError(
            f"Atom ID array {index_array!r} must have shape ({len(atoms)},), "
            f"got {ids.shape}."
        )
    if len(np.unique(ids)) != len(ids):
        raise ValueError(
            f"Atom ID array {index_array!r} contains duplicate persistent IDs."
        )
    return ids


# Backward-friendly descriptive alias.
ensure_global_index = ensure_index_array


def get_index_maps(
    atoms: Atoms,
    index_array: str | None = "global_index",
) -> tuple[np.ndarray, dict[int, int]]:
    """Return mappings between local ASE indices and persistent atom IDs.

    Parameters
    ----------
    atoms
        Current ASE ``Atoms`` object.
    index_array
        Name of the persistent-ID array.  If ``None`` or absent, local ASE
        indices are used as IDs.

    Returns
    -------
    local_to_id
        ``local_to_id[local_index] -> persistent_atom_id``.
    id_to_local
        ``id_to_local[persistent_atom_id] -> local_index``.

    Notes
    -----
    Persistent IDs must be unique within an ``Atoms`` object.  Slicing an ASE
    object preserves per-atom arrays, so a sliced object can legitimately have
    IDs such as ``[15, 27, 400, 1200]`` while its local indices remain
    ``[0, 1, 2, 3]``.
    """
    local_to_id = ensure_index_array(atoms, index_array=index_array)

    id_to_local = {
        int(atom_id): int(local_idx)
        for local_idx, atom_id in enumerate(local_to_id)
    }

    return local_to_id, id_to_local


def atom_ids(
    atoms: Atoms,
    index_array: str | None = "global_index",
) -> np.ndarray:
    """Return persistent IDs for all atoms in current local order."""
    local_to_id, _ = get_index_maps(atoms, index_array=index_array)
    return local_to_id.copy()


def local_index(
    atoms: Atoms,
    atom_id: int,
    index_array: str | None = "global_index",
) -> int:
    """Convert one persistent atom ID to a local ASE index."""
    _, id_to_local = get_index_maps(atoms, index_array=index_array)
    try:
        return id_to_local[int(atom_id)]
    except KeyError as exc:
        raise KeyError(
            f"Persistent atom ID {int(atom_id)} is not present in this Atoms object."
        ) from exc


def local_indices(
    atoms: Atoms,
    ids: Iterable[int],
    index_array: str | None = "global_index",
) -> np.ndarray:
    """Convert persistent atom IDs to local ASE indices, preserving order."""
    ids = [int(i) for i in ids]
    _, id_to_local = get_index_maps(atoms, index_array=index_array)

    missing = [i for i in ids if i not in id_to_local]
    if missing:
        raise KeyError(
            "Persistent atom IDs are not present in this Atoms object: "
            f"{missing[:10]}" + (" ..." if len(missing) > 10 else "")
        )

    return np.fromiter(
        (id_to_local[i] for i in ids),
        dtype=int,
        count=len(ids),
    )


# -----------------------------------------------------------------------------
# Bond graph construction and connected components
# -----------------------------------------------------------------------------


def build_bond_graph(
    atoms: Atoms,
    scale: float = 1.15,
    skin: float = 0.0,
    periodic: bool = True,
    index_array: str | None = "global_index",
) -> BondGraph:
    """Build a covalent bond graph using persistent atom IDs as graph nodes.

    ``graph[i] = [(j, S_ij), ...]`` where ``i`` and ``j`` are persistent atom
    IDs and ``S_ij`` is the integer lattice shift satisfying

    ``r_j + S_ij @ cell - r_i``

    for the bonded periodic image.

    Local ASE indices are used internally for all coordinate-array access, then
    converted to persistent IDs when graph edges are stored.

    A bond satisfies |r_j+S_ij @ cell-r_i| <= scale*(rcov_i+rcov_j).
    Cell and positions are Angstrom, S_ij integer image shifts. skin expands
    neighbor candidates only; it does not change the final bond criterion.
    """
    numbers = atoms.get_atomic_numbers()
    cell = atoms.cell.array
    local_to_id, _ = get_index_maps(atoms, index_array=index_array)

    # If periodic=False, make the neighbor search itself non-periodic rather
    # than merely discarding offsets afterwards.
    search_atoms = atoms
    if not periodic and np.any(atoms.pbc):
        search_atoms = atoms.copy()
        search_atoms.pbc = False

    cutoffs = scale * covalent_radii[numbers]

    nl = NeighborList(
        cutoffs,
        skin=skin,
        bothways=True,
        self_interaction=False,
    )
    nl.update(search_atoms)

    graph: BondGraph = defaultdict(list)

    for local_i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(local_i)

        for local_j, S in zip(neighbors, offsets):
            local_j = int(local_j)

            # NeighborList is bothways=True. Store each bond once here, then
            # explicitly insert both graph directions with opposite shifts.
            if local_j <= local_i:
                continue

            S = np.asarray(S, dtype=int)

            if periodic:
                rij = (
                    atoms.positions[local_j]
                    + S @ cell
                    - atoms.positions[local_i]
                )
            else:
                S = np.zeros(3, dtype=int)
                rij = atoms.positions[local_j] - atoms.positions[local_i]

            dij = np.linalg.norm(rij)
            cutoff_ij = scale * (
                covalent_radii[numbers[local_i]]
                + covalent_radii[numbers[local_j]]
            )

            if dij <= cutoff_ij:
                i = int(local_to_id[local_i])
                j = int(local_to_id[local_j])

                graph[i].append((j, S.copy()))
                graph[j].append((i, -S.copy()))

    return graph


def _graph_nodes(graph: Mapping[int, Sequence[tuple[int, np.ndarray]]]) -> set[int]:
    """Return all node IDs appearing either as graph keys or neighbors."""
    nodes = {int(i) for i in graph}
    for neighbors in graph.values():
        nodes.update(int(j) for j, _ in neighbors)
    return nodes


def connected_components(
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    n_atoms: int | Iterable[int] | None = None,
    *,
    atom_ids: Iterable[int] | None = None,
) -> list[list[int]]:
    """Find connected components using persistent graph-node IDs.

    Parameters
    ----------
    graph
        Bond graph whose node IDs may be arbitrary persistent integers.
    n_atoms
        Backward-compatible argument.

        * If an integer is supplied, IDs are assumed to be exactly
          ``0 .. n_atoms-1``.  A ``ValueError`` is raised if the graph contains
          IDs outside that range, because that usually means persistent IDs
          were mistaken for local indices.
        * If an iterable is supplied, it is interpreted as the atom IDs.
    atom_ids
        Preferred explicit iterable of all persistent IDs that should be
        considered.  Supplying this is necessary to retain isolated atoms,
        which do not appear in the bond graph.

    Returns
    -------
    components
        Lists of persistent atom IDs.
    """
    if atom_ids is not None and n_atoms is not None:
        raise TypeError("Pass either atom_ids or n_atoms, not both.")

    if atom_ids is not None:
        ids = [int(i) for i in atom_ids]
    elif n_atoms is None:
        # Without an explicit ID list, isolated atoms cannot be recovered.
        ids = sorted(_graph_nodes(graph))
    elif np.isscalar(n_atoms):
        n = int(n_atoms)
        ids = list(range(n))
        bad_nodes = sorted(i for i in _graph_nodes(graph) if i < 0 or i >= n)
        if bad_nodes:
            raise ValueError(
                "connected_components(graph, n_atoms=...) assumes graph IDs "
                "are local contiguous indices 0..n_atoms-1, but graph contains "
                f"persistent IDs outside that range (e.g. {bad_nodes[:5]}). "
                "Use atom_ids=mfg.atom_ids(atoms) instead."
            )
    else:
        # Allows a gradual migration from old positional calls:
        # connected_components(graph, some_atom_id_iterable)
        ids = [int(i) for i in n_atoms]

    if len(set(ids)) != len(ids):
        raise ValueError("atom_ids contains duplicate IDs.")

    allowed = set(ids)
    graph_nodes = _graph_nodes(graph)
    outside = graph_nodes - allowed
    if outside:
        raise ValueError(
            "Bond graph contains IDs not present in atom_ids: "
            f"{sorted(outside)[:10]}" + (" ..." if len(outside) > 10 else "")
        )

    seen: set[int] = set()
    components: list[list[int]] = []

    for start in ids:
        if start in seen:
            continue

        comp: list[int] = []
        queue = deque([start])
        seen.add(start)

        while queue:
            i = queue.popleft()
            comp.append(i)

            for j, _ in graph.get(i, []):
                j = int(j)
                if j not in seen:
                    seen.add(j)
                    queue.append(j)

        components.append(comp)

    return components


def component_lookup(
    components: Sequence[Iterable[int]],
) -> dict[int, int]:
    """Map persistent atom ID -> component index.

    A dictionary is intentionally used instead of a NumPy array so that sparse
    or non-contiguous persistent IDs remain safe after slicing/reordering.
    """
    lookup: dict[int, int] = {}

    for component_id, component in enumerate(components):
        for atom_id in component:
            atom_id = int(atom_id)
            if atom_id in lookup:
                raise ValueError(
                    f"Persistent atom ID {atom_id} occurs in more than one component."
                )
            lookup[atom_id] = component_id

    return lookup


# -----------------------------------------------------------------------------
# Unwrapping / positioning complete molecular components
# -----------------------------------------------------------------------------


def unwrap_component(
    atoms: Atoms,
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    component: Iterable[int],
    index_array: str | None = "global_index",
    *,
    strict: bool = False,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Unwrap one connected molecular component under PBC.

    Parameters
    ----------
    atoms
        ASE object containing the component.
    graph
        Bond graph keyed by persistent atom IDs.
    component
        Persistent atom IDs in the component.  Output coordinates follow this
        same order.
    index_array
        Persistent-ID array on ``atoms``.
    strict
        If ``True``, raise when a graph cycle implies inconsistent lattice
        shifts or when ``component`` is not connected.  If ``False``, preserve
        the historical permissive behavior for such cases.

    Returns
    -------
    shifts
        ``{atom_id: integer lattice shift}`` relative to the chosen root.
    unwrapped
        Cartesian coordinates in ``component`` order.

    Set root shift n_root=0, propagate n_j=n_i+S_ij along graph edges,
    and return r_whole,i=r_i+n_i @ cell. Output follows component input order.
    """
    component = [int(i) for i in component]
    if not component:
        return {}, np.empty((0, 3), dtype=float)

    component_set = set(component)
    if len(component_set) != len(component):
        raise ValueError("component contains duplicate atom IDs.")

    local = local_indices(atoms, component, index_array=index_array)
    cell = atoms.cell.array

    shifts: dict[int, np.ndarray] = {}
    root = component[0]
    shifts[root] = np.zeros(3, dtype=int)
    queue = deque([root])

    while queue:
        i = queue.popleft()

        for j, Sij in graph.get(i, []):
            j = int(j)
            if j not in component_set:
                continue

            proposed = shifts[i] + np.asarray(Sij, dtype=int)

            if j not in shifts:
                shifts[j] = proposed
                queue.append(j)
            elif strict and not np.array_equal(shifts[j], proposed):
                raise ValueError(
                    "Inconsistent periodic shifts encountered while unwrapping "
                    f"component: atom {j} has {shifts[j]} and proposed {proposed}."
                )

    missing = component_set - shifts.keys()
    if missing:
        if strict:
            raise ValueError(
                "unwrap_component expected one connected component, but these "
                f"atom IDs were unreachable from root {root}: {sorted(missing)}"
            )
        for atom_id in missing:
            shifts[atom_id] = np.zeros(3, dtype=int)

    shift_array = np.vstack([shifts[i] for i in component])
    unwrapped = atoms.positions[local] + shift_array @ cell

    return shifts, unwrapped


def place_component(
    atoms: Atoms,
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    component: Iterable[int],
    image_shift: Iterable[int] = (0, 0, 0),
    index_array: str | None = "global_index",
    *,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Unwrap a component, then translate the whole molecule by a cell image.

    Returns persistent IDs, Cartesian coordinates, and absolute per-atom lattice
    shifts, all in the component's input order.

    Absolute shift Q_i=n_i+image_shift from unwrap_component;
    xyz_i=r_i+Q_i @ cell in Angstrom.
    """
    component = [int(i) for i in component]
    image_shift = np.asarray(image_shift, dtype=int)
    if image_shift.shape != (3,):
        raise ValueError("image_shift must have shape (3,).")

    relative_shifts, unwrapped = unwrap_component(
        atoms,
        graph,
        component,
        index_array=index_array,
        strict=strict,
    )

    absolute_shifts = {
        atom_id: relative_shifts[atom_id] + image_shift
        for atom_id in component
    }

    xyz = unwrapped + image_shift @ atoms.cell.array

    return np.asarray(component, dtype=int), xyz, absolute_shifts


def place_component_from_anchor(
    atoms: Atoms,
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    component: Iterable[int],
    anchor_atom: int,
    anchor_shift: Iterable[int],
    index_array: str | None = "global_index",
    *,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Place a complete component so one atom occupies a requested PBC image.

    This is useful for shell construction: a geometric contact identifies the
    periodic image of one atom, while the bond graph determines how the rest of
    that molecule must be unwrapped around it.

    Given relative shifts n_i, set translation=anchor_shift-n_anchor.
    Q_i=n_i+translation and xyz_i=r_i+Q_i @ cell, so the anchor occupies
    the requested absolute image.
    """
    component = [int(i) for i in component]
    anchor_atom = int(anchor_atom)
    anchor_shift = np.asarray(anchor_shift, dtype=int)

    if anchor_atom not in set(component):
        raise ValueError(f"anchor_atom {anchor_atom} is not in component.")
    if anchor_shift.shape != (3,):
        raise ValueError("anchor_shift must have shape (3,).")

    relative_shifts, _ = unwrap_component(
        atoms,
        graph,
        component,
        index_array=index_array,
        strict=strict,
    )

    translation = anchor_shift - relative_shifts[anchor_atom]
    absolute_shifts = {
        atom_id: relative_shifts[atom_id] + translation
        for atom_id in component
    }

    local = local_indices(atoms, component, index_array=index_array)
    shift_array = np.vstack([absolute_shifts[i] for i in component])
    xyz = atoms.positions[local] + shift_array @ atoms.cell.array

    return np.asarray(component, dtype=int), xyz, absolute_shifts


def component_center_of_mass(
    atoms: Atoms,
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    component: Iterable[int],
    index_array: str | None = "global_index",
    *,
    strict: bool = False,
) -> np.ndarray:
    """Center of mass of a component after molecular PBC unwrapping.

    R=sum_i(m_i*r_whole,i)/sum_i m_i, using component IDs, ASE masses,
    and unwrap_component positions; result is Angstrom.
    """
    component = [int(i) for i in component]
    if not component:
        raise ValueError("component must contain at least one atom.")

    local = local_indices(atoms, component, index_array=index_array)
    _, xyz = unwrap_component(
        atoms,
        graph,
        component,
        index_array=index_array,
        strict=strict,
    )
    masses = atoms.get_masses()[local]
    return np.average(xyz, axis=0, weights=masses)


def repair_fragmented_molecules(
    atoms: Atoms,
    bond_scale: float = 1.15,
    neighbor_skin: float = 0.0,
    centroid_position_frac: np.ndarray | None = None,
    index_array: str | None = "global_index",
) -> Atoms:
    """Return a copy with every bonded molecule made contiguous under PBC.

    After graph unwrapping r'_i=r_i+n_i @ cell, optional centering uses
    t=(centroid_position_frac-mean(r') @ inverse(cell)) @ cell.
    All atoms receive t; this centers the geometric centroid, not the COM.
    """
    repaired = atoms.copy()

    graph = build_bond_graph(
        repaired,
        scale=bond_scale,
        skin=neighbor_skin,
        index_array=index_array,
    )

    ids, id_to_local = get_index_maps(repaired, index_array=index_array)
    components = connected_components(graph, atom_ids=ids)
    new_positions = repaired.positions.copy()

    for comp in components:
        if len(comp) == 1:
            continue

        _, unwrapped = unwrap_component(
            repaired,
            graph,
            comp,
            index_array=index_array,
        )

        local = np.fromiter(
            (id_to_local[int(atom_id)] for atom_id in comp),
            dtype=int,
            count=len(comp),
        )
        new_positions[local] = unwrapped

    repaired.positions[:] = new_positions

    if centroid_position_frac is not None:
        centroid_position_frac = np.asarray(centroid_position_frac, dtype=float)
        if centroid_position_frac.shape != (3,):
            raise ValueError("centroid_position_frac must have shape (3,).")

        cell = repaired.cell.array
        centroid_cart = np.mean(repaired.positions, axis=0)
        centroid_frac = np.linalg.solve(cell.T, centroid_cart)
        shift_frac = centroid_position_frac - centroid_frac
        repaired.positions += shift_frac @ cell

    return repaired


# -----------------------------------------------------------------------------
# Reusable geometric shell neighbor index
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellNeighborIndex:
    """Precomputed atom-atom geometric neighbor lookup for one Atoms frame."""

    cutoff: float
    neighbors_local: np.ndarray
    distances: np.ndarray
    relative_shifts: np.ndarray
    seed: np.ndarray
    local_to_id: np.ndarray
    id_to_local: dict[int, int]


@dataclass(frozen=True)
class ShellContacts:
    """Unique pair-to-shell contacts in a common periodic coordinate frame."""

    center_ids: np.ndarray
    atom_ids: np.ndarray
    distances: np.ndarray
    shifts: np.ndarray

    def __len__(self) -> int:
        return len(self.atom_ids)


def build_shell_neighbor_index(
    atoms: Atoms,
    cutoff: float = 3.0,
    index_array: str | None = "global_index",
) -> ShellNeighborIndex:
    """Build a reusable geometric neighbor lookup for shell construction.

    Build this once per ``Atoms`` frame.  Pair-specific shell queries then touch
    only the neighbor-list entries of atoms in that molecular pair.

    Store contacts satisfying |r_j+S_ij @ cell-r_i|<cutoff, in Angstrom.
    Rebuild after geometry/cell changes; this index is specific to one frame.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0.")

    i, j, d, S = neighbor_list(
        "ijdS",
        atoms,
        cutoff=cutoff,
        self_interaction=False,
    )

    seed = first_neighbors(len(atoms), i)
    local_to_id, id_to_local = get_index_maps(atoms, index_array=index_array)

    return ShellNeighborIndex(
        cutoff=float(cutoff),
        neighbors_local=np.asarray(j, dtype=int),
        distances=np.asarray(d, dtype=float),
        relative_shifts=np.asarray(S, dtype=int),
        seed=np.asarray(seed, dtype=int),
        local_to_id=local_to_id.copy(),
        id_to_local=id_to_local,
    )


def get_shell_contacts(
    pair_shifts: Mapping[int, Iterable[int]],
    shell_index: ShellNeighborIndex,
    *,
    exclude_pair: bool = True,
) -> ShellContacts:
    """Find atoms within the shell cutoff of any atom in a placed pair.

    Parameters
    ----------
    pair_shifts
        Mapping ``persistent_pair_atom_id -> absolute integer lattice shift``.
        All pair shifts must refer to one common local-cluster coordinate frame.
    shell_index
        Precomputed output of :func:`build_shell_neighbor_index`.
    exclude_pair
        Exclude atoms belonging to the reactive pair itself.

    Returns
    -------
    ShellContacts
        ``shifts[k]`` is the *absolute* lattice shift for ``atom_ids[k]`` in the
        same coordinate frame as ``pair_shifts``.

    Notes
    -----
    ASE's neighbor shift is relative to the center atom.  If the center already
    has absolute shift ``Q_i``, a neighbor contact with relative shift ``S_ij``
    belongs at absolute shift ``Q_j = Q_i + S_ij``.

    Contacts are deduplicated by ``(atom_id, absolute_shift)`` and the shortest
    pair-atom contact is retained for each such periodic image.
    """
    pair_shift_map: dict[int, np.ndarray] = {}
    for atom_id, shift in pair_shifts.items():
        atom_id = int(atom_id)
        shift = np.asarray(shift, dtype=int)
        if shift.shape != (3,):
            raise ValueError(
                f"Shift for pair atom {atom_id} must have shape (3,), got {shift.shape}."
            )
        if atom_id not in shell_index.id_to_local:
            raise KeyError(
                f"Pair atom ID {atom_id} is not present in the Atoms object used "
                "to build this ShellNeighborIndex."
            )
        pair_shift_map[atom_id] = shift

    pair_ids = set(pair_shift_map)

    # key -> (center_id, atom_id, distance, absolute_shift)
    best: dict[tuple[int, int, int, int], tuple[int, int, float, np.ndarray]] = {}

    for center_id, center_shift in pair_shift_map.items():
        center_local = shell_index.id_to_local[center_id]
        start = shell_index.seed[center_local]
        stop = shell_index.seed[center_local + 1]

        neigh_local = shell_index.neighbors_local[start:stop]
        neigh_distance = shell_index.distances[start:stop]
        neigh_relative_shift = shell_index.relative_shifts[start:stop]
        neigh_ids = shell_index.local_to_id[neigh_local]

        for atom_id, distance, rel_shift in zip(
            neigh_ids,
            neigh_distance,
            neigh_relative_shift,
        ):
            atom_id = int(atom_id)
            if exclude_pair and atom_id in pair_ids:
                continue

            absolute_shift = center_shift + rel_shift
            key = (
                atom_id,
                int(absolute_shift[0]),
                int(absolute_shift[1]),
                int(absolute_shift[2]),
            )

            previous = best.get(key)
            if previous is None or float(distance) < previous[2]:
                best[key] = (
                    center_id,
                    atom_id,
                    float(distance),
                    np.asarray(absolute_shift, dtype=int).copy(),
                )

    if not best:
        return ShellContacts(
            center_ids=np.empty(0, dtype=int),
            atom_ids=np.empty(0, dtype=int),
            distances=np.empty(0, dtype=float),
            shifts=np.empty((0, 3), dtype=int),
        )

    # Stable deterministic output: shortest contacts first, then atom/image.
    values = sorted(
        best.values(),
        key=lambda x: (x[2], x[1], int(x[3][0]), int(x[3][1]), int(x[3][2])),
    )

    return ShellContacts(
        center_ids=np.asarray([x[0] for x in values], dtype=int),
        atom_ids=np.asarray([x[1] for x in values], dtype=int),
        distances=np.asarray([x[2] for x in values], dtype=float),
        shifts=np.vstack([x[3] for x in values]),
    )


def shell_component_ids(
    contacts: ShellContacts,
    atom_to_component: Mapping[int, int],
    *,
    exclude_atom_ids: Iterable[int] = (),
) -> list[int]:
    """Return component IDs touched by shell contacts.

    ``atom_to_component`` should normally come from :func:`component_lookup`.
    ``exclude_atom_ids`` can be the reactive pair to ensure its own components
    are not returned as shell molecules.
    """
    excluded_components = {
        atom_to_component[int(atom_id)]
        for atom_id in exclude_atom_ids
        if int(atom_id) in atom_to_component
    }

    touched: set[int] = set()
    missing: list[int] = []

    for atom_id in contacts.atom_ids:
        atom_id = int(atom_id)
        if atom_id not in atom_to_component:
            missing.append(atom_id)
            continue

        component_id = int(atom_to_component[atom_id])
        if component_id not in excluded_components:
            touched.add(component_id)

    if missing:
        raise KeyError(
            "Shell contact atom IDs were missing from atom_to_component: "
            f"{missing[:10]}" + (" ..." if len(missing) > 10 else "")
        )

    return sorted(touched)


def place_shell_components(
    atoms: Atoms,
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    components: Sequence[Iterable[int]],
    atom_to_component: Mapping[int, int],
    contacts: ShellContacts,
    *,
    exclude_atom_ids: Iterable[int] = (),
    index_array: str | None = "global_index",
    strict: bool = False,
) -> list[dict[str, object]]:
    """Complete and place every shell molecule using its closest contact anchor.

    Each touched component is unwrapped with :func:`unwrap_component`; its
    closest shell-contact atom supplies the absolute periodic image for the
    entire molecule.

    Choose each molecule's contact with minimum recorded distance.
    For its anchor a, Q_i=n_i+(Q_a-n_a) and xyz_i=r_i+Q_i @ cell;
    n_i are relative bond-unwrapping shifts.
    """
    shell_ids = shell_component_ids(
        contacts,
        atom_to_component,
        exclude_atom_ids=exclude_atom_ids,
    )

    contact_component = np.asarray(
        [atom_to_component[int(atom_id)] for atom_id in contacts.atom_ids],
        dtype=int,
    )

    placed: list[dict[str, object]] = []

    for component_id in shell_ids:
        candidates = np.flatnonzero(contact_component == component_id)
        if len(candidates) == 0:
            continue

        best = int(candidates[np.argmin(contacts.distances[candidates])])
        anchor_atom = int(contacts.atom_ids[best])
        anchor_shift = contacts.shifts[best]

        ids, xyz, shifts = place_component_from_anchor(
            atoms,
            graph,
            components[component_id],
            anchor_atom=anchor_atom,
            anchor_shift=anchor_shift,
            index_array=index_array,
            strict=strict,
        )

        placed.append(
            {
                "component_id": int(component_id),
                "indices": ids,
                "xyz": xyz,
                "shifts": shifts,
                "anchor_atom": anchor_atom,
                "anchor_shift": np.asarray(anchor_shift, dtype=int).copy(),
                "anchor_distance": float(contacts.distances[best]),
            }
        )

    return placed


# -----------------------------------------------------------------------------
# Graph conversion / graph-neighborhood utilities
# -----------------------------------------------------------------------------


def bond_graph_to_networkx(graph: Mapping[int, Sequence[tuple[int, np.ndarray]]]) -> nx.Graph:
    """Convert the bond graph to a NetworkX graph."""
    G = nx.Graph()

    # Preserve isolated nodes when they exist as explicit graph keys.
    G.add_nodes_from(int(i) for i in graph)

    for i, neighbors in graph.items():
        i = int(i)
        for j, S in neighbors:
            G.add_edge(
                i,
                int(j),
                shift=np.asarray(S, dtype=int).copy(),
            )

    return G


def find_rings(graph, max_size=None):
    """Return chordless cycles in the bond graph."""
    G = bond_graph_to_networkx(graph)
    return list(nx.chordless_cycles(G, length_bound=max_size))


def n_neighbor_subgraph(graph, subset, n=1):
    """Extract subset atoms plus all atoms within ``n`` bond hops."""
    if n < 0:
        raise ValueError("n must be >= 0")

    subset = set(map(int, subset))
    shells = [subset.copy()]
    selected_atoms = subset.copy()
    frontier = subset.copy()

    for _ in range(n):
        next_frontier = set()

        for i in frontier:
            for j, _ in graph.get(i, []):
                j = int(j)
                if j not in selected_atoms:
                    next_frontier.add(j)

        shells.append(next_frontier)
        selected_atoms.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    subgraph = defaultdict(list)
    for i in selected_atoms:
        for j, S in graph.get(i, []):
            j = int(j)
            if j in selected_atoms:
                subgraph[i].append((j, np.asarray(S, dtype=int).copy()))

    return subgraph, selected_atoms, shells


def n_neighbor_indices(graph, subset, n=1):
    """Return persistent atom IDs within <= ``n`` bond hops of ``subset``."""
    if n < 0:
        raise ValueError("n must be >= 0")

    subset = set(map(int, subset))
    selected = subset.copy()
    frontier = subset.copy()

    for _ in range(n):
        next_frontier = {
            int(j)
            for i in frontier
            for j, _ in graph.get(i, [])
            if int(j) not in selected
        }

        selected.update(next_frontier)
        frontier = next_frontier

        if not frontier:
            break

    return selected


def extract_neighborhood(graph, subset, n=1):
    """Return selected IDs and their induced graph within ``n`` bond hops."""
    selected = n_neighbor_indices(graph, subset, n)
    subgraph = defaultdict(list)

    for i in selected:
        for j, S in graph.get(i, []):
            j = int(j)
            if j in selected:
                subgraph[i].append((j, np.asarray(S, dtype=int).copy()))

    return selected, subgraph


def find_element_chains(
    graph,
    elements,
    pattern,
    unique_reverse=True,
):
    """Find simple bonded paths whose element sequence matches ``pattern``."""
    if isinstance(pattern, str):
        pattern = list(pattern)

    pattern = tuple(pattern)
    if len(pattern) == 0:
        return []

    matches = []
    seen = set()

    def dfs(path, depth):
        if depth == len(pattern) - 1:
            result = tuple(path)

            if unique_reverse:
                canonical = min(result, result[::-1])
                if canonical in seen:
                    return
                seen.add(canonical)

            matches.append(result)
            return

        current = path[-1]
        required_element = pattern[depth + 1]

        for neighbor, _ in graph.get(current, []):
            neighbor = int(neighbor)

            if neighbor in path:
                continue
            if elements.get(neighbor) != required_element:
                continue

            dfs(path + [neighbor], depth + 1)

    for atom, element in elements.items():
        atom = int(atom)
        if element == pattern[0]:
            dfs([atom], 0)

    return matches


def complete_molecule(graph, seed):
    """Return complete graph component(s) containing one or more seed atom IDs."""
    if isinstance(seed, Mapping):
        seeds = {int(i) for i in seed.keys()}
    elif isinstance(seed, (int, np.integer)):
        seeds = {int(seed)}
    else:
        seeds = {int(i) for i in seed}

    if not seeds:
        return set(), defaultdict(list)

    visited = set(seeds)
    queue = deque(seeds)

    while queue:
        i = queue.popleft()

        for j, _ in graph.get(i, []):
            j = int(j)
            if j not in visited:
                visited.add(j)
                queue.append(j)

    subgraph = defaultdict(list)
    for i in visited:
        for j, S in graph.get(i, []):
            j = int(j)
            if j in visited:
                subgraph[i].append((j, np.asarray(S, dtype=int).copy()))

    return visited, subgraph


######################################################################
#######-------------------------------------------------------########
####### Below is before refactoring.  Saving for reference.   ########
#######-------------------------------------------------------########
######################################################################

# from collections import deque, defaultdict
# from typing import Literal, Optional
# from ase.data import covalent_radii
# from ase.neighborlist import NeighborList, natural_cutoffs
# from ase import Atom, Atoms
# import numpy as np
# import pandas as pd
# from .coreio import get_symbol_from_mass
# from .geom_utils import add_global_index
# import networkx as nx

# from collections import defaultdict

# import numpy as np
# from ase import Atoms
# from ase.data import covalent_radii
# from ase.neighborlist import NeighborList

# import inspect
# import functools


# @add_global_index
# def build_bond_graph(
#     atoms: Atoms,
#     scale: float = 1.15,
#     skin: float = 0.0,
#     periodic: bool = True,
#     index_array: str | None = "global_index",
# ) -> dict[int, list[tuple[int, np.ndarray]]]:
#     """
#     Build a bonded graph from distances under PBC.

#     Parameters
#     ----------
#     atoms : ase.Atoms
#         Atomic structure.

#     scale : float
#         Bond criterion scale factor on covalent radii sum.

#     skin : float
#         ASE NeighborList skin.

#     periodic : bool
#         Whether to use periodic offsets.

#     index_array : str or None
#         Name of an ASE atom array containing persistent/global atom indices.

#         If the array exists, graph keys use those indices.

#         If None or the array does not exist, local ASE indices are used.

#     Returns
#     -------
#     graph : dict[int, list[tuple[int, np.ndarray]]]
#         graph[i] = [(j, S_ij), ...]

#         i and j are persistent/global indices when `index_array`
#         is available.

#         S_ij is the integer lattice shift such that

#             r_j + S_ij @ cell - r_i

#         gives the bonded periodic image.
#     """

#     numbers = atoms.get_atomic_numbers()
#     cell = atoms.cell.array

#     # Map local ASE indices -> persistent/global indices
#     if (
#         index_array is not None
#         and index_array in atoms.arrays
#     ):
#         global_indices = np.asarray(
#             atoms.arrays[index_array],
#             dtype=int,
#         )
#     else:
#         global_indices = np.arange(len(atoms), dtype=int)

#     # NeighborList expects one cutoff radius per atom
#     cutoffs = scale * covalent_radii[numbers]

#     nl = NeighborList(
#         cutoffs,
#         skin=skin,
#         bothways=True,
#         self_interaction=False,
#     )

#     nl.update(atoms)

#     graph = defaultdict(list)

#     for local_i in range(len(atoms)):
#         neighbors, offsets = nl.get_neighbors(local_i)

#         for local_j, S in zip(neighbors, offsets):

#             # Each bond is returned both ways.
#             # Compare LOCAL indices here.
#             if local_j <= local_i:
#                 continue

#             S = np.asarray(S, dtype=int)

#             if periodic:
#                 rij = (
#                     atoms.positions[local_j]
#                     + S @ cell
#                     - atoms.positions[local_i]
#                 )
#             else:
#                 rij = (
#                     atoms.positions[local_j]
#                     - atoms.positions[local_i]
#                 )
#                 S = np.zeros(3, dtype=int)

#             dij = np.linalg.norm(rij)

#             cutoff_ij = scale * (
#                 covalent_radii[numbers[local_i]]
#                 + covalent_radii[numbers[local_j]]
#             )

#             if dij <= cutoff_ij:

#                 # Convert local ASE indices to persistent indices
#                 i = int(global_indices[local_i])
#                 j = int(global_indices[local_j])

#                 graph[i].append((j, S.copy()))
#                 graph[j].append((i, -S.copy()))

#     return graph

# # def build_bond_graph(atoms: Atoms,
# #                      scale: float = 1.15,
# #                      skin: float = 0.0,
# #                      periodic: bool = True
# #                      ) -> dict[int, list[tuple[int, np.ndarray]]]:
# #     """
# #     Build a bonded graph from distances under PBC.

# #     atoms : ase.Atoms object
# #     scale : float
# #         Bond criterion scale factor on covalent radii sum.
# #     skin : float
# #         ASE NeighborList skin.

# #     Returns
# #     -------
# #     graph : dict
# #         graph[i] = list of (j, S_ij)
# #         where S_ij is the integer lattice shift such that
# #         r_j + S_ij @ cell - r_i
# #         is the bonded minimum-image vector.
# #     """
# #     numbers = atoms.get_atomic_numbers()
# #     cell = atoms.cell.array

# #     # Pair cutoff = scale * (r_cov_i + r_cov_j)
# #     # NeighborList expects one radius per atom, and pairs interact if overlaps exist
# #     cutoffs = [scale * covalent_radii[z] for z in numbers]

# #     nl = NeighborList(cutoffs, skin=skin, bothways=True, self_interaction=False)
# #     nl.update(atoms)

# #     graph = defaultdict(list)


# #     for i in range(len(atoms)):
# #         indices, offsets = nl.get_neighbors(i)

# #         for j, S in zip(indices, offsets):
# #             if j <= i:
# #                 continue

# #             if periodic:
# #                 S = np.asarray(S, dtype=int)
# #                 rij = atoms.positions[j] + S @ cell - atoms.positions[i]
# #             elif not periodic:
# #                 rij = atoms.positions[j] - atoms.positions[i]

# #             dij = np.linalg.norm(rij)

# #             cutoff_ij = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])

# #             if dij <= cutoff_ij:
# #                 graph[i].append((j, S))
# #                 graph[j].append((i, -S))

# #     return graph


# def connected_components(graph: dict,
#                          n_atoms: int,
#                          ) -> list[list[int]]:
#     """
#     graph : from build_bond_graph
#     Find connected components in the bond graph.
#     """
#     seen = set()
#     components = []

#     for start in range(n_atoms):
#         if start in seen:
#             continue

#         comp = []
#         queue = deque([start])
#         seen.add(start)

#         while queue:
#             i = queue.popleft()
#             comp.append(i)

#             for j, _ in graph.get(i, []):
#                 if j not in seen:
#                     seen.add(j)
#                     queue.append(j)

#         components.append(comp)

#     return components


# def unwrap_component(atoms: Atoms,
#                      graph: dict,
#                      component: list[int]
#                      ) -> tuple[dict[int, np.ndarray], np.ndarray]:
#     """
#     Assign integer lattice shifts n_i to atoms in one connected component
#     so bonded neighbors become contiguous in Cartesian space.

#     If atom i has shift n_i, then neighbor j gets:
#         n_j = n_i + S_ij
#     """
#     cell = atoms.cell.array
#     component_set = set(component)

#     shifts = {}
#     root = component[0]
#     shifts[root] = np.zeros(3, dtype=int)

#     queue = deque([root])

#     while queue:
#         i = queue.popleft()

#         for j, Sij in graph.get(i, []):
#             if j not in component_set:
#                 continue

#             proposed = shifts[i] + Sij

#             if j not in shifts:
#                 shifts[j] = proposed
#                 queue.append(j)
#             else:
#                 # Ignore inconsistency here; usually means overconnected graph
#                 pass

#     for i in component:
#         if i not in shifts:
#             shifts[i] = np.zeros(3, dtype=int)

#     unwrapped = np.array([
#         atoms.positions[i] + shifts[i] @ cell
#         for i in component
#     ])

#     return shifts, unwrapped

# def repair_fragmented_molecules(
#     atoms: Atoms,
#     bond_scale: float = 1.15,
#     neighbor_skin: float = 0.0,
#     centroid_position_frac: np.ndarray = None,
#     # return_components=False,
#     ) -> Atoms:
#     """
#     Repair molecules fragmented across periodic boundaries.

#     Parameters
#     ----------
#     atoms : ase.Atoms
#     bond_scale : float
#         Bond criterion scale factor on covalent radii sum.
#     neighbor_skin : float
#         ASE NeighborList skin.

#     Returns
#     -------
#     repaired : ase.Atoms
#         New Atoms object with repaired positions.
#     info : dict, optional
#         Extra information if return_components=True
#     """
#     repaired = atoms.copy()

#     graph = build_bond_graph(repaired, scale=bond_scale, skin=neighbor_skin)
#     components = connected_components(graph, len(repaired))

#     new_positions = repaired.positions.copy()

#     for comp_idx, comp in enumerate(components):
#         if len(comp) == 1:
#             continue

#         atom_shifts, unwrapped = unwrap_component(repaired, graph, comp)

#         for k, atom_idx in enumerate(comp):
#             new_positions[atom_idx] = unwrapped[k]

#     repaired.positions[:] = new_positions

#     if centroid_position_frac is not None:
#         cell = repaired.cell.array
#         centroid_cart = np.mean(repaired.positions, axis=0)
#         centroid_frac = np.linalg.solve(cell.T, centroid_cart)
#         shift_frac = centroid_position_frac - centroid_frac
#         shift_cart = shift_frac @ cell
#         repaired.positions += shift_cart

#     return repaired

# def bond_graph_to_networkx(graph: dict) -> nx.Graph:
#     """
#     Convert a bond graph to a NetworkX graph.

#     Parameters
#     ----------
#     graph : dict
#         Bond graph where keys are atom indices and values are lists of (neighbor_index, shift) tuples.

#     Returns
#     -------
#     G : networkx.Graph
#         NetworkX graph representation of the bond graph.
#     """
#     G = nx.Graph()

#     for i, neighbors in graph.items():
#         i = int(i)

#         for j, S in neighbors:
#             j = int(j)

#             # graph contains each bond twice, but nx.Graph
#             # automatically collapses duplicates
#             G.add_edge(
#                 i,
#                 j,
#                 shift=np.asarray(S, dtype=int)
#             )

#     return G

# def find_rings(graph, max_size=None):
#     G = bond_graph_to_networkx(graph)

#     rings = list(
#         nx.chordless_cycles(
#             G,
#             length_bound=max_size
#         )
#     )

#     return rings

# from collections import defaultdict
# import numpy as np


# def n_neighbor_subgraph(
#     graph,
#     subset,
#     n=1,
# ):
#     """
#     Extract the induced subgraph containing a subset of atoms and all
#     atoms within N bond hops of that subset.

#     Parameters
#     ----------
#     graph : dict
#         Bond graph of the form

#             graph[i] = [(j, S_ij), ...]

#         where S_ij is the periodic lattice shift.

#     subset : iterable[int]
#         Starting atom indices.

#     n : int
#         Number of nearest-neighbor shells / graph hops to include.

#         n = 0:
#             only the starting subset

#         n = 1:
#             subset + directly bonded neighbors

#         n = 2:
#             subset + first + second neighbors

#         etc.

#     Returns
#     -------
#     subgraph : defaultdict(list)
#         Induced bond graph containing all selected atoms and all bonds
#         between them.

#     selected_atoms : set[int]
#         All atoms within <= N hops of the starting subset.

#     shells : list[set[int]]
#         shells[k] contains atoms exactly k hops from the starting subset.

#         shells[0] = starting subset
#         shells[1] = first neighbors
#         shells[2] = second neighbors
#         ...

#     """

#     if n < 0:
#         raise ValueError("n must be >= 0")

#     subset = set(map(int, subset))

#     # shells[k] = atoms exactly k bond hops from subset
#     shells = [subset.copy()]

#     selected_atoms = subset.copy()
#     frontier = subset.copy()

#     # Breadth-first search
#     for _ in range(n):
#         next_frontier = set()

#         for i in frontier:
#             for j, _ in graph.get(i, []):
#                 j = int(j)

#                 if j not in selected_atoms:
#                     next_frontier.add(j)

#         shells.append(next_frontier)

#         selected_atoms.update(next_frontier)
#         frontier = next_frontier

#         # Stop early if there are no more atoms to discover
#         if not frontier:
#             break

#     # Construct induced subgraph
#     subgraph = defaultdict(list)

#     for i in selected_atoms:
#         for j, S in graph.get(i, []):
#             j = int(j)

#             if j in selected_atoms:
#                 subgraph[i].append(
#                     (
#                         j,
#                         np.asarray(S, dtype=int).copy()
#                     )
#                 )

#     return subgraph, selected_atoms, shells

# def n_neighbor_indices(graph, subset, n=1):
#     """
#     Return all atom indices within <= n bond hops of subset.
#     """

#     if n < 0:
#         raise ValueError("n must be >= 0")

#     subset = set(map(int, subset))

#     selected = subset.copy()
#     frontier = subset.copy()

#     for _ in range(n):
#         next_frontier = {
#             int(j)
#             for i in frontier
#             for j, _ in graph.get(i, [])
#             if int(j) not in selected
#         }

#         selected.update(next_frontier)
#         frontier = next_frontier

#         if not frontier:
#             break

#     return selected

# def extract_neighborhood(graph, subset, n=1):
#     selected = n_neighbor_indices(graph, subset, n)

#     subgraph = defaultdict(list)

#     for i in selected:
#         for j, S in graph.get(i, []):
#             j = int(j)

#             if j in selected:
#                 subgraph[i].append(
#                     (
#                         j,
#                         np.asarray(S, dtype=int).copy()
#                     )
#                 )

#     return selected, subgraph

# def find_element_chains(
#     graph,
#     elements,
#     pattern,
#     unique_reverse=True,
# ):
#     """
#     Find all simple bonded paths whose element sequence matches `pattern`.

#     Parameters
#     ----------
#     graph : dict
#         Bond graph:
#             graph[i] = [(j, S_ij), ...]

#     elements : dict[int, str]
#         Mapping from atom index to element symbol.

#         Example:
#             {0: "C", 1: "H", 2: "O", ...}

#     pattern : sequence[str] | str
#         Element sequence to search for.

#         Examples:
#             ["H", "C", "C", "H"]
#             ["O", "C", "C", "C", "C", "O"]

#         A string such as "HCCH" also works for one-letter
#         element symbols.

#     unique_reverse : bool
#         If True, paths that differ only by direction are counted once.

#     Returns
#     -------
#     matches : list[tuple[int, ...]]
#         Atom-index paths matching the requested element sequence.
#     """

#     if isinstance(pattern, str):
#         pattern = list(pattern)

#     pattern = tuple(pattern)

#     if len(pattern) == 0:
#         return []

#     matches = []
#     seen = set()

#     def dfs(path, depth):
#         # Full pattern matched
#         if depth == len(pattern) - 1:
#             result = tuple(path)

#             if unique_reverse:
#                 canonical = min(result, result[::-1])

#                 if canonical in seen:
#                     return

#                 seen.add(canonical)

#             matches.append(result)
#             return

#         current = path[-1]
#         required_element = pattern[depth + 1]

#         for neighbor, _ in graph.get(current, []):
#             neighbor = int(neighbor)

#             # Require a simple path: don't reuse atoms
#             if neighbor in path:
#                 continue

#             # Element must match next position
#             if elements.get(neighbor) != required_element:
#                 continue

#             dfs(path + [neighbor], depth + 1)

#     # Start DFS from every atom matching the first element
#     for atom, element in elements.items():
#         atom = int(atom)

#         if element == pattern[0]:
#             dfs([atom], 0)

#     return matches

# from collections import defaultdict, deque
# import numpy as np


# def complete_molecule(graph, seed):
#     """
#     Find all atoms belonging to the connected component containing `seed`.

#     Parameters
#     ----------
#     graph : dict
#         Full bond graph:

#             graph[i] = [(j, S_ij), ...]

#     seed : int or iterable[int] or dict
#         Starting atom(s).

#         Can be:
#             17
#             [13, 15, 16]
#             {13, 15, 16}
#             subgraph

#         If a dict is supplied, its keys are used as the seed atoms.

#     Returns
#     -------
#     atoms : set[int]
#         All atom indices belonging to the connected component(s)
#         containing the seed atoms.

#     subgraph : defaultdict(list)
#         Induced bond graph for the completed molecule.
#     """

#     # Normalize seed input
#     if isinstance(seed, dict):
#         seeds = {int(i) for i in seed.keys()}

#     elif isinstance(seed, (int, np.integer)):
#         seeds = {int(seed)}

#     else:
#         seeds = {int(i) for i in seed}

#     if not seeds:
#         return set(), defaultdict(list)

#     # Breadth-first search
#     visited = set(seeds)
#     queue = deque(seeds)

#     while queue:
#         i = queue.popleft()

#         for j, _ in graph.get(i, []):
#             j = int(j)

#             if j not in visited:
#                 visited.add(j)
#                 queue.append(j)

#     # Construct induced subgraph
#     subgraph = defaultdict(list)

#     for i in visited:
#         for j, S in graph.get(i, []):
#             j = int(j)

#             if j in visited:
#                 subgraph[i].append(
#                     (
#                         j,
#                         np.asarray(S, dtype=int).copy(),
#                     )
#                 )

#     return visited, subgraph
