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

from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import first_neighbors, neighbor_list



AtomID: TypeAlias = int
Shift: TypeAlias = np.ndarray
BondGraph: TypeAlias = dict[AtomID, list[tuple[AtomID, Shift]]]

# id(ID array) -> (ID values, {ID: local row}); see get_index_maps.
_INDEX_MAP_CACHE: OrderedDict = OrderedDict()
# Active only inside frame_cache(); see unwrap_component.
_UNWRAP_MEMO: dict | None = None


@contextmanager
def frame_cache():
    """Memoize unwrap_component within one frame whose atoms and graph do not change.

    Inside the block, repeated unwraps of the same component (pair placement,
    geometry, shells, site COMs) are computed once. Leave the block before
    moving to another frame or mutating positions/graph: object ids can be
    reused, so the memo must never outlive the frame it was built for.
    """
    global _UNWRAP_MEMO
    previous, _UNWRAP_MEMO = _UNWRAP_MEMO, {}
    try:
        yield
    finally:
        _UNWRAP_MEMO = previous


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

    Results are cached per ID array (validated by content on every call), so
    repeated lookups on one frame no longer rebuild the dictionary. Treat the
    returned dictionary as read-only.
    """
    if index_array is not None and index_array in atoms.arrays:
        current = atoms.arrays[index_array]
        cached = _INDEX_MAP_CACHE.get(id(current))
        # The mapping depends only on the ID values, so a content match is
        # always a valid hit even if Python reused the array's id().
        if cached is not None and cached[0].shape == current.shape and np.array_equal(cached[0], current):
            _INDEX_MAP_CACHE.move_to_end(id(current))
            return cached[0].copy(), cached[1]

    local_to_id = ensure_index_array(atoms, index_array=index_array)

    id_to_local = {
        int(atom_id): int(local_idx)
        for local_idx, atom_id in enumerate(local_to_id)
    }
    if index_array is not None:
        _INDEX_MAP_CACHE[id(atoms.arrays[index_array])] = (local_to_id.copy(), id_to_local)
        while len(_INDEX_MAP_CACHE) > 32:
            _INDEX_MAP_CACHE.popitem(last=False)

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


def periodic_pairs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: Sequence[bool],
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """All ordered pairs (i, j, S) with |r_j + S @ cell - r_i| < cutoff, excluding (i, i, 0).

    Returns local indices i, j, integer image shifts S (n, 3) and distances d,
    sorted by (i, j, S). Both directions are present and periodic self-images
    (i == j, S != 0) are included, as in ``ase.neighborlist.neighbor_list``.

    Fully periodic nonsingular cells and fully nonperiodic systems use a SciPy
    KD-tree: positions are wrapped into the cell, ghost images are added only
    within ``cutoff`` of a face (several layers if the cutoff exceeds a face
    spacing), and shifts are mapped back to the input coordinates. Mixed
    periodicity falls back to ASE's ``neighbor_list``. Distances are recomputed
    from the input positions as |r_j + S @ cell - r_i|.
    """
    from scipy.spatial import cKDTree

    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    if cutoff <= 0 or len(positions) == 0:
        return (np.empty(0, int), np.empty(0, int), np.empty((0, 3), int), np.empty(0, float))

    if pbc.all() and abs(np.linalg.det(cell)) > 1e-12:
        fractional = np.linalg.solve(cell.T, positions.T).T
        images = np.floor(fractional).astype(int)
        inside = fractional - images
        volume = abs(np.linalg.det(cell))
        spacing = np.array([volume / np.linalg.norm(np.cross(cell[(k+1) % 3], cell[(k+2) % 3]))
                            for k in range(3)])
        margin = cutoff / spacing
        layers = np.ceil(margin).astype(int)
        ghost_rows, ghost_shifts = [], []
        for shift in np.array(np.meshgrid(*[np.arange(-n, n + 1) for n in layers], indexing="ij")).reshape(3, -1).T:
            moved = inside + shift
            keep = np.all((moved >= -margin) & (moved <= 1 + margin), axis=1)
            ghost_rows.append(np.flatnonzero(keep))
            ghost_shifts.append(np.repeat(shift[None, :], keep.sum(), axis=0))
        ghost_rows = np.concatenate(ghost_rows)
        ghost_shifts = np.concatenate(ghost_shifts)
        found = cKDTree(inside @ cell).sparse_distance_matrix(
            cKDTree((inside[ghost_rows] + ghost_shifts) @ cell), cutoff, output_type="ndarray")
        i = found["i"].astype(int)
        j = ghost_rows[found["j"]]
        # r_wrapped = r - images @ cell, so S = s + images_i - images_j in input coordinates.
        S = ghost_shifts[found["j"]] + images[i] - images[j]
    elif not pbc.any():
        found = cKDTree(positions).sparse_distance_matrix(cKDTree(positions), cutoff, output_type="ndarray")
        i, j = found["i"].astype(int), found["j"].astype(int)
        S = np.zeros((len(i), 3), dtype=int)
    else:
        search = Atoms(positions=positions, cell=cell, pbc=pbc)
        i, j, S = neighbor_list("ijS", search, cutoff=cutoff, self_interaction=False)
        S = np.asarray(S, dtype=int)

    distances = np.linalg.norm(positions[j] + S @ cell - positions[i], axis=1)
    keep = (distances < cutoff) & ~((i == j) & ~S.any(axis=1))
    i, j, S, distances = i[keep], j[keep], S[keep], distances[keep]
    order = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
    return i[order], j[order], S[order], distances[order]


def build_bond_graph(
    atoms: Atoms,
    scale: float = 1.3,
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

    A bond satisfies |r_j+S_ij @ cell-r_i| < scale*(rcov_i+rcov_j).
    Cell and positions are Angstrom, S_ij integer image shifts. skin is kept
    for API compatibility; it never changed the final bond criterion.
    The default scale, 1.3, is the package-wide bond criterion.
    Each adjacency list is sorted by (neighbor ID, shift), so traversal order
    (and hence component atom order) does not depend on geometry. Periodic
    self-image bonds (an atom bonded to its own image) are not stored.
    """
    numbers = atoms.get_atomic_numbers()
    local_to_id, _ = get_index_maps(atoms, index_array=index_array)
    radii = scale * covalent_radii[numbers]
    pbc = atoms.pbc if periodic else np.zeros(3, dtype=bool)
    i, j, S, d = periodic_pairs(atoms.positions, atoms.cell.array, pbc,
                                2 * float(radii.max()) if len(radii) else 0.)
    bonded = (j > i) & (d < radii[i] + radii[j])
    i, j, S = i[bonded], j[bonded], S[bonded]

    entries = defaultdict(list)
    for a, b, shift in zip(local_to_id[i].tolist(), local_to_id[j].tolist(), S):
        entries[a].append((b, tuple(shift.tolist())))
        entries[b].append((a, tuple((-shift).tolist())))
    graph: BondGraph = defaultdict(list)
    for node in sorted(entries):
        graph[node] = [(neighbor, np.array(shift, dtype=int)) for neighbor, shift in sorted(entries[node])]
    return graph


def induced_subgraph(
    graph: Mapping[int, Sequence[tuple[int, np.ndarray]]],
    ids: Iterable[int],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Bonds of ``graph`` whose both ends lie in ``ids`` (persistent atom IDs).

    For one molecule this equals rebuilding the bond graph on that molecule
    alone, without another neighbor search. Adjacency order is preserved.
    """
    ids = set(map(int, ids))
    return {i: [(j, s) for j, s in graph.get(i, []) if j in ids] for i in ids}


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
    if _UNWRAP_MEMO is not None:
        key = (id(atoms), id(graph), tuple(component), index_array, strict)
        if key not in _UNWRAP_MEMO:
            _UNWRAP_MEMO[key] = _unwrap_component(atoms, graph, component, index_array, strict)
        shifts, unwrapped = _UNWRAP_MEMO[key]
        return dict(shifts), unwrapped.copy()
    return _unwrap_component(atoms, graph, component, index_array, strict)


def _unwrap_component(atoms, graph, component, index_array, strict):
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
    bond_scale: float = 1.3,
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

    i, j, S, d = periodic_pairs(atoms.positions, atoms.cell.array, atoms.pbc, float(cutoff))

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
