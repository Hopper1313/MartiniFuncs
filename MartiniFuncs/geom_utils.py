"""Geometry / reaction utilities for MartiniFuncs.

This module treats ``global_index`` as a persistent atom ID. Whenever IDs are
used to access an ASE ``Atoms`` object they are first converted to local ASE
indices via :mod:`MartiniFuncs.graph`.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from ase.neighborlist import primitive_neighbor_list

from . import graph as mfg


# -----------------------------------------------------------------------------
# Compatibility decorator / ID helpers
# -----------------------------------------------------------------------------


def add_global_index(func):
    """Backward-compatible decorator ensuring ``atoms.global_index`` exists."""
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        atoms = bound.arguments.get("atoms")
        if atoms is not None:
            mfg.ensure_index_array(atoms, "global_index")
        return func(*args, **kwargs)

    return wrapper


@add_global_index
def get_element_map(atoms: Atoms, index_array: str = "global_index") -> dict[int, str]:
    """Return ``persistent_atom_id -> element symbol``."""
    ids, _ = mfg.get_index_maps(atoms, index_array=index_array)
    return {
        int(atom_id): symbol
        for atom_id, symbol in zip(ids, atoms.get_chemical_symbols())
    }


def _subset(atoms: Atoms, atom_ids: Iterable[int], index_array="global_index") -> Atoms:
    """Slice ``atoms`` using persistent IDs rather than local ASE indices."""
    local = mfg.local_indices(atoms, atom_ids, index_array=index_array)
    return atoms[local]


def _weighted_com(xyz: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Return sum_i(masses[i]*xyz[i])/sum_i masses[i]; xyz is (N,3)
    in Angstrom and masses are positive atomic masses.
    """
    return np.average(np.asarray(xyz, float), axis=0, weights=np.asarray(masses, float))


def _site_center_of_mass(
    atoms: Atoms,
    graph,
    component: Sequence[int],
    site_ids: Iterable[int],
    index_array: str = "global_index",
) -> np.ndarray:
    """COM of a subset of atoms in the *unwrapped frame of its molecule*.

    Unwrap the component with lattice shifts n_i, then compute
    sum_{i in site_ids} m_i*(r_i+n_i @ cell)/sum_{i in site_ids} m_i.
    """
    site_ids = [int(i) for i in site_ids]
    shifts, _ = mfg.unwrap_component(
        atoms,
        graph,
        component,
        index_array=index_array,
    )
    local = mfg.local_indices(atoms, site_ids, index_array=index_array)
    shift_array = np.vstack([shifts[i] for i in site_ids])
    xyz = atoms.positions[local] + shift_array @ atoms.cell.array
    return _weighted_com(xyz, atoms.get_masses()[local])


def _wrap_points(points: np.ndarray, cell, pbc) -> tuple[np.ndarray, np.ndarray]:
    """Wrap arbitrary Cartesian points and return removed integer cell shifts.

    Returns ``wrapped, image`` with ``points = wrapped + image @ cell``.
    Only periodic dimensions are wrapped.

    For row-cell H, f=points @ inverse(H); image=floor(f) only on
    periodic axes and zero elsewhere. wrapped=points-image @ H.
    """
    points = np.asarray(points, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)

    frac = np.linalg.solve(cell.T, points.T).T
    image = np.zeros_like(frac, dtype=int)
    image[:, pbc] = np.floor(frac[:, pbc]).astype(int)
    wrapped = points - image @ cell
    return wrapped, image


# -----------------------------------------------------------------------------
# Reaction-site identification
# -----------------------------------------------------------------------------


@add_global_index
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

    if len(molecules[0]) == 11 and len(molecules[1]) == 23:
        cpd_index, n4t_index = 0, 1
    elif len(molecules[1]) == 11 and len(molecules[0]) == 23:
        cpd_index, n4t_index = 1, 0
    else:
        raise ValueError(
            f"Unexpected molecule sizes, got {[len(m) for m in molecules]}, "
            "but expected 11 and 23."
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


def get_mal_CC(bond_graph, elements):
    """Return the maleimide C=C atoms from element-pattern intersections."""
    hcch = mfg.find_element_chains(bond_graph, elements, "HCCH")
    occcco = mfg.find_element_chains(bond_graph, elements, "OCCCCO")
    hcch_atoms = {i for chain in hcch for i in chain}
    occcco_atoms = {i for chain in occcco for i in chain}
    return hcch_atoms & occcco_atoms


# -----------------------------------------------------------------------------
# Molecular pair search
# -----------------------------------------------------------------------------


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
        centers.append(_site_center_of_mass(atoms, graph, comp, mal_cc))
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

    cpd = [c for c in components if len(c) == 11]
    mal = [c for c in components if len(c) == 23]
    product = [c for c in components if len(c) == 34]
    other = [c for c in components if len(c) not in (11, 23, 34)]
    return graph, components, cpd, mal, product, other


@add_global_index
def get_all_prereactive_pairs(
    atoms: Atoms,
    scale: float = 1.3,
    reactive_radius: float = 3.25,
    printing: bool = False,
    *,
    criteria=None,
):
    """Find all CPD / dienophile molecular pairs inside ``reactive_radius``.

    Returns the same API as the historical implementation:

    ``pairs``
        ``[[cpd_atom_ids, mal_atom_ids], ...]`` using persistent atom IDs.

    ``cpd_shift``
        Integer lattice translation to apply to the *unwrapped CPD molecule*
        so it lies beside the corresponding unwrapped MAL molecule.

    Unlike the previous implementation, molecular and reactive-site centroids
    are calculated in unwrapped molecular coordinate frames, so the function
    does not require ``repair_fragmented_molecules`` first.

    ``criteria`` optionally accepts PairCriteria. The default preserves the
    distance screen; use assess_prereactive_pairs separately to retain reasons
    and geometry measurements for rejected candidates.

    Screen d=min_n |R_CPD+n @ cell-R_CC| < reactive_radius, where
    R_CPD is the mass-weighted COM of all CPD atoms and R_CC is the
    mass-weighted COM of the dienophile reactive C=C. Both are computed from
    whole molecules. Keep the nearest periodic image; returned shift is
    S+mal_image-cpd_image. This is not a forming-bond distance criterion.
    """
    if not np.isfinite(reactive_radius) or reactive_radius <= 0:
        raise ValueError("reactive_radius must be > 0")

    elements = get_element_map(atoms, index_array="global_index")
    graph, _, cpd_components, mal_components, _, other = _reaction_components(
        atoms, scale
    )

    if other:
        raise ValueError(
            f"There are {len(other)} molecules that do not match expected sizes "
            "(11, 23, 34 atoms). Check bond scale / chemistry."
        )

    # CPD search point = COM of the entire unwrapped CPD molecule.
    cpd_centers = np.asarray([
        mfg.component_center_of_mass(atoms, graph, comp)
        for comp in cpd_components
    ])

    # MAL search point = COM of the reactive C=C in the unwrapped MAL frame.
    mal_centers = _mal_site_centers(atoms, graph, mal_components, elements)

    if printing:
        print(len(mal_centers), "mal CC double bonds found")
        print(len(cpd_centers), "CPD molecules found")

    if len(cpd_centers) == 0 or len(mal_centers) == 0:
        return [], []

    pairs, cpd_shift = _pairs_from_centers(
        cpd_centers, mal_centers, cpd_components, mal_components,
        atoms.cell.array, atoms.pbc, reactive_radius)

    if criteria is not None:
        from .pair_selection import assess_prereactive_pairs
        assessments = assess_prereactive_pairs(atoms, pairs, cpd_shift,
                                               criteria=criteria, scale=scale)
        keep = [a.pair_index for a in assessments if a.accepted]
        pairs, cpd_shift = [pairs[i] for i in keep], [cpd_shift[i] for i in keep]

    if printing:
        print(len(pairs), "pre-reactive pairs found")

    return pairs, cpd_shift


def _pairs_from_centers(cpd_centers, mal_centers, cpd_components, mal_components,
                        cell, pbc, reactive_radius):
    """Shared center-distance screen, preserving image choice and candidate order.

    Select |C_CPD + shift @ cell - C_CC| < reactive_radius (Angstrom).
    Wrapped neighbor shift S becomes shift=S+mal_image-cpd_image for whole
    molecules. Sort by distance, then MAL and CPD component indices.
    """
    if not len(cpd_centers) or not len(mal_centers):
        return [], []
    # Put both point sets into one primary periodic representation while
    # remembering which image was removed from each unwrapped centroid.
    mal_wrapped, mal_image = _wrap_points(mal_centers, cell, pbc)
    cpd_wrapped, cpd_image = _wrap_points(cpd_centers, cell, pbc)

    nmal = len(mal_wrapped)
    ncpd = len(cpd_wrapped)
    positions = np.vstack((mal_wrapped, cpd_wrapped))

    # Dummy species labels let ASE discard MAL-MAL / CPD-CPD pairs at cutoff
    # evaluation time. Only 1-2 contacts are relevant.
    numbers = np.concatenate((
        np.ones(nmal, dtype=int),
        np.full(ncpd, 2, dtype=int),
    ))

    i, j, d, S = primitive_neighbor_list(
        "ijdS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff={(1, 2): float(reactive_radius)},
        numbers=numbers,
        self_interaction=False,
    )

    mask = (i < nmal) & (j >= nmal)
    mal_i = i[mask]
    cpd_i = j[mask] - nmal
    distances = d[mask]
    relative_shifts = S[mask]

    if len(mal_i) == 0:
        return [], []

    # Multiple periodic images of the same molecular pair can in principle be
    # inside a large cutoff. Retain only the closest image for each pair.
    best: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for m, c, dist, shift in zip(mal_i, cpd_i, distances, relative_shifts):
        key = (int(m), int(c))
        prev = best.get(key)
        if prev is None or float(dist) < prev[0]:
            # Unwrapped centroids satisfy:
            #   C_unwrapped = C_wrapped + cpd_image @ cell
            # The translation applied to the unwrapped CPD is therefore:
            #   T = S + mal_image - cpd_image
            pair_shift = (
                np.asarray(shift, dtype=int)
                + mal_image[int(m)]
                - cpd_image[int(c)]
            )
            best[key] = (float(dist), pair_shift)

    # Deterministic order: distance, then MAL/CPD component indices.
    ordered = sorted(best.items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1]))

    pairs = [
        [list(cpd_components[c]), list(mal_components[m])]
        for (m, c), _ in ordered
    ]
    cpd_shift = [value[1].tolist() for _, value in ordered]

    return pairs, cpd_shift


# -----------------------------------------------------------------------------
# Pair -> atom shell -> completed shell molecules -> finite local cluster
# -----------------------------------------------------------------------------


@add_global_index
def get_pair_shell_clusters(
    atoms: Atoms,
    pairs,
    shifts,
    *,
    shell_radius: float = 3.0,
    scale: float = 1.3,
    graph=None,
    components=None,
):
    """Build complete finite local clusters around pre-reactive molecular pairs.

    The geometric shell is defined as all atoms within ``shell_radius`` of any
    atom in the placed reactive pair. Any molecule touched by that geometric
    shell is then completed and unwrapped using the existing bond graph.

    Parameters
    ----------
    atoms
        Full periodic frame.
    pairs, shifts
        Output of :func:`get_all_prereactive_pairs`.
    shell_radius
        Atom-to-pair shell cutoff in Angstrom.
    scale
        Bond-graph scale, used only if ``graph`` is not supplied.
    graph, components
        Optional precomputed molecular graph/components for reuse.

    Returns
    -------
    list[dict]
        One record per molecular pair. ``cluster_xyz`` and ``symbols`` can be
        written directly as a finite XYZ structure.

    Select contacts with min_{i in pair}|r_j+Q_j @ cell-(r_i+Q_i @ cell)|
    < shell_radius, then retain each touched molecule in full. A shell molecule's
    closest contact sets its image; bond shifts place its remaining atoms.
    Thus completed molecule atoms can lie outside the geometric cutoff.
    """
    if len(pairs) != len(shifts):
        raise ValueError("pairs and shifts must have the same length")

    if graph is None:
        graph = mfg.build_bond_graph(
            atoms,
            scale=scale,
            index_array="global_index",
        )
    if components is None:
        components = mfg.connected_components(
            graph,
            atom_ids=mfg.atom_ids(atoms),
        )

    atom_to_component = mfg.component_lookup(components)
    shell_index = mfg.build_shell_neighbor_index(
        atoms,
        cutoff=shell_radius,
        index_array="global_index",
    )

    symbols_all = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    clusters = []

    for pair_number, ((cpd_ids, mal_ids), cpd_image_shift) in enumerate(
        zip(pairs, shifts)
    ):
        cpd_ids = [int(i) for i in cpd_ids]
        mal_ids = [int(i) for i in mal_ids]
        cpd_image_shift = np.asarray(cpd_image_shift, dtype=int)

        # MAL is the reference image. CPD receives the molecular shift returned
        # by get_all_prereactive_pairs(). Each molecule is unwrapped internally.
        cpd_arr, cpd_xyz, cpd_shifts = mfg.place_component(
            atoms,
            graph,
            cpd_ids,
            image_shift=cpd_image_shift,
        )
        mal_arr, mal_xyz, mal_shifts = mfg.place_component(
            atoms,
            graph,
            mal_ids,
            image_shift=(0, 0, 0),
        )

        pair_shift_map = {**mal_shifts, **cpd_shifts}
        pair_ids = np.concatenate((cpd_arr, mal_arr))
        pair_xyz = np.vstack((cpd_xyz, mal_xyz))

        contacts = mfg.get_shell_contacts(
            pair_shift_map,
            shell_index,
            exclude_pair=True,
        )

        shell_components = mfg.place_shell_components(
            atoms,
            graph,
            components,
            atom_to_component,
            contacts,
            exclude_atom_ids=pair_ids,
            index_array="global_index",
        )

        id_chunks = [cpd_arr, mal_arr]
        xyz_chunks = [cpd_xyz, mal_xyz]
        cluster_shift_map = {**cpd_shifts, **mal_shifts}

        for shell in shell_components:
            id_chunks.append(np.asarray(shell["indices"], dtype=int))
            xyz_chunks.append(np.asarray(shell["xyz"], dtype=float))
            cluster_shift_map.update(shell["shifts"])

        cluster_ids = np.concatenate(id_chunks)
        cluster_xyz = np.vstack(xyz_chunks)
        cluster_local = mfg.local_indices(atoms, cluster_ids)
        cluster_symbols = symbols_all[cluster_local]
        cluster_shift_array = np.vstack([
            cluster_shift_map[int(atom_id)]
            for atom_id in cluster_ids
        ])

        # Local indices refer to rows in the extracted finite cluster/XYZ.
        # The cluster is deliberately assembled as CPD + MAL + shell molecules,
        # so the original reactive pair always occupies the first rows.
        cpd_local_indices = np.arange(len(cpd_arr), dtype=int)
        mal_local_indices = np.arange(
            len(cpd_arr),
            len(cpd_arr) + len(mal_arr),
            dtype=int,
        )
        cluster_local_indices = np.arange(len(cluster_ids), dtype=int)

        clusters.append({
            "pair_index": pair_number,
            "cpd_ids": cpd_arr,
            "mal_ids": mal_arr,
            "cpd_local_indices": cpd_local_indices,
            "mal_local_indices": mal_local_indices,
            "cpd_shift": cpd_image_shift.copy(),
            "pair_ids": pair_ids,
            "pair_local_indices": np.concatenate((
                cpd_local_indices,
                mal_local_indices,
            )),
            "pair_xyz": pair_xyz,
            "pair_shifts": pair_shift_map,
            "shell_contacts": contacts,
            "shell_components": shell_components,
            "cluster_ids": cluster_ids,
            # Explicit cluster identity metadata. Local indices are row
            # numbers in cluster_xyz / the written XYZ; global IDs are the
            # persistent IDs from the source Atoms object.
            "cluster_local_indices": cluster_local_indices,
            "cluster_global_ids": cluster_ids.copy(),
            "cluster_xyz": cluster_xyz,
            "cluster_shifts": cluster_shift_array,
            "symbols": cluster_symbols,
        })

    return clusters


# -----------------------------------------------------------------------------
# Positions / velocities / forces for pair or cluster output
# -----------------------------------------------------------------------------


def _raw_vector(raw_atoms: Mapping[int, Mapping[str, float]], atom_ids, cols):
    values = []
    for atom_id in atom_ids:
        atom_id = int(atom_id)
        if atom_id not in raw_atoms:
            raise KeyError(
                f"Persistent atom ID {atom_id} is absent from raw LAMMPS data."
            )
        record = raw_atoms[atom_id]
        missing = [col for col in cols if col not in record]
        if missing:
            raise KeyError(
                f"LAMMPS atom {atom_id} is missing columns {missing}."
            )
        values.append(tuple(record[col] for col in cols))
    return np.asarray(values, dtype=float)


def _source_vectors(atoms, source, ids):
    """Join synchronized normalized frames, or historical raw tuples, by ID."""
    from .lammpsio import LammpsFrame
    if isinstance(source, LammpsFrame):
        if atoms.info.get("timestep") != source.timestep:
            raise ValueError("Geometry and force source have different timesteps")
        local = mfg.local_indices(source.atoms, ids)
        if "lammps_forces" not in source.atoms.arrays:
            raise ValueError("Frame has no force columns")
        result = {"forces": source.atoms.arrays["lammps_forces"][local],
                  "force_units": "eV/Angstrom", "velocity_units": "ASE",
                  "lammps_units": source.units, "timestep": source.timestep,
                  "frame_index": source.index,
                  "source": source.atoms.info["source"],
                  "id_offset": source.id_offset,
                  "source_cell": source.atoms.cell.array.copy(),
                  "source_pbc": source.atoms.pbc.copy()}
        if "momenta" in source.atoms.arrays:
            result["velocities"] = source.atoms.get_velocities()[local]
        return result
    if "timestep" in atoms.info and atoms.info["timestep"] != source[0]:
        raise ValueError("Geometry and raw force data have different timesteps; specify the same frame index")
    raw = source[-1]
    result = {"forces": _raw_vector(raw, ids, ("fx", "fy", "fz")),
              "force_units": "source", "velocity_units": "source", "timestep": source[0]}
    if all(all(k in raw[int(i)] for k in ("vx", "vy", "vz")) for i in ids):
        result["velocities"] = _raw_vector(raw, ids, ("vx", "vy", "vz"))
    return result


@add_global_index
def get_pair_positions_and_forces(
    atoms: Atoms,
    mylammpsread,
    pairs,
    shifts,
    *,
    bond_scale: float = 1.3,
    graph=None,
):
    """Backward-compatible pair extraction using persistent IDs safely.

    Placed coordinates obey r'_i=r_i+(n_i+shift) @ cell for CPD and
    r'_i=r_i+n_i @ cell for the dienophile, using graph bond shifts n_i.
    Source vectors are selected by persistent IDs, without recalculation or rotation.
    An optional graph must belong to this frame; it avoids rebuilding connectivity.
    """
    if len(pairs) != len(shifts):
        raise ValueError("pairs and shifts must have the same length")
    if graph is None:
        graph = mfg.build_bond_graph(atoms, scale=bond_scale)
    data = {}

    for k, ((cpd_ids, mal_ids), cpd_shift) in enumerate(zip(pairs, shifts)):
        cpd_arr, cpd_xyz, _ = mfg.place_component(
            atoms, graph, cpd_ids, image_shift=cpd_shift
        )
        mal_arr, mal_xyz, _ = mfg.place_component(
            atoms, graph, mal_ids, image_shift=(0, 0, 0)
        )
        ids = np.concatenate((cpd_arr, mal_arr))
        xyz = np.vstack((cpd_xyz, mal_xyz))
        local = mfg.local_indices(atoms, ids)

        cpd_local_indices = np.arange(len(cpd_arr), dtype=int)
        mal_local_indices = np.arange(
            len(cpd_arr),
            len(cpd_arr) + len(mal_arr),
            dtype=int,
        )

        data[k] = {
            "atom_ids": ids,
            "positions": xyz,
            **_source_vectors(atoms, mylammpsread, ids),
            "symbols": np.asarray(atoms.get_chemical_symbols(), dtype=object)[local],
            "cpd_local_indices": cpd_local_indices,
            "mal_local_indices": mal_local_indices,
            "cpd_global_ids": np.asarray(cpd_arr, dtype=int),
            "mal_global_ids": np.asarray(mal_arr, dtype=int),
            # For pair-only output, the extracted pair is the whole cluster.
            "cluster_local_indices": np.arange(len(ids), dtype=int),
            "cluster_global_ids": np.asarray(ids, dtype=int),
        }

    return data


def get_cluster_positions_and_forces(atoms: Atoms, mylammpsread, clusters):
    """Attach LAMMPS forces/velocities to records from get_pair_shell_clusters.

    Positions already include lattice translations r'=r+Q @ cell. Source
    forces/velocities are looked up by persistent ID in the same output order;
    translations do not change vector components.
    """
    data = {}

    for k, cluster in enumerate(clusters):
        ids = np.asarray(cluster["cluster_ids"], dtype=int)
        data[k] = {
            "atom_ids": ids,
            "positions": np.asarray(cluster["cluster_xyz"], dtype=float),
            **_source_vectors(atoms, mylammpsread, ids),
            "symbols": np.asarray(cluster["symbols"], dtype=object),
            # Preserve the identity of the original reactive pair all the way
            # to XYZ output. "local" means row index in this extracted cluster.
            "cpd_local_indices": np.asarray(
                cluster["cpd_local_indices"], dtype=int
            ),
            "mal_local_indices": np.asarray(
                cluster["mal_local_indices"], dtype=int
            ),
            "cpd_global_ids": np.asarray(cluster["cpd_ids"], dtype=int),
            "mal_global_ids": np.asarray(cluster["mal_ids"], dtype=int),
            "cluster_local_indices": np.asarray(
                cluster["cluster_local_indices"], dtype=int
            ),
            "cluster_global_ids": np.asarray(
                cluster["cluster_global_ids"], dtype=int
            ),
        }

    return data

def pair_data_to_extxyz(data, filename):
    """Write pair/cluster records as an ASE Extended XYZ trajectory.

    Each extracted cluster is converted to an :class:`ase.Atoms` object and
    written with ``format="extxyz"``.  The atom identity metadata is stored as
    per-atom arrays, so it survives an ``ase.io.read`` round trip without
    parsing the XYZ comment line.

    Per-atom arrays written
    -----------------------
    ``global_index``
        Persistent atom ID from the original periodic structure.

    ``local_index``
        Zero-based row/index of the atom in this extracted cluster.

    ``frame_local_index`` / ``lammps_id`` (when provided by the record)
        Zero-based row in the source ASE frame / original LAMMPS atom ID.

    ``pair_role``
        ``"CPD"`` for the CPD reactant, ``"MAL"`` for the dienophile, and
        ``"shell"`` for atoms belonging only to the surrounding shell.

    ``lammps_forces`` / ``velocities``
        Three-component vectors from the record's forces/velocities fields.

    Notes
    -----
    The filename may still end in ``.xyz``.  ``format="extxyz"`` explicitly
    selects ASE's Extended XYZ writer.

    No numerical unit conversion occurs: lammps_forces and velocities are
    copied verbatim with their record unit metadata. A LammpsFrame-derived
    record has eV/Angstrom forces and ASE internal velocities; legacy tuples
    retain raw source units. The stored velocities array is not ASE momenta.
    """
    from contextlib import nullcontext
    from ase.io.extxyz import output_column_format

    images = []
    nclusters = len(data)

    for i in range(nclusters):
        record = data[i]

        positions = np.asarray(record["positions"], dtype=float)
        symbols = np.asarray(record["symbols"], dtype=object)
        natoms = len(positions)

        if positions.shape != (natoms, 3):
            raise ValueError(
                f"Cluster {i}: positions must have shape (N, 3); "
                f"got {positions.shape}."
            )
        if len(symbols) != natoms:
            raise ValueError(
                f"Cluster {i}: symbols has length {len(symbols)}, "
                f"but positions has length {natoms}."
            )

        cluster = Atoms(
            symbols=symbols.tolist(),
            positions=positions,
            pbc=False,
        )

        # ------------------------------------------------------------------
        # Persistent and local identity
        # ------------------------------------------------------------------
        global_ids = np.asarray(
            record.get("cluster_global_ids", record.get("atom_ids", [])),
            dtype=int,
        )
        local_ids = np.asarray(
            record.get("cluster_local_indices", np.arange(natoms, dtype=int)),
            dtype=int,
        )

        if global_ids.shape != (natoms,):
            raise ValueError(
                f"Cluster {i}: cluster_global_ids/atom_ids must contain "
                f"one ID per atom; got {len(global_ids)} IDs for {natoms} atoms."
            )
        if local_ids.shape != (natoms,):
            raise ValueError(
                f"Cluster {i}: cluster_local_indices must contain one index "
                f"per atom; got {len(local_ids)} for {natoms} atoms."
            )

        cluster.new_array("global_index", global_ids.copy())
        cluster.new_array("local_index", local_ids.copy())
        for name in ("frame_local_index", "lammps_id"):
            if name in record:
                values = np.asarray(record[name], dtype=int)
                if values.shape != (natoms,):
                    raise ValueError(f"{name} must contain one index per exported atom")
                cluster.new_array(name, values.copy())

        # ------------------------------------------------------------------
        # Original reactive-pair membership
        # ------------------------------------------------------------------
        cpd_local = np.asarray(
            record.get("cpd_local_indices", []),
            dtype=int,
        )
        mal_local = np.asarray(
            record.get("mal_local_indices", []),
            dtype=int,
        )

        for name, idx in (("CPD", cpd_local), ("MAL", mal_local)):
            if np.any((idx < 0) | (idx >= natoms)):
                raise IndexError(
                    f"Cluster {i}: {name}_local_indices contains an index "
                    f"outside [0, {natoms})."
                )

        if np.intersect1d(cpd_local, mal_local).size:
            raise ValueError(
                f"Cluster {i}: CPD and MAL local-index sets overlap."
            )

        pair_role = np.full(natoms, "shell", dtype="U5")
        pair_role[cpd_local] = "CPD"
        pair_role[mal_local] = "MAL"
        cluster.new_array("pair_role", pair_role)

        # ------------------------------------------------------------------
        # Existing dynamical data
        # ------------------------------------------------------------------
        if "forces" in record:
            forces = np.asarray(record["forces"], dtype=float)
            if forces.shape != (natoms, 3):
                raise ValueError(
                    f"Cluster {i}: forces must have shape ({natoms}, 3); "
                    f"got {forces.shape}."
                )
            cluster.new_array("lammps_forces", forces.copy())

        if "velocities" in record:
            velocities = np.asarray(record["velocities"], dtype=float)
            if velocities.shape != (natoms, 3):
                raise ValueError(
                    f"Cluster {i}: velocities must have shape ({natoms}, 3); "
                    f"got {velocities.shape}."
                )
            cluster.new_array("velocities", velocities.copy())

        # Small structure-level metadata belongs in Atoms.info.  The long
        # atom mappings stay in per-atom arrays above rather than bloating the
        # Extended XYZ header/comment line.
        cluster.info["cluster_index"] = int(i)
        cluster.info["n_clusters"] = int(nclusters)
        cluster.info["n_cpd"] = int(len(cpd_local))
        cluster.info["n_mal"] = int(len(mal_local))
        cluster.info["n_shell"] = int(natoms - len(cpd_local) - len(mal_local))

        for key in ("force_units", "velocity_units", "lammps_units", "timestep",
                    "frame_index", "source", "id_offset", "source_cell", "source_pbc",
                    "structure_index", "candidate_index", "time_ps"):
            if key in record:
                cluster.info[key] = record[key]
        images.append(cluster)

    # ASE's default %16.8f truncates small MD force components. Retain its
    # Extended XYZ header/schema but serialize floating values with 17 digits.
    context = nullcontext(filename) if hasattr(filename, "write") else open(filename, "w", encoding="utf-8")
    with context as stream:
        for cluster in images:
            arrays = {"symbols": np.asarray(cluster.get_chemical_symbols()),
                      "positions": cluster.positions,
                      **{k: v for k, v in cluster.arrays.items() if k not in {"numbers", "positions"}}}
            columns = list(arrays)
            comment, _, _, _ = output_column_format(cluster, columns, arrays)
            stream.write(f"{len(cluster)}\n{comment}\n")
            # One %-format per row ("%.17g" % x == format(x, ".17g"); "%s" == str).
            blocks, tokens = [], []
            for name in columns:
                array = np.asarray(arrays[name])
                block = array.reshape(len(cluster), -1)
                tokens += ["%.17g" if array.dtype.kind in "fd" else "%s"] * block.shape[1]
                blocks.append(block.tolist())
            line = " ".join(tokens) + "\n"
            stream.write("".join(line % tuple(v for part in row for v in part) for row in zip(*blocks)))



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

            cpd_local = data[i].get("cpd_local_indices", [])
            mal_local = data[i].get("mal_local_indices", [])
            cpd_global = data[i].get("cpd_global_ids", [])
            mal_global = data[i].get("mal_global_ids", [])
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


# -----------------------------------------------------------------------------
# Generic geometry utilities
# -----------------------------------------------------------------------------


def align_atoms(mobile, reference, indices=None):
    """Align finite clusters by row correspondence; rotate stored vectors too.

    Returns the historical ``(aligned, rmsd)`` tuple. For persistent-ID matching
    and access to the rotation/translation, use ``geometry.align_geometry``.

    Equal-weight Kabsch fit: minimize sum_i |r_mobile,i @ R+t-r_reference,i|^2
    with det(R)=+1. Return the aligned copy and sqrt(mean squared distance);
    matching is by row, as delegated to geometry.align_geometry(id_array=None).
    """
    from .geometry import align_geometry

    aligned, rmsd, _ = align_geometry(mobile, reference, indices, id_array=None)
    return aligned, rmsd


@add_global_index
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
