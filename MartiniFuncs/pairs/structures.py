"""Finite pair and pair+shell structures with source forces, and their Extended XYZ writer.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from ase import Atoms
from MartiniFuncs.structure import graph as mfg
from MartiniFuncs.structure.molecules import add_global_index


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
    from MartiniFuncs.io.lammps import LammpsFrame
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
