"""Possible reaction products inferred from current-frame connected components.

Atom count is a screening criterion, not proof of chemical connectivity or a
particular product. Short contacts can join components under a distance cutoff.
"""
import numpy as np
from MartiniFuncs.structure import graph


class PossibleReactionError(ValueError):
    """Stop fixed-membership analysis; candidates contains inspectable evidence."""
    def __init__(self, candidates):
        self.candidates = candidates
        first = candidates[0]
        super().__init__(f"Possible reaction at source frame {first['frame']} "
                         f"(timestep {first['timestep']}): connected component has "
                         f"{first['atom_count']} atoms. Fixed-species analysis stopped. "
                         "Inspect exception.candidates and scan_possible_products; "
                         "atom count alone does not confirm a reaction.")

    def __reduce__(self):
        """Preserve candidate evidence when propagated from a frame worker."""
        return type(self), (self.candidates,)


def _bonded_candidates(atoms, cutoff):
    """Atom pairs (i, j, minimum-image distance) closer than cutoff, for connectivity only.

    Orthorhombic periodic cells with cutoff below half the shortest side use one
    periodic KD-tree (no ghost images, shifts or sorting): connectivity needs only
    whether some image is bonded, and the minimum image decides that. Other cells
    use graph.periodic_pairs. Both give the same connected components.
    """
    from scipy.spatial import cKDTree
    cell = atoms.cell.array
    lengths = np.diagonal(cell)
    if (atoms.pbc.all() and np.array_equal(cell, np.diag(lengths)) and np.all(lengths > 0)
            and cutoff < lengths.min() / 2):
        inside = np.mod(atoms.positions - np.asarray(atoms.get_celldisp()).ravel(), lengths)
        inside[inside >= lengths] = 0.0   # mod can return L for tiny negative values
        pairs = cKDTree(inside, boxsize=lengths).query_pairs(cutoff, output_type="ndarray")
        delta = inside[pairs[:, 1]] - inside[pairs[:, 0]]
        delta -= lengths * np.rint(delta / lengths)
        return pairs[:, 0], pairs[:, 1], np.linalg.norm(delta, axis=1)
    i, j, _, d = graph.periodic_pairs(atoms.positions, cell, atoms.pbc, cutoff)
    return i, j, d


def _bond_components(atoms, bond_scale):
    """Bonded rows (i < j) and connected components of the current frame.

    The bond rule is graph.build_bond_graph's, d < bond_scale*(r_cov_i+r_cov_j),
    without building the ID-keyed adjacency dictionary: only component labels
    are needed. Returns (i, j, component count, label per atom row).
    """
    from ase.data import covalent_radii
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    radii = bond_scale * covalent_radii[atoms.numbers]
    i, j, d = _bonded_candidates(atoms, 2*float(radii.max()))
    bonded = (j > i) & (d < radii[i] + radii[j])
    i, j = i[bonded], j[bonded]
    count, labels = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(atoms), len(atoms))),
                                         directed=False)
    return i, j, count, labels


def topology_change(atoms, groups, molecule_sizes, *, bond_scale=1.3):
    """Components of the current bond graph that are not one of the reference molecules.

    groups[row] is each atom's reference-molecule index and molecule_sizes its
    atom count. With the same bond rule as graph.build_bond_graph, the frame has
    the reference topology iff no bond joins two reference molecules and every
    molecule is one component. Otherwise return the changed components (merged
    or split pieces) as a sorted tuple of sorted atom-ID tuples; () if unchanged.
    The fast check costs one neighbor search; details only when something changed.
    """
    i, j, count, labels = _bond_components(atoms, bond_scale)
    if count == len(molecule_sizes) and not np.any(groups[i] != groups[j]):
        return ()
    ids = graph.ensure_index_array(atoms)
    changed = []
    for label in range(count):
        rows = np.flatnonzero(labels == label)
        members = np.unique(groups[rows])
        if len(members) > 1 or len(rows) != molecule_sizes[members[0]]:
            changed.append(tuple(sorted(int(x) for x in ids[rows])))
    return tuple(sorted(changed))


def find_possible_products(atoms, *, reactant_atom_counts, bond_scale=1.3):
    """Find components with size sum(reactant_atom_counts) in one current frame.

    The target is the sum of the two reactants' atom counts (n_A + n_B for an
    A + B adduct). Supply counts for the actual molecules, including all
    hydrogens.
    Rebuild the periodic bond graph using distance <= bond_scale*(r_cov_i+r_cov_j)
    in Angstrom, then examine connected components. Do not use LAMMPS molecule
    IDs or the initial topology: those may stay unchanged after a reaction.

    Return atom IDs, current-frame local indices, atom count and formula. The
    formula is evidence, not an additional filter. Larger aggregates and other
    reactions are outside this exact-size screen; persistence is not required.
    """
    counts = tuple(reactant_atom_counts)
    if len(counts) != 2 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0 for n in counts):
        raise ValueError("Supply two positive integer reactant atom counts")
    if not np.isfinite(bond_scale) or bond_scale <= 0:
        raise ValueError("bond_scale must be positive and finite")
    ids = graph.ensure_index_array(atoms)
    _, _, count, labels = _bond_components(atoms, bond_scale)
    sizes = np.bincount(labels, minlength=count)
    hits = []
    for label in np.flatnonzero(sizes == sum(counts)):
        rows = np.flatnonzero(labels == label)
        order = np.argsort(ids[rows])
        local = rows[order].tolist()
        hits.append(dict(atom_ids=[int(x) for x in ids[rows][order]], local_indices=local, atom_count=len(local),
                         formula=atoms[local].get_chemical_formula()))
    return sorted(hits, key=lambda hit: hit["atom_ids"][0])


def scan_possible_products(frames, *, reactant_atom_counts, bond_scale=1.3):
    """Scan an iterable of LammpsFrame objects and retain every candidate occurrence.

    Each record adds source frame index and LAMMPS timestep. Identical atom-ID
    sets in consecutive records allow inspection of persistence/reversibility.
    Every supplied frame is checked; use an unstrided iter_lammps_frames iterator
    to include events between analysis samples. A product already in frame zero
    is flagged, but its formation time cannot be inferred from this trajectory.
    """
    records = []
    for frame in frames:
        records.extend(dict(frame=int(frame.index), timestep=int(frame.timestep), **hit)
                       for hit in find_possible_products(frame.atoms,
                           reactant_atom_counts=reactant_atom_counts, bond_scale=bond_scale))
    return records
