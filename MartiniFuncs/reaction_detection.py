"""Possible reaction products inferred from current-frame connected components.

Atom count is a screening criterion, not proof of chemical connectivity or a
particular product. Short contacts can join components under a distance cutoff.
"""
import numpy as np
from . import graph


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


def find_possible_products(atoms, *, reactant_atom_counts, bond_scale=1.3):
    """Find components with size sum(reactant_atom_counts) in one current frame.

    For CPD C5H6 (11 atoms) and the current MAL C11H9NO2 (23 atoms), the target
    is 34. Supply counts for the actual molecules, including all hydrogens.
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
    lookup = {int(atom_id): i for i, atom_id in enumerate(ids)}
    bonded = graph.build_bond_graph(atoms, scale=bond_scale)
    hits = []
    for component in graph.connected_components(bonded, atom_ids=ids):
        if len(component) != sum(counts):
            continue
        atom_ids = sorted(map(int, component))
        local = [lookup[i] for i in atom_ids]
        hits.append(dict(atom_ids=atom_ids, local_indices=local, atom_count=len(local),
                         formula=atoms[local].get_chemical_formula()))
    return hits


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
