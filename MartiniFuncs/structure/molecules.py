"""Small persistent-ID helpers: element maps and ID subsets.

Component and site centers of mass are structure.graph.component_center_of_mass;
point wrapping is structure.periodic.wrap_points.
"""
from __future__ import annotations
from collections.abc import Iterable
from ase import Atoms
from MartiniFuncs.structure import graph as mfg


def get_element_map(atoms: Atoms, index_array: str = "global_index") -> dict[int, str]:
    """Return ``persistent_atom_id -> element symbol`` (creates the ID array if absent)."""
    ids, _ = mfg.get_index_maps(atoms, index_array=index_array)
    return {
        int(atom_id): symbol
        for atom_id, symbol in zip(ids, atoms.get_chemical_symbols())
    }


def _subset(atoms: Atoms, atom_ids: Iterable[int], index_array="global_index") -> Atoms:
    """Slice ``atoms`` using persistent IDs rather than local ASE indices."""
    local = mfg.local_indices(atoms, atom_ids, index_array=index_array)
    return atoms[local]
